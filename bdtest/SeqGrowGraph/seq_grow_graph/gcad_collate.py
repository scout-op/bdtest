import torch
import torch.nn.functional as F
import numpy as np
from mmengine.dataset.utils import default_collate
from mmengine.registry import FUNCTIONS
from torch.nn.utils.rnn import pad_sequence
from .debug_utils import debug_print

VOCAB_PAD_ID = 0  # [PAD]
MODEL_VOCAB_SIZE = 576  # must match ARRNTRHead.num_center_classes


@FUNCTIONS.register_module()
def gcad_collate_fn(data_batch: list) -> dict:
    """为 GCAD 拓扑序列/掩码与几何数据定制的 collate_fn。

    - 提取每个样本 Det3DDataSample.metainfo 中的 'gt_topo_seq' 与 'gt_logits_mask' 并 padding
    - 组装为 inputs['gt_topo_seq_input']、inputs['gt_topo_seq_target']、inputs['gt_logits_mask']
    - 同时处理几何数据：'gt_nodes_xy'、'gt_geom_blocks'，并生成 mask
    - 其余 inputs 使用 default_collate
    """
    # 1) 分离 inputs 与 data_samples
    inputs_batch = []
    data_samples_list = []
    for item in data_batch:
        inputs_batch.append(item['inputs'])
        data_samples_list.append(item['data_samples'])

    # 2) 常规 collate（图像等）
    collated_inputs = default_collate(inputs_batch)
    debug_print('collate/batch', lambda: f"B={len(data_samples_list)}; base_keys={list(collated_inputs.keys())[:5]}")

    # 3) 收集并 padding 拓扑序列（Topo-AR）
    topo_seq_list = []
    logits_mask_list = []
    for ds in data_samples_list:
        topo_seq_list.append(torch.from_numpy(ds.metainfo['gt_topo_seq']))
        logits_mask_list.append(torch.from_numpy(ds.metainfo['gt_logits_mask']))

    padded_topo_seq = pad_sequence(topo_seq_list, batch_first=True, padding_value=VOCAB_PAD_ID)  # [B, Lmax]
    padded_logits_mask = pad_sequence(logits_mask_list, batch_first=True, padding_value=False)   # [B, Lmax, Vsmall]
    debug_print('collate/topo', lambda: f"padded_topo_seq={tuple(padded_topo_seq.shape)}, padded_logits_mask={tuple(padded_logits_mask.shape)}")

    # 切分输入与目标，以及对齐掩码
    inputs_seq = padded_topo_seq[:, :-1].contiguous()   # [B, L]
    target_seq = padded_topo_seq[:, 1:].contiguous()    # [B, L]
    logits_mask = padded_logits_mask[:, :-1, :].contiguous()  # [B, L, V_small]

    # 扩展到模型词表大小（与 ARRNTRHead 输出维度一致）
    if logits_mask.shape[-1] < MODEL_VOCAB_SIZE:
        pad = MODEL_VOCAB_SIZE - logits_mask.shape[-1]
        logits_mask = F.pad(logits_mask, (0, pad), value=False)
    debug_print('collate/topo2', lambda: (
        f"inputs_seq={tuple(inputs_seq.shape)}, target_seq={tuple(target_seq.shape)}, "
        f"logits_mask={tuple(logits_mask.shape)}, mask_mean={float(logits_mask.float().mean()):.4f}"
    ))

    assert inputs_seq.shape[1] == target_seq.shape[1] == logits_mask.shape[1], \
        f"Collate error: Seq({inputs_seq.shape}), Target({target_seq.shape}), Mask({logits_mask.shape}) length mismatch"

    # 放入 inputs
    collated_inputs['gt_topo_seq_input'] = inputs_seq
    collated_inputs['gt_topo_seq_target'] = target_seq
    collated_inputs['gt_logits_mask'] = logits_mask

    # 4) 收集并 padding 几何节点（Geom-Diffusion）
    nodes_xy_list = []  # [N_i, 2]
    nodes_mask_list = []  # [N_i]
    for ds in data_samples_list:
        nx = torch.from_numpy(ds.metainfo['gt_nodes_xy']).to(torch.float32)  # [N_i, 2]
        nodes_xy_list.append(nx)
        nodes_mask_list.append(torch.ones(nx.shape[0], dtype=torch.bool))

    if len(nodes_xy_list) == 0:
        collated_inputs['gt_nodes_xy'] = torch.zeros(0, 0, 2)
        collated_inputs['gt_nodes_mask'] = torch.zeros(0, 0, dtype=torch.bool)
    else:
        padded_nodes_xy = pad_sequence(nodes_xy_list, batch_first=True, padding_value=0.0)  # [B, N_max, 2]
        padded_nodes_mask = pad_sequence(nodes_mask_list, batch_first=True, padding_value=False)  # [B, N_max]
        collated_inputs['gt_nodes_xy'] = padded_nodes_xy
        collated_inputs['gt_nodes_mask'] = padded_nodes_mask
        debug_print('collate/nodes', lambda: f"nodes_xy={tuple(padded_nodes_xy.shape)}, nodes_mask={tuple(padded_nodes_mask.shape)}")

    # 5) 收集并 padding ADD_NODE 对齐索引（用于从 AR 隐状态中索引每个节点的状态）
    add_idx_list = []  # [N_i]
    for ds in data_samples_list:
        arr = ds.metainfo.get('gt_add_node_indices', None)
        if arr is None:
            arr = np.zeros((0,), dtype=np.int64)
        add_idx_list.append(torch.from_numpy(arr).to(torch.long))
    if len(add_idx_list) == 0:
        collated_inputs['gt_add_node_indices'] = torch.zeros(0, 0, dtype=torch.long)
    else:
        padded_add_idx = pad_sequence(add_idx_list, batch_first=True, padding_value=-1)  # [B, N_max]
        collated_inputs['gt_add_node_indices'] = padded_add_idx
        debug_print('collate/add_idx', lambda: f"add_idx={tuple(padded_add_idx.shape)}")

    # 6) 收集并 padding 几何块（可选）
    geom_blocks_tensors = []  # [M_i, 2+K_max]
    geom_blocks_masks = []    # [M_i]
    K_max = 0
    for ds in data_samples_list:
        blocks = ds.metainfo.get('gt_geom_blocks', [])
        for blk in blocks:
            K_max = max(K_max, int(len(blk['coeff'])))

    for ds in data_samples_list:
        blocks = ds.metainfo.get('gt_geom_blocks', [])
        Mi = len(blocks)
        if Mi == 0:
            geom_blocks_tensors.append(torch.zeros(0, 2 + K_max, dtype=torch.long))
            geom_blocks_masks.append(torch.zeros(0, dtype=torch.bool))
            continue
        t = torch.zeros(Mi, 2 + K_max, dtype=torch.long)
        m = torch.ones(Mi, dtype=torch.bool)
        for i, blk in enumerate(blocks):
            t[i, 0] = int(blk['src'])
            t[i, 1] = int(blk['dst'])
            coeff = torch.as_tensor(blk['coeff'], dtype=torch.long)
            if K_max > 0 and coeff.numel() > 0:
                t[i, 2:2 + coeff.numel()] = coeff
        geom_blocks_tensors.append(t)
        geom_blocks_masks.append(m)

    if len(geom_blocks_tensors) == 0:
        collated_inputs['gt_geom_blocks'] = torch.zeros(0, 0, 0, dtype=torch.long)
        collated_inputs['gt_geom_blocks_mask'] = torch.zeros(0, 0, dtype=torch.bool)
    else:
        padded_geom_blocks = pad_sequence(geom_blocks_tensors, batch_first=True, padding_value=0)  # [B, M_max, 2+K_max]
        padded_geom_blocks_mask = pad_sequence(geom_blocks_masks, batch_first=True, padding_value=False)  # [B, M_max]
        collated_inputs['gt_geom_blocks'] = padded_geom_blocks
        collated_inputs['gt_geom_blocks_mask'] = padded_geom_blocks_mask
        debug_print('collate/geom', lambda: f"K_max={K_max}; geom_blocks={tuple(padded_geom_blocks.shape)}, geom_mask={tuple(padded_geom_blocks_mask.shape)}")

    return {'inputs': collated_inputs, 'data_samples': data_samples_list}
