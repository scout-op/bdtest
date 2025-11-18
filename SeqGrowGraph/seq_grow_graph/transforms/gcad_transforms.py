import numpy as np
from typing import List

from mmdet3d.registry import TRANSFORMS
from ..debug_utils import debug_print


# --- 词表定义（静态 256 指针） ---
# [PAD]=0, [BOS]=1, [EOS]=2, [SEP]=3, [ADD_NODE]=4,
# [FROM]=5, [TO]=6, [NONE]=7, [NODE_0]=8..[NODE_255]=263
VOCAB = {
    'PAD': 0,
    'BOS': 1,
    'EOS': 2,
    'SEP': 3,
    'ADD_NODE': 4,
    'FROM': 5,
    'TO': 6,
    'NONE': 7,
    'NODE_PTR_OFFSET': 8,
    'MAX_NODES': 256,
}
VOCAB_SIZE = VOCAB['NODE_PTR_OFFSET'] + VOCAB['MAX_NODES']  # 8 + 256 = 264


@TRANSFORMS.register_module()
class TransformEventsToSeqAndMask:
    """将 `gt_topo_events` 转换为 AR 训练用的 token 序列与合法性掩码。

    输入（results 必须包含）：
      - gt_topo_events: List[dict]，由 TransformTopoGeomDualStreams 生成，
        每项包含：{'node':int, 'xy':[x,y], 'in_ptrs':List[int], 'out_ptrs':List[int]}。

    输出（写回 results）：
      - gt_topo_seq: np.ndarray[int64]，形如 [BOS, ... , EOS]
      - gt_logits_mask: np.ndarray[bool]，形状 [L, V]，L 与 gt_topo_seq 等长
        第 i 行约束 token[i] 预测 token[i+1] 的合法范围。
    """

    def __init__(self,
                 node_ptr_offset: int = VOCAB['NODE_PTR_OFFSET'],
                 max_nodes: int = VOCAB['MAX_NODES'],
                 vocab_size: int = VOCAB_SIZE):
        self.ptr_offset = node_ptr_offset
        self.max_nodes = max_nodes
        self.vocab_size = vocab_size

        # 预构基础掩码
        self.mask_allow_none_and_pointers = np.zeros(self.vocab_size, dtype=bool)
        self.mask_allow_none_and_pointers[VOCAB['NONE']] = True

        self.mask_allow_from = np.zeros(self.vocab_size, dtype=bool)
        self.mask_allow_from[VOCAB['FROM']] = True

        self.mask_allow_to = np.zeros(self.vocab_size, dtype=bool)
        self.mask_allow_to[VOCAB['TO']] = True

        self.mask_allow_sep = np.zeros(self.vocab_size, dtype=bool)
        self.mask_allow_sep[VOCAB['SEP']] = True

        self.mask_allow_add_or_eos = np.zeros(self.vocab_size, dtype=bool)
        self.mask_allow_add_or_eos[VOCAB['ADD_NODE']] = True
        self.mask_allow_add_or_eos[VOCAB['EOS']] = True

        self.mask_allow_pad = np.zeros(self.vocab_size, dtype=bool)
        self.mask_allow_pad[VOCAB['PAD']] = True

    def __call__(self, results: dict) -> dict:
        topo_events: List[dict] = results.get('gt_topo_events', [])

        gt_topo_seq = [VOCAB['BOS']]
        add_node_positions = []  # 记录每个节点对应的 [ADD_NODE] 在序列中的位置
        logits_mask_list = []  # 每步的合法性掩码，约束下一 token

        # BOS 步：只能预测 [ADD_NODE]/[EOS]
        logits_mask_list.append(self.mask_allow_add_or_eos)

        current_node_count = 0
        for event in topo_events:
            if current_node_count >= self.max_nodes:
                break

            # 1) [ADD_NODE]
            gt_topo_seq.append(VOCAB['ADD_NODE'])
            add_node_positions.append(len(gt_topo_seq) - 1)
            logits_mask_list.append(self.mask_allow_from)

            # 2) [FROM]
            gt_topo_seq.append(VOCAB['FROM'])
            mask_from = self.mask_allow_none_and_pointers.copy()
            if current_node_count > 0:
                mask_from[self.ptr_offset: self.ptr_offset + current_node_count] = True
            logits_mask_list.append(mask_from)

            # FROM 的内容（允许多个父指针或 NONE 结束）
            if not event['in_ptrs']:
                gt_topo_seq.append(VOCAB['NONE'])
                # NONE -> 只能预测 [TO]
                logits_mask_list.append(self.mask_allow_to)
            else:
                for ptr in event['in_ptrs']:
                    if ptr < self.max_nodes:
                        gt_topo_seq.append(self.ptr_offset + ptr)
                        mask_from_multi = mask_from.copy()
                        mask_from_multi[VOCAB['TO']] = True  # 允许结束 FROM
                        logits_mask_list.append(mask_from_multi)

            # 3) [TO]
            gt_topo_seq.append(VOCAB['TO'])
            mask_to = self.mask_allow_none_and_pointers.copy()
            if current_node_count > 0:
                mask_to[self.ptr_offset: self.ptr_offset + current_node_count] = True
            logits_mask_list.append(mask_to)

            # TO 的内容（允许多个子指针或 NONE 结束）
            if not event['out_ptrs']:
                gt_topo_seq.append(VOCAB['NONE'])
                # NONE -> 只能预测 [SEP]
                logits_mask_list.append(self.mask_allow_sep)
            else:
                for ptr in event['out_ptrs']:
                    if ptr < self.max_nodes:
                        gt_topo_seq.append(self.ptr_offset + ptr)
                        mask_to_multi = mask_to.copy()
                        mask_to_multi[VOCAB['SEP']] = True  # 允许结束 TO
                        logits_mask_list.append(mask_to_multi)

            # 4) [SEP]
            gt_topo_seq.append(VOCAB['SEP'])
            logits_mask_list.append(self.mask_allow_add_or_eos)

            current_node_count += 1

        # 5) [EOS]
        if gt_topo_seq[-1] != VOCAB['EOS']:
            gt_topo_seq.append(VOCAB['EOS'])
        logits_mask_list.append(self.mask_allow_pad)

        # 输出为 numpy，保证 Pack 时的类型一致
        results['gt_topo_seq'] = np.array(gt_topo_seq, dtype=np.int64)
        results['gt_logits_mask'] = np.stack(logits_mask_list, axis=0).astype(bool)
        results['gt_add_node_indices'] = np.array(add_node_positions, dtype=np.int64)
        assert len(results['gt_topo_seq']) == len(results['gt_logits_mask']), \
            f"Seq len {len(results['gt_topo_seq'])} != Mask len {len(results['gt_logits_mask'])}"
        debug_print('xform/seq', lambda: (
            f"events={len(topo_events)}; seq_len={len(results['gt_topo_seq'])}; "
            f"mask={results['gt_logits_mask'].shape}; add_pos={len(results['gt_add_node_indices'])}"
        ))
        return results
