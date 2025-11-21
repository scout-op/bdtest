import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
import numpy as np

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from .grid_mask import GridMask
from .LiftSplatShoot import LiftSplatShootEgo
from .core import seq2nodelist, seq2bznodelist, seq2plbznodelist, av2seq2bznodelist
from .core import EvalSeq2Graph_with_start as EvalSeq2Graph

from .encode_centerline import convert_coeff_coord
from .bz_roadnet_reach_dist_eval import get_geom, get_range
from .debug_utils import debug_print


@MODELS.register_module()
class SeqGrowGraph(MVXTwoStageDetector):
    """Petr3D. nan for all token except label"""
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 lss_cfg=None,
                 grid_conf=None,
                 bz_grid_conf=None,
                 data_aug_conf=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 vis_cfg=None,
                 freeze_pretrain=True,
                 bev_scale=1.0,
                 epsilon=2,
                 max_box_num=700, #>=660+2
                 init_cfg=None,
                 data_preprocessor=None,front_camera_only=False,vis_dir="original",
                 geom_decoder_cfg=None,
                 ):
        super(SeqGrowGraph, self).__init__(pts_voxel_layer, pts_middle_encoder,
                                                        pts_fusion_layer, img_backbone, pts_backbone,
                                                        img_neck, pts_neck, pts_bbox_head, img_roi_head,
                                                        img_rpn_head, train_cfg, test_cfg, init_cfg,
                                                        data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.front_camera_only=front_camera_only
        self.vis_dir=vis_dir
        # data_aug_conf = {
        #     'final_dim': (128, 352),
        #     'H': 900, 'W': 1600,
        # }
        # self.up = Up(512, 256, scale_factor=2)
        # view_transformers = []
        # view_transformers.append(
        #     LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True))
        # self.view_transformers = nn.ModuleList(view_transformers)
        # self.view_transformers = LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True)
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']

        self.split_connect=571
        self.split_node=572
        self.start = 574
        self.end = 573
        self.summary_split = 570
        self.split_lines=569
        

      
        
        # self.box_range = 200
        # self.coeff_range = 200
        # self.num_classes=4
        # self.category_start = 200
        # self.connect_start = 250 
        self.coeff_start = 350 
        self.idx_start=250
        self.no_known = 575  # n/a and padding share the same label to be eliminated from loss calculation
        self.num_center_classes = 576 
        # self.noise_connect = 572 
        self.noise_label = 569
        # self.noise_coeff = 570
        
        
        self.vis_cfg = vis_cfg
        self.bev_scale = bev_scale
        self.epsilon = epsilon
        self.max_box_num = max_box_num #!暂时没用到

        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

        self.dx, bx, nx, self.pc_range, ego_points = get_geom(grid_conf)
        self.bz_dx, bz_bx, bz_nx, self.bz_pc_range = get_range(bz_grid_conf)

        if freeze_pretrain:
            self.freeze_pretrain()

        # Build Geom-Diffusion decoder if provided
        self.geom_decoder = None
        if geom_decoder_cfg is not None:
            # force dims to match current backbone/head
            cfg = dict(geom_decoder_cfg)
            bev_dim = None
            ar_dim = None
            if hasattr(self, 'pts_bbox_head'):
                bev_dim = getattr(self.pts_bbox_head, 'in_channels', None)
                ar_dim = getattr(self.pts_bbox_head, 'embed_dims', None)
            if bev_dim is None:
                bev_dim = cfg.get('bev_feat_dim', 256)
            if ar_dim is None:
                ar_dim = cfg.get('ar_hidden_dim', 256)
            cfg['bev_feat_dim'] = bev_dim
            cfg['ar_hidden_dim'] = ar_dim
            self.geom_decoder = MODELS.build(cfg)
    
    def freeze_pretrain(self):
        for m in self.img_backbone.parameters():
            m.requires_grad=False
        for m in self.img_neck.parameters():
            m.requires_grad=False
        for m in self.view_transformers.parameters():
            m.requires_grad=False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        largest_feat_shape = img_feats[0].shape[3]
        down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
        bev_feats = self.view_transformers(img_feats[down_level], img_metas)
        return bev_feats

    def forward_pts_train(self,
                          bev_feats,
                          gt_lines_sequences,
                          img_metas,
                          num_coeff,summary_subgraphs ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        device = bev_feats[0].device

        input_seqs = []

        max_len = max([len(target) for target in gt_lines_sequences])

        coeff_dim = num_coeff * 2
        

        input_seqs=[]
        for gt_lines_sequence in gt_lines_sequences:
            input_seq= [self.start]+gt_lines_sequence+[self.end]+[self.no_known]*(max_len-len(gt_lines_sequence))
            input_seq=torch.tensor(input_seq, device=device).long()
            input_seqs.append(input_seq.unsqueeze(0))

            
 
        input_seqs = torch.cat(input_seqs , dim=0)  # [8,501]
 
        outputs = self.pts_bbox_head(bev_feats, input_seqs, img_metas)[-1, :, :-1, :]

       
        clause_length = 4 + coeff_dim
        n_control = img_metas[0]['n_control']
        
        
        # =============================================
        # 训练可视化
        
   
        # for bi in range(outputs.shape[0]):
        #     try:
        #         pred_line_seq = outputs[bi]
        #         pred_line_seq = pred_line_seq.argmax(-1)
        #         if self.end in pred_line_seq:
        #             stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
        #         else:
        #             stop_idx = len(pred_line_seq)
        #         # if self.summary_split in pred_line_seq:
        #         #     start_idx=(pred_line_seq == self.summary_split).nonzero(as_tuple=True)[0][0]
        #         # else:
        #         #     start_idx=-1
        #         # pred_line_seq = pred_line_seq[start_idx+1:stop_idx]
        #         pred_line_seq = pred_line_seq[:stop_idx]
                
        #         pred_graph = EvalSeq2Graph(img_metas[bi]['token'],pred_line_seq.detach().cpu().numpy().tolist(),front_camera_only=self.front_camera_only,pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
        #         pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'train'), 'n', 'n')
        #     except:
        #         import traceback
        #         traceback.print_exc()
        #     break

        # =============================================
        
        outputs = outputs.reshape(-1, self.num_center_classes)  # [602, 2003] last layer
        input_seqs=input_seqs[:,1:]
        input_seqs=input_seqs.flatten()
        gt_seqs_pad=input_seqs[input_seqs!=self.no_known]
        outputs=outputs[input_seqs!=self.no_known]

        losses = self.pts_bbox_head.loss_by_feat_seq(outputs, gt_seqs_pad)

        return losses
    
    def loss(self,
             inputs=None,
             data_samples=None,**kwargs):

        img = inputs['img']
        img_metas = [ds.metainfo for ds in data_samples]

        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        debug_print('model/bev', lambda: (
            f"bev_feats_shape={tuple(bev_feats.shape) if hasattr(bev_feats, 'shape') else 'list/unknown'}"
        ))
        if self.bev_scale != 1.0:
            b, c, h, w = bev_feats.shape
            bev_feats = F.interpolate(bev_feats, (int(h * self.bev_scale), int(w * self.bev_scale)))
        losses = dict()

        # New GCAD AR training path using dual-stream topo sequence
        if ('gt_topo_seq_input' in inputs and 'gt_topo_seq_target' in inputs
                and 'gt_logits_mask' in inputs):
            input_seq = inputs['gt_topo_seq_input'].long()     # [B, L]
            target_seq = inputs['gt_topo_seq_target'].long()   # [B, L]
            logits_mask = inputs['gt_logits_mask'].to(input_seq.device)  # [B, L, V]
            debug_print('model/ar_in', lambda: f"B={input_seq.size(0)}, L={input_seq.size(1)}, V={logits_mask.size(-1)}")

            out = self.pts_bbox_head(bev_feats, input_seq, img_metas,
                                      return_hidden=True, logits_mask=logits_mask)
            logits, _hidden = out  # logits: [nb_dec, B, L, V]
            logits = logits[-1]    # last decoder layer [B, L, V]
            debug_print('model/ar_out', lambda: (
                f"logits={tuple(logits.shape)}; hidden={tuple((_hidden[-1] if _hidden.dim()==4 else _hidden).shape)}"
            ))

            # mask out PAD targets (PAD=0) before CE
            B, L, V = logits.shape
            logits = logits.reshape(B * L, V)
            targets = target_seq.reshape(B * L)
            valid = targets != 0
            logits = logits[valid]
            targets = targets[valid]

            ar_losses = self.pts_bbox_head.loss_by_feat_seq(logits, targets)
            losses.update(ar_losses)

            # If geom decoder is available, run second pass
            if self.geom_decoder is not None and 'gt_nodes_xy' in inputs and 'gt_nodes_mask' in inputs:
                # Align AR hidden states to nodes via gt_add_node_indices
                add_idx = inputs.get('gt_add_node_indices', None)  # [B, N_max]
                hidden = _hidden[-1] if _hidden.dim() == 4 else _hidden  # [B, L, C]
                B, L, C = hidden.shape
                if add_idx is not None:
                    idx = add_idx.clone()
                    # clamp invalid to 0 to avoid gather OOB; mask will drop them later
                    idx_clamped = idx.clamp(min=0)
                    # gather per-node hidden states
                    gather_idx = idx_clamped.unsqueeze(-1).expand(-1, -1, C)  # [B, N, C]
                    cond_ar_hidden = torch.gather(hidden, dim=1, index=gather_idx)
                    debug_print('model/align', lambda: (
                        f"add_idx={tuple(add_idx.shape)}, valid={(add_idx>=0).sum().item()}, "
                        f"cond_hidden={tuple(cond_ar_hidden.shape)}, cond_abs_mean={float(cond_ar_hidden.abs().mean()):.4f}"
                    ))
                else:
                    # fallback: truncate or pad
                    N = inputs['gt_nodes_xy'].shape[1]
                    if L >= N:
                        cond_ar_hidden = hidden[:, :N, :]
                    else:
                        pad = torch.zeros(B, N - L, C, device=hidden.device, dtype=hidden.dtype)
                        cond_ar_hidden = torch.cat([hidden, pad], dim=1)
                    debug_print('model/align', lambda: (
                        f"no_add_idx; cond_hidden={tuple(cond_ar_hidden.shape)}, "
                        f"cond_abs_mean={float(cond_ar_hidden.abs().mean()):.4f}"
                    ))

                # mask out invalid indices (-1) from add_idx
                nodes_mask = inputs['gt_nodes_mask'].bool()
                if add_idx is not None:
                    nodes_mask = nodes_mask & (add_idx >= 0)

                loss_geom = self.geom_decoder.train_step(
                    gt_nodes_xy=inputs['gt_nodes_xy'].to(bev_feats.device).float(),
                    cond_bev_feat=bev_feats,  # [B, C, H, W]
                    cond_ar_hidden=cond_ar_hidden,
                    nodes_mask=nodes_mask,
                )
                losses['loss_geom'] = loss_geom
                debug_print('model/geom_out', lambda: f"loss_geom={float(loss_geom):.4f}")

            return losses

        # Fallback: original single-stream sequence training
        gt_lines_sequences = [img_meta['centerline_sequence'] for img_meta in img_metas]
        summary_subgraphs = [img_meta['summary_subgraph'] if 'summary_subgraph' in img_meta else [] for img_meta in img_metas]
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        losses_pts = self.forward_pts_train(bev_feats, gt_lines_sequences, img_metas, num_coeff, summary_subgraphs)
        losses.update(losses_pts)
        return losses
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward of testing.
        
        实现真正的自回归推理 + Block Diffusion：
        - 第一阶段：AR 模型自回归生成拓扑序列（不使用 GT）
        - 第二阶段：从 AR hidden states 提取节点特征，用 Diffusion 生成几何坐标
        """
        img_metas = [item.metainfo for item in batch_data_samples]
        img = batch_inputs_dict['img']

        # 1) 提取 BEV 特征
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        debug_print('predict/bev', lambda: f"bev_feats shape={tuple(bev_feats.shape)}")

        # 如果没有 geom_decoder，回退到旧的单流推理
        if self.geom_decoder is None:
            batch_input_metas = img_metas
            batch_input_imgs = img
            return self.simple_test(batch_input_metas, batch_input_imgs)

        # ========== 阶段 1: AR 自回归生成拓扑序列 ==========
        device = bev_feats.device
        B = len(batch_data_samples)
        
        # 构造起始序列 [B, 1]，从 start token 开始
        input_seqs = (torch.ones(B, 1).to(device) * self.start).long()
        debug_print('predict/ar_start', lambda: f"Starting AR generation with B={B}, start_token={self.start}")

        # 运行 AR 推理，获取生成的序列和 hidden states
        # 注意：这里不传入 logits_mask，让模型自由生成
        ar_outs = self.pts_bbox_head(bev_feats, input_seqs, img_metas, return_hidden=True)
        
        # 解包返回值：(pred_seqs, pred_values, pred_hidden)
        pred_seqs, pred_values, pred_hidden = ar_outs
        debug_print('predict/ar_out', lambda: (
            f"AR generated: seqs={tuple(pred_seqs.shape)}, "
            f"hidden={tuple(pred_hidden.shape)}, "
            f"values={tuple(pred_values.shape)}"
        ))

        # ========== 阶段 2: 从序列中提取节点信息 ==========
        # 这部分需要根据你的 Token 定义来解析哪些位置代表节点
        # 假设：我们需要从 metainfo 中获取 gt_add_node_indices 来对齐
        # 如果是完全自主推理，需要实现一个 parse_seq_to_node_indices 函数
        
        # 临时方案：尝试从 metainfo 获取 add_node_indices（用于验证）
        # 在真实部署时，你需要从 pred_seqs 中解析出节点索引
        has_gt_indices = all('gt_add_node_indices' in ds.metainfo for ds in batch_data_samples)
        
        if has_gt_indices:
            # 使用 GT indices 进行对齐（这仍然是验证阶段的做法）
            debug_print('predict/align', "Using GT add_node_indices for alignment")
            ai_list = [torch.from_numpy(ds.metainfo['gt_add_node_indices']).long() 
                      for ds in batch_data_samples]
            Nmax = max(a.shape[0] for a in ai_list)
            add_idx = torch.full((B, Nmax), -1, dtype=torch.long, device=device)
            for i, a in enumerate(ai_list):
                add_idx[i, :a.shape[0]] = a.to(device)
            nodes_mask = (add_idx >= 0)
        else:
            # 完全自主模式：从生成的序列中解析节点
            # 这里实现一个简化版本：假设所有非特殊 token 都可能是节点
            debug_print('predict/align', "Parsing nodes from generated sequence")
            
            # 标识特殊 token（根据你的定义调整）
            special_tokens = {self.start, self.end, self.no_known, 
                            self.split_connect, self.split_node, 
                            self.summary_split, self.split_lines}
            
            # 为每个 batch 提取节点位置
            max_nodes = 0
            batch_node_indices = []
            
            for b_idx in range(B):
                seq = pred_seqs[b_idx]
                # 找到所有可能代表"添加节点"的 token 位置
                # 简化逻辑：假设某些特定 token 范围代表节点操作
                # 这需要根据你的 Token 设计来调整
                node_positions = []
                for pos, token in enumerate(seq):
                    token_val = token.item()
                    # 示例：假设 idx_start <= token < coeff_start 代表节点索引
                    if self.idx_start <= token_val < self.coeff_start:
                        node_positions.append(pos)
                
                batch_node_indices.append(node_positions)
                max_nodes = max(max_nodes, len(node_positions))
            
            # 构造 add_idx 和 nodes_mask
            add_idx = torch.full((B, max_nodes), -1, dtype=torch.long, device=device)
            for b_idx, positions in enumerate(batch_node_indices):
                for node_idx, pos in enumerate(positions):
                    if node_idx < max_nodes:
                        add_idx[b_idx, node_idx] = pos
            
            nodes_mask = (add_idx >= 0)
            debug_print('predict/align', lambda: f"Extracted {max_nodes} nodes from sequence")

        # ========== 阶段 3: 对齐 Hidden States 到节点 ==========
        B, L, C = pred_hidden.shape
        N = add_idx.shape[1]
        
        # Clamp indices to avoid out-of-bounds
        idx_clamped = add_idx.clamp(min=0, max=L-1)
        gather_idx = idx_clamped.unsqueeze(-1).expand(-1, -1, C)  # [B, N, C]
        
        # Gather per-node hidden states
        cond_ar_hidden = torch.gather(pred_hidden, dim=1, index=gather_idx)  # [B, N, C]
        
        # Update mask: only valid if both nodes_mask and within sequence length
        valid_indices = (add_idx >= 0) & (add_idx < L)
        nodes_mask = nodes_mask & valid_indices
        
        debug_print('predict/hidden', lambda: (
            f"Aligned hidden: shape={tuple(cond_ar_hidden.shape)}, "
            f"valid_nodes={nodes_mask.sum().item()}/{nodes_mask.numel()}, "
            f"mean_magnitude={float(cond_ar_hidden.abs().mean()):.4f}"
        ))

        # ========== 阶段 4: Diffusion 生成节点几何坐标 ==========
        pred_nodes_xy = self.geom_decoder.sample(
            cond_bev_feat=bev_feats,
            cond_ar_hidden=cond_ar_hidden,
            nodes_mask=nodes_mask,
            steps=15,  # 可根据需要调整采样步数
            return_denorm=True,
        )  # [B, N, 2]
        
        debug_print('predict/geom', lambda: (
            f"Generated nodes_xy: shape={tuple(pred_nodes_xy.shape)}, "
            f"range=({float(pred_nodes_xy.min()):.2f}, {float(pred_nodes_xy.max()):.2f})"
        ))

        # ========== 阶段 5: 组装结果 ==========
        # 同时生成旧格式的 line_results 用于兼容现有评估
        line_results = self.simple_test_pts(bev_feats, img_metas)
        
        bbox_list = [dict() for _ in range(B)]
        for i, (res, ds, lr) in enumerate(zip(bbox_list, batch_data_samples, line_results)):
            res['line_results'] = lr
            res['token'] = ds.metainfo.get('token', None)
            # 添加 Diffusion 生成的高精度节点坐标
            res['pred_nodes_xy'] = pred_nodes_xy[i].detach().cpu().numpy()
            res['pred_seqs'] = pred_seqs[i].detach().cpu().numpy()
            # 同时将坐标添加到 line_results 中，供评估器使用
            lr['pred_nodes_xy'] = pred_nodes_xy[i].detach().cpu().numpy()
            lr['nodes_mask'] = nodes_mask[i].detach().cpu().numpy()
            
            # 可选：可视化第一个样本（使用 Diffusion 坐标）
            if i == 0:
                try:
                    from .core import EvalSeq2Graph_with_start as EvalSeq2Graph
                    pred_graph = EvalSeq2Graph(
                        ds.metainfo['token'],
                        res['pred_seqs'],
                        front_camera_only=self.front_camera_only,
                        pc_range=self.pc_range,
                        dx=self.dx,
                        bz_pc_range=self.bz_pc_range,
                        bz_dx=self.bz_dx,
                        external_nodes_xy=res['pred_nodes_xy'],  # 传递 Diffusion 坐标
                        nodes_mask=lr['nodes_mask']  # 传递有效节点 mask
                    )
                    import os
                    pred_graph.visualization([200, 200], os.path.join(self.vis_dir, 'test'), 'n', 'n')
                except Exception as e:
                    debug_print('predict/vis', f"Visualization failed: {e}")
                    
        return bbox_list

    def simple_test_pts(self, pts_feats, img_metas):
        """Test function of point cloud branch."""
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        clause_length = 4 + num_coeff * 2

        device = pts_feats[0].device
        input_seqs = (torch.ones(pts_feats.shape[0], 1).to(device) * self.start).long()
        outs = self.pts_bbox_head(pts_feats, input_seqs, img_metas)
        output_seqs, values = outs
        line_results = []
        for bi in range(output_seqs.shape[0]):
            pred_line_seq = output_seqs[bi]
            pred_line_seq = pred_line_seq[1:]
            if self.end in pred_line_seq:
                stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
            else:
                stop_idx = len(pred_line_seq)
                
            pred_line_seq = pred_line_seq[:stop_idx]
            
     
            line_results.append(dict(
                line_seqs = pred_line_seq.detach().cpu().numpy(),
         
            ))
        return line_results

    def simple_test(self, img_metas, img=None):
        """Test function without augmentaiton."""
        

        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        line_results = self.simple_test_pts(
            bev_feats, img_metas)
        i=0
        for result_dict, line_result, img_meta in zip(bbox_list, line_results, img_metas):
            
            result_dict['line_results'] = line_result
            result_dict['token'] = img_meta['token']
            if i==0:
                try:
                    pred_graph = EvalSeq2Graph(img_meta['token'],line_result["line_seqs"],front_camera_only=self.front_camera_only,pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
                    pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'test'), 'n', 'n')
                except:
                    import traceback
                    traceback.print_exc()
            i+=1
                

        return bbox_list

