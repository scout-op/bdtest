import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from .debug_utils import debug_print


@MODELS.register_module()
class SimpleGeomDecoder(nn.Module):
    """
    A simple MLP-based diffusion "drawer" to regress node coordinates via denoising.
    Conditions on global BEV feature and per-node AR hidden state.
    """

    def __init__(self, bev_feat_dim: int, ar_hidden_dim: int, model_dim: int = 256,
                 coord_dim: int = 2, n_steps: int = 1000,
                 xy_normalize: bool = True, norm_mode: str = 'minus1_1'):
        super().__init__()
        self.model_dim = model_dim
        self.coord_dim = coord_dim
        self.n_steps = int(n_steps)
        self.xy_normalize = bool(xy_normalize)
        self.norm_mode = str(norm_mode)

        # Conditions
        self.bev_proj = nn.Linear(bev_feat_dim, model_dim)
        self.ar_proj = nn.Linear(ar_hidden_dim, model_dim)

        # Embeddings
        self.coord_embed = nn.Linear(coord_dim, model_dim)
        self.time_embed = nn.Embedding(self.n_steps, model_dim)

        # MLP backbone
        in_dim = model_dim * 3  # coord + ar + bev
        hid = model_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, coord_dim),  # predict noise on xy
        )

        self._loss = nn.MSELoss(reduction='none')

    def forward(self, noised_nodes_xy: torch.Tensor, timesteps: torch.Tensor,
                cond_bev_feat: torch.Tensor, cond_ar_hidden: torch.Tensor,
                nodes_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noised_nodes_xy: [B, N, 2]
            timesteps: [B]
            cond_bev_feat: [B, C] or [B, C, H, W]
            cond_ar_hidden: [B, N, C_ar]
            nodes_mask: [B, N] (bool)
        Returns:
            pred_noise: [B, N, 2]
        """
        B, N, _ = noised_nodes_xy.shape

        # time emb [B, D]
        t = timesteps.clamp(0, self.n_steps - 1).long()  # 非 in-place 版本，避免内存冲突
        t_emb = self.time_embed(t)  # [B, D]

        # coord emb [B, N, D]
        x_emb = self.coord_embed(noised_nodes_xy) + t_emb.unsqueeze(1)

        # bev global cond [B, D]
        if cond_bev_feat.dim() == 4:
            # [B, C, H, W] -> [B, C]
            cond_bev_feat = cond_bev_feat.mean(dim=(2, 3))
        bev_emb = self.bev_proj(cond_bev_feat).unsqueeze(1).expand(B, N, -1)  # [B, N, D]

        # ar per-node cond
        ar_emb = self.ar_proj(cond_ar_hidden)  # [B, N, D]

        feat = torch.cat([x_emb, ar_emb, bev_emb], dim=-1)  # [B, N, 3D]
        pred_noise = self.mlp(feat)  # [B, N, 2]
        return pred_noise

    def compute_loss(self, target_noise: torch.Tensor, pred_noise: torch.Tensor,
                     nodes_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked MSE: sum over valid nodes and coords, normalized by valid count*coord_dim.
        """
        loss = self._loss(pred_noise, target_noise)  # [B, N, 2]
        if nodes_mask is None:
            return loss.mean()
        mask = nodes_mask.unsqueeze(-1).float()
        loss = (loss * mask).sum()
        denom = mask.sum().clamp_min(1.0) * self.coord_dim
        return loss / denom

    def train_step(self, gt_nodes_xy: torch.Tensor, cond_bev_feat: torch.Tensor,
                   cond_ar_hidden: torch.Tensor, nodes_mask: torch.Tensor) -> torch.Tensor:
        """
        One DDPM-style denoising step training with a simple linear schedule.
        Args:
            gt_nodes_xy: [B, N, 2]
            cond_bev_feat: [B, C] or [B, C, H, W]
            cond_ar_hidden: [B, N, C_ar]
            nodes_mask: [B, N]
        Returns:
            masked MSE loss
        """
        device = gt_nodes_xy.device
        B = gt_nodes_xy.size(0)
        t = torch.randint(0, self.n_steps, (B,), device=device).long()
        noise = torch.randn_like(gt_nodes_xy)

        # Simplified alpha_bar schedule: linear
        alpha_bar = 1.0 - t.float() / max(self.n_steps - 1, 1)
        alpha_bar = alpha_bar.view(B, 1, 1)

        # Optional coordinate normalization using BEV H/W
        gt_for_noise = gt_nodes_xy
        H = W = None
        if self.xy_normalize and cond_bev_feat.dim() == 4:
            H, W = cond_bev_feat.shape[-2], cond_bev_feat.shape[-1]
            denom_x = max(W - 1, 1)
            denom_y = max(H - 1, 1)
            if self.norm_mode == 'minus1_1':
                x = gt_nodes_xy[..., 0] / denom_x * 2.0 - 1.0
                y = gt_nodes_xy[..., 1] / denom_y * 2.0 - 1.0
            else:  # 'zero_1'
                x = gt_nodes_xy[..., 0] / denom_x
                y = gt_nodes_xy[..., 1] / denom_y
            gt_for_noise = torch.stack([x, y], dim=-1)

        noised = torch.sqrt(alpha_bar) * gt_for_noise + torch.sqrt(1.0 - alpha_bar) * noise
        pred_noise = self.forward(noised, t, cond_bev_feat, cond_ar_hidden, nodes_mask)
        loss = self.compute_loss(noise, pred_noise, nodes_mask)

        # Debug ranges (raw and normalized if enabled)
        def _ranges_str():
            try:
                if nodes_mask is not None and nodes_mask.any():
                    raw = gt_nodes_xy[nodes_mask]
                    raw_min = raw.min(dim=0).values.tolist()
                    raw_max = raw.max(dim=0).values.tolist()
                else:
                    raw_min = ["NA", "NA"]; raw_max = ["NA", "NA"]
                if self.xy_normalize and cond_bev_feat.dim() == 4:
                    if nodes_mask is not None and nodes_mask.any():
                        nm = gt_for_noise[nodes_mask]
                        nmin = nm.min(dim=0).values.tolist(); nmax = nm.max(dim=0).values.tolist()
                    else:
                        nmin = ["NA", "NA"]; nmax = ["NA", "NA"]
                    return f"raw_min={raw_min}, raw_max={raw_max}, norm_min={nmin}, norm_max={nmax}, HxW={H}x{W}"
                else:
                    return f"raw_min={raw_min}, raw_max={raw_max}"
            except Exception:
                return "ranges=NA"

        debug_print('geom/train', lambda: (
            f"gt={tuple(gt_nodes_xy.shape)}, bev={tuple(cond_bev_feat.shape)}, "
            f"ar={tuple(cond_ar_hidden.shape)}, mask={tuple(nodes_mask.shape)}, loss={float(loss):.4f}, "
            f"{_ranges_str()}"
        ))
        return loss

    @torch.no_grad()
    def sample(self,
               cond_bev_feat: torch.Tensor,
               cond_ar_hidden: torch.Tensor,
               nodes_mask: torch.Tensor,
               steps: int = 50,
               return_denorm: bool = True) -> torch.Tensor:
        """
        Deterministic DDPM-like reverse sampling with a simple linear alpha_bar schedule.

        Args:
            cond_bev_feat: [B, C, H, W]
            cond_ar_hidden: [B, N, C_ar]
            nodes_mask: [B, N] (bool)
            steps: number of reverse steps (<= n_steps)
            return_denorm: if True and xy_normalize, map back to pixel/grid coords
        Returns:
            coords: [B, N, 2] (denormalized to [0,W-1]/[0,H-1] if enabled)
        """
        device = cond_ar_hidden.device
        B, N, _ = cond_ar_hidden.shape
        steps = int(max(1, min(steps, self.n_steps)))

        # init x_T ~ N(0, I)
        x = torch.randn(B, N, self.coord_dim, device=device)
        mask = nodes_mask.bool()

        # discrete reverse timesteps (ints) from n_steps-1 -> 0 with `steps` samples
        t_seq = torch.linspace(self.n_steps - 1, 0, steps, device=device).long()

        # Debug: Check inputs for NaN
        if torch.isnan(cond_bev_feat).any():
            debug_print('geom/sample', "NaN detected in cond_bev_feat!")
        if torch.isnan(cond_ar_hidden).any():
            debug_print('geom/sample', "NaN detected in cond_ar_hidden!")

        for i, tt in enumerate(t_seq):
            # If current timestep is 0, we are already at clean data (alpha_bar=1.0).
            # No further denoising needed, and formula would divide by zero.
            if tt == 0:
                break

            # predict noise at time t
            t_batch = tt.expand(B)
            eps = self.forward(x, t_batch, cond_bev_feat, cond_ar_hidden, mask)  # [B, N, 2]
            
            if torch.isnan(eps).any():
                debug_print('geom/sample', f"NaN detected in eps at step {i}, t={tt.item()}")
                # break # Let it flow to see if x becomes NaN

            # compute alpha_bar_t and previous
            denom = max(self.n_steps - 1, 1)
            # Fix: clamp alpha_bar_t to avoid 0.0 which causes division by zero in alpha_t
            alpha_bar_t = (1.0 - tt.float() / denom).clamp(min=1e-4)
            
            if i + 1 < len(t_seq):
                tt_prev = t_seq[i + 1]
                alpha_bar_prev = (1.0 - tt_prev.float() / denom).clamp(min=1e-4)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # avoid zeros
            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)
            
            # alpha_t = alpha_bar_t / alpha_bar_prev (Note: standard definition is alpha_bar_t = alpha_bar_{t-1} * alpha_t)
            # But here the schedule is defined directly on alpha_bar.
            # Let's derive alpha_t consistent with the schedule:
            # alpha_bar_t = alpha_bar_prev * alpha_t  => alpha_t = alpha_bar_t / alpha_bar_prev
            # Wait, t goes from T -> 0. 
            # t_seq[i] is current t, t_seq[i+1] is next step (smaller t, closer to 0).
            # So alpha_bar_t (current, noisy) < alpha_bar_prev (next, less noisy).
            # Actually, in diffusion notation:
            # t: current step (e.g. 999) -> alpha_bar_t is small
            # t-1: next step (e.g. 980) -> alpha_bar_{t-1} is larger
            # The formula x_{t-1} = ... uses alpha_t which is usually alpha_bar_t / alpha_bar_{t-1}
            # BUT your formula seems to use alpha_t as (alpha_bar_prev / alpha_bar_t) which would be > 1 ??
            
            # Let's look at the formula used:
            # x = (x - (1 - alpha_t) / ... * eps) / sqrt(alpha_t)
            # This looks like the standard DDPM posterior mean formula where alpha_t is the single-step noise level (close to 1).
            # Standard: alpha_bar_t = product(alpha_1...alpha_t)
            # So alpha_t = alpha_bar_t / alpha_bar_{t-1}
            
            # In your linear schedule: alpha_bar is linear w.r.t t.
            # alpha_bar(999) = 0, alpha_bar(998) > 0.
            
            # Let's stick to the fix of clamping alpha_bar_t to avoid 0.
            # And ensure alpha_t is calculated correctly as single step transition.
            # If we define alpha_t = alpha_bar_t / alpha_bar_{t-1} (where t-1 is the next step in generation, i.e. smaller noise)
            # Wait, t-1 has LESS noise, so alpha_bar_{t-1} is LARGER.
            # So alpha_t = alpha_bar_t / alpha_bar_{t-1} would be < 1. Correct.
            
            # Current code: alpha_t = (alpha_bar_prev / alpha_bar_t)
            # alpha_bar_prev corresponds to t_seq[i+1] (smaller t, larger alpha_bar).
            # So alpha_bar_prev > alpha_bar_t.
            # So alpha_t > 1. This is WRONG for the standard DDPM formula if alpha_t is meant to be < 1.
            
            # Let's assume the standard formula:
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps)
            # Here alpha_t should be < 1.
            # So it should be alpha_bar_t / alpha_bar_prev.
            
            alpha_t = (alpha_bar_t / alpha_bar_prev).clamp(min=1e-6, max=1.0)

            # estimate x0 and compute x_{t-1} (deterministic, sigma=0)
            # x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-6) * eps) / torch.sqrt(alpha_t)
            
            # Re-check the formula. 
            # If alpha_t is indeed alpha_bar_t / alpha_bar_prev, then it is < 1.
            # Then (1 - alpha_t) is positive.
            # This matches standard DDPM.
            
            x = (x - (1 - alpha_t) / sqrt_one_minus_ab_t * eps) / torch.sqrt(alpha_t)

            # enforce mask (invalid nodes keep zero)
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                x = x * m
            
            if torch.isnan(x).any():
                debug_print('geom/sample', f"NaN detected in x at step {i}, t={tt.item()}")
                break

        # Now x approximates x0 in normalized space if xy_normalize else raw space
        if self.xy_normalize and return_denorm and cond_bev_feat.dim() == 4:
            H, W = cond_bev_feat.shape[-2], cond_bev_feat.shape[-1]
            denom_x = max(W - 1, 1)
            denom_y = max(H - 1, 1)
            if self.norm_mode == 'minus1_1':
                x_denorm_x = (x[..., 0].clamp(-1, 1) + 1.0) * 0.5 * denom_x
                x_denorm_y = (x[..., 1].clamp(-1, 1) + 1.0) * 0.5 * denom_y
            else:  # 'zero_1'
                x_denorm_x = x[..., 0].clamp(0, 1) * denom_x
                x_denorm_y = x[..., 1].clamp(0, 1) * denom_y
            coords = torch.stack([x_denorm_x, x_denorm_y], dim=-1)
        else:
            coords = x

        # apply mask safety
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            coords = coords * m
            
        if torch.isnan(coords).any():
             debug_print('geom/sample', "NaN detected in final coords!")
             
        return coords
