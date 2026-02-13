# models/mamba_block.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class VisionMambaBlock(nn.Module):
    """
    [升级版] 支持 AdaLN 时间注入的 Vision Mamba Block
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        
        # 1. 基础 Norm
        self.norm = nn.LayerNorm(dim)
        
        # 2. [新增] AdaLN 调制层: 从 time_emb (dim) 映射到 2*dim (scale + shift)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        # 初始化为 0，让初始训练更稳定 (Zero-Init)
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        
        # 正向 Mamba
        self.mamba_fwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        # 反向 Mamba
        self.mamba_bwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, time_emb=None):
        # x: [B, C, H, W]
        # time_emb: [B, C] (与 x 的通道数 dim 相同)
        
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # --- [关键] AdaLN ---
        if time_emb is not None:
            # 计算 scale, shift: [B, 2*C] -> [B, 1, 2*C]
            style = self.adaLN_modulation(time_emb).unsqueeze(1)
            scale, shift = style.chunk(2, dim=2)
            # Modulate: x * (1 + scale) + shift
            x_norm = self.norm(x_flat) * (1 + scale) + shift
        else:
            x_norm = self.norm(x_flat)
        # -------------------

        # 双向扫描
        out_fwd = self.mamba_fwd(x_norm)
        out_bwd = self.mamba_bwd(x_norm.flip(1)).flip(1)
        
        out = out_fwd + out_bwd
        out = self.proj(out)
        
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return x + out