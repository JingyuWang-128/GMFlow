# models/mamba_block.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class VisionMambaBlock(nn.Module):
    """
    针对 2D 图像优化的双向 Mamba 块。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # 正向 Mamba
        self.mamba_fwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        # 反向 Mamba
        self.mamba_bwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.norm(x_flat)

        # 双向扫描
        out_fwd = self.mamba_fwd(x_norm)
        out_bwd = self.mamba_bwd(x_norm.flip(1)).flip(1)
        
        out = out_fwd + out_bwd
        out = self.proj(out)
        
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return x + out