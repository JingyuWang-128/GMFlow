# models/gen_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_block import VisionMambaBlock

class TriStreamMambaUNet(nn.Module):
    def __init__(self, in_channels=3, dim=128, secret_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        self.text_proj = nn.Linear(768, dim)

        self.secret_proj = nn.Linear(secret_dim, dim * 4)
        
        # [新增] 防止特征注入数值爆炸的关键层
        self.secret_norm = nn.GroupNorm(32, dim * 4)
        self.secret_scale = nn.Parameter(torch.zeros(1)) 

        self.inc = nn.Conv2d(in_channels, dim, 3, padding=1)

        self.down1 = nn.Sequential(
            VisionMambaBlock(dim),
            nn.Conv2d(dim, dim * 2, 4, 2, 1)
        )

        self.down2 = nn.Sequential(
            VisionMambaBlock(dim * 2),
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)
        )

        self.bot = VisionMambaBlock(dim * 4)

        self.up1_conv = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.up1_block = VisionMambaBlock(dim * 2)

        self.up2_conv = nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1)
        self.up2_block = VisionMambaBlock(dim)

        self.outc = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x_t, t, text_emb, secret_emb):

        t_emb = self.time_mlp(t.view(-1, 1))
        txt_emb = self.text_proj(text_emb.mean(dim=1))
        cond = (t_emb + txt_emb).unsqueeze(-1).unsqueeze(-1)

        x1 = self.inc(x_t) + cond
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        h_struc = x3

        # 调整 secret_emb 大小以匹配 bottleneck
        s_feat = torch.nn.functional.interpolate(
            secret_emb,
            size=x3.shape[2:]
        )

        # [新增] 归一化处理，防止数值爆炸
        s_feat = self.secret_proj(s_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        s_feat = self.secret_norm(s_feat)
        
        # [新增] 使用 scale 控制注入强度 (h_tex = h_struc + s_feat * (1 + scale))
        h_tex = h_struc + s_feat * (1.0 + self.secret_scale)

        x = self.bot(h_tex)

        x = self.up1_conv(x)
        x = x + x2
        x = self.up1_block(x)

        x = self.up2_conv(x)
        x = x + x1
        x = self.up2_block(x)

        v_pred = self.outc(x)

        return v_pred, h_struc, h_tex

class MambaDecoderHead(nn.Module):
    def __init__(self, in_channels=3, dim=128,
                 num_quantizers=4, codebook_size=1024):
        super().__init__()

        self.num_q = num_quantizers
        self.cb_size = codebook_size

        self.net = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, dim, 4, 2, 1),
            nn.ReLU(),
            VisionMambaBlock(dim),

            # 128 -> 64
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.ReLU(),
            VisionMambaBlock(dim * 2),
        )

        self.head = nn.Conv2d(
            dim * 2,
            num_quantizers * codebook_size,
            1
        )

    # [修复关键] 这里增加了 target_shape 参数
    def forward(self, x, target_shape=None):
        """
        Args:
            x: Input tensor
            target_shape: (H, W) tuple. 如果提供，将强制调整输出分辨率以匹配 RQ-VAE。
        """
        feat = self.net(x) # [B, dim*2, H_feat, W_feat]

        # [Critical Fix] 动态分辨率对齐
        # 确保 Decoder 输出的空间维度与 RQ-VAE 的 Latent Code 严格一致
        if target_shape is not None:
            if feat.shape[-2:] != target_shape:
                feat = F.adaptive_avg_pool2d(feat, target_shape)

        logits = self.head(feat) # [B, Q*K, H_target, W_target]
        B, _, H, W = logits.shape

        # [Critical Fix] 显式维度拆分，防止 view 错位
        # 1. 拆分 Quantizers 和 Codebook
        logits = logits.view(B, self.num_q, self.cb_size, H, W)
        
        # 2. 展平空间维度 (H, W) -> N
        # 顺序必须是 (B, Q, K, N) 以匹配 Loss 函数中的处理
        logits = logits.view(B, self.num_q, self.cb_size, -1)

        return logits