# models/gen_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_block import VisionMambaBlock

class SecretModulator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid() 
        )
        self.shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self, x, s_feat):
        B, C, H, W = x.shape
        s_avg = s_feat.mean(dim=[2, 3])
        scale = self.scale(s_avg).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(s_avg).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift + s_feat

class TriStreamMambaUNet(nn.Module):
    def __init__(self, in_channels=3, dim=128, secret_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text_proj = nn.Linear(768, dim)
        self.secret_proj = nn.Sequential(
            nn.Conv2d(secret_dim, dim * 4, 1),
            nn.GroupNorm(32, dim * 4),
            nn.SiLU()
        )
        self.modulator = SecretModulator(dim * 4)

        self.inc = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.down1 = nn.Sequential(VisionMambaBlock(dim), nn.Conv2d(dim, dim * 2, 4, 2, 1))
        self.down2 = nn.Sequential(VisionMambaBlock(dim * 2), nn.Conv2d(dim * 2, dim * 4, 4, 2, 1))
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

        s_feat = F.interpolate(secret_emb, size=h_struc.shape[2:])
        s_feat = self.secret_proj(s_feat)
        h_tex = self.modulator(h_struc, s_feat)

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
                 num_quantizers=4, embed_dim=256):
        """
        [关键修改] 输出 Embedding 用于检索，而不是 Logits
        """
        super().__init__()
        self.num_q = num_quantizers
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, dim, 4, 2, 1), 
            nn.ReLU(),
            VisionMambaBlock(dim),
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.ReLU(),
            VisionMambaBlock(dim * 2),
        )
        # 输出特征维度：Num_Quantizers * Embed_Dim (4 * 256 = 1024)
        # 而不是之前的 4 * 1024 (Logits)
        self.head = nn.Conv2d(dim * 2, num_quantizers * embed_dim, 1)

    def forward(self, x, target_shape=None):
        feat = self.net(x) 
        if target_shape is not None:
            if feat.shape[-2:] != target_shape:
                feat = F.adaptive_avg_pool2d(feat, target_shape)

        out = self.head(feat) 
        B, _, H, W = out.shape
        # Reshape to [B, Q, C, H, W]
        out = out.view(B, self.num_q, self.embed_dim, H, W)
        return out