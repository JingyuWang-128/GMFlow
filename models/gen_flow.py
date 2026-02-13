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
        
        # 时间编码层
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim) # 输出维度必须与 Block 的 dim 一致
        )
        
        # 文本编码
        self.text_proj = nn.Linear(768, dim)

        # 秘密信息编码
        self.secret_proj = nn.Sequential(
            nn.Conv2d(secret_dim, dim * 4, 1),
            nn.GroupNorm(32, dim * 4),
            nn.SiLU()
        )
        self.modulator = SecretModulator(dim * 4)

        # --- Encoder ---
        self.inc = nn.Conv2d(in_channels, dim, 3, padding=1)
        # 这里的 Block 现在需要 dim=128
        self.enc_block1 = VisionMambaBlock(dim) 
        self.down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        
        self.enc_block2 = VisionMambaBlock(dim * 2)
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)

        # --- Bottleneck ---
        self.bot = VisionMambaBlock(dim * 4)

        # --- Decoder ---
        self.up1_conv = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.up1_block = VisionMambaBlock(dim * 2)
        
        self.up2_conv = nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1)
        self.up2_block = VisionMambaBlock(dim)

        self.outc = nn.Conv2d(dim, 3, 3, padding=1)
        
        # 为了适配不同维度的 AdaLN，需要将 t_emb 投影到对应维度
        self.time_proj_dim = nn.Identity()         # for dim
        self.time_proj_dim2 = nn.Linear(dim, dim*2) # for dim*2
        self.time_proj_dim4 = nn.Linear(dim, dim*4) # for dim*4

    def forward(self, x_t, t, text_emb, secret_emb):
        # 1. 准备 Condition
        t_emb = self.time_mlp(t.view(-1, 1)) # [B, dim]
        txt_emb = self.text_proj(text_emb.mean(dim=1)) # [B, dim]
        
        # 融合 Time 和 Text 作为全局 Condition
        global_cond = t_emb + txt_emb # [B, dim]

        # 预计算不同尺度的 Condition
        cond_dim = global_cond
        cond_dim2 = self.time_proj_dim2(global_cond)
        cond_dim4 = self.time_proj_dim4(global_cond)

        # 2. U-Net Forward
        x1 = self.inc(x_t)
        # [关键] 注入时间到 Block
        x1 = self.enc_block1(x1, time_emb=cond_dim) 
        
        x2 = self.down1(x1)
        x2 = self.enc_block2(x2, time_emb=cond_dim2)
        
        x3 = self.down2(x2)
        h_struc = x3 # Bottleneck input

        # 3. Secret Injection
        s_feat = F.interpolate(secret_emb, size=h_struc.shape[2:])
        s_feat = self.secret_proj(s_feat)
        h_tex = self.modulator(h_struc, s_feat)

        # Bottleneck Block
        x = self.bot(h_tex, time_emb=cond_dim4)

        # Decoder
        x = self.up1_conv(x)
        x = x + x2 # Skip connection
        x = self.up1_block(x, time_emb=cond_dim2)

        x = self.up2_conv(x)
        x = x + x1 # Skip connection
        x = self.up2_block(x, time_emb=cond_dim)

        v_pred = self.outc(x)

        return v_pred, h_struc, h_tex

# DecoderHead 不需要改，因为它做的是分类/检索，不需要时间 t
class MambaDecoderHead(nn.Module):
    def __init__(self, in_channels=3, dim=128,
                 num_quantizers=4, embed_dim=256):
        super().__init__()
        self.num_q = num_quantizers
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, dim, 4, 2, 1), 
            nn.ReLU(),
            VisionMambaBlock(dim), # Decoder 这里也可以选择不加 time_emb
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.ReLU(),
            VisionMambaBlock(dim * 2),
        )
        self.head = nn.Conv2d(dim * 2, num_quantizers * embed_dim, 1)

    def forward(self, x, target_shape=None):
        feat = self.net(x) 
        if target_shape is not None:
            if feat.shape[-2:] != target_shape:
                feat = F.adaptive_avg_pool2d(feat, target_shape)

        out = self.head(feat) 
        B, _, H, W = out.shape
        out = out.view(B, self.num_q, self.embed_dim, H, W)
        return out