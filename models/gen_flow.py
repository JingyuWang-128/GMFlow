# models/gen_flow.py
import torch
import torch.nn as nn
from .mamba_block import VisionMambaBlock

class TriStreamMambaUNet(nn.Module):
    """
    GenMamba-Flow 核心生成器。
    实现三流解耦：Semantic(Frozen), Structure, Texture。
    """
    def __init__(self, in_channels=3, dim=128, secret_dim=256):
        super().__init__()
        
        # 1. 语义流注入 (简单 MLP 处理 Time + Text)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.text_proj = nn.Linear(768, dim) # CLIP dim
        
        # 2. 秘密信息投影 (Texture 流注入)
        self.secret_proj = nn.Linear(secret_dim, dim)

        # U-Net Encoder
        self.inc = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.down1 = nn.Sequential(VisionMambaBlock(dim), nn.Conv2d(dim, dim*2, 4, 2, 1))
        self.down2 = nn.Sequential(VisionMambaBlock(dim*2), nn.Conv2d(dim*2, dim*4, 4, 2, 1))
        
        # Bottleneck (Tri-Stream Injection Happens Here)
        self.bot = VisionMambaBlock(dim*4)
        
        # U-Net Decoder
        self.up1 = nn.Sequential(nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1), VisionMambaBlock(dim*2))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(dim*2, dim, 4, 2, 1), VisionMambaBlock(dim))
        
        self.outc = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x_t, t, text_emb, secret_emb):
        """
        x_t: Noisy Image [B, 3, H, W]
        t: Time [B]
        text_emb: [B, Seq, 768]
        secret_emb: [B, Secret_Dim, H', W'] (来自 RQVAE 的 quantized feature)
        """
        # Embeddings
        t_emb = self.time_mlp(t.view(-1, 1))
        # 简化 Text Pooling
        txt_emb = self.text_proj(text_emb.mean(dim=1)) 
        
        cond = (t_emb + txt_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Encoder
        x1 = self.inc(x_t) + cond
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # --- Tri-Stream Injection Point ---
        # x3 视为 Structure Stream
        h_struc = x3 
        
        # Texture Stream = Structure + Secret
        # 需将 secret_emb 调整到当前 feature map 大小
        s_feat = torch.nn.functional.interpolate(secret_emb, size=x3.shape[2:])
        s_feat = self.secret_proj(s_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 注入！
        h_tex = h_struc + s_feat
        
        # Bottleneck 处理 Texture 流
        x = self.bot(h_tex)
        
        # Decoder
        x = self.up1(x + x2) # Skip connection
        x = self.up2(x + x1)
        
        v_pred = self.outc(x)
        
        return v_pred, h_struc, h_tex

class MambaDecoderHead(nn.Module):
    """
    用于从隐写图像恢复 Secret Indices 的分类头。
    用于计算鲁棒性梯度。
    """
    def __init__(self, in_channels=3, dim=128, num_quantizers=4, codebook_size=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, dim, 4, 2, 1), nn.ReLU(), # 256->128
            VisionMambaBlock(dim),
            nn.Conv2d(dim, dim*2, 4, 2, 1), nn.ReLU(),       # 128->64
            VisionMambaBlock(dim*2),
            nn.Conv2d(dim*2, dim*4, 4, 2, 1), nn.ReLU()      # 64->32 (Match RQVAE latent size)
        )
        # 输出 Logits: [B, Num_Q * Codebook_Size, H, W]
        self.head = nn.Conv2d(dim*4, num_quantizers * codebook_size, 1)
        self.num_q = num_quantizers
        self.cb_size = codebook_size

    def forward(self, x):
        feat = self.net(x)
        logits = self.head(feat)
        # Reshape to [B, Num_Q, Codebook_Size, H, W] -> Flatten for loss
        B, _, H, W = logits.shape
        logits = logits.view(B, self.num_q, self.cb_size, H*W)
        return logits