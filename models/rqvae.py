# models/rqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ

# --- 1. 基础组件 (对标 VQ-GAN 的高性能模块) ---

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # num_groups=32 是 VQ-GAN 的标准设置
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    def forward(self, x):
        return self.gn(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        
        self.norm1 = GroupNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = GroupNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = Swish()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.shortcut(x)

class AttnBlock(nn.Module):
    """
    自注意力模块，用于捕获全局依赖关系。
    这对于高质量图像重建至关重要。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.norm = GroupNorm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw
        h_ = torch.bmm(v, w_)      # b, c,hw
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 使用卷积下采样，不使用 MaxPool，保留更多信息
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

    def forward(self, x):
        return self.conv(self.pad(x))

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        # 最近邻插值 + 卷积，避免棋盘效应 (Checkerboard Artifacts)
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)

# --- 2. 主模型 (High-Complexity RQ-VAE) ---

class SecretRQVAE(nn.Module):
    """
    Enhanced RQ-VAE: 
    工业级结构，包含 ResBlock 和 Attention，但保留了 num_quantizers=4 和 embed_dim=256 的接口。
    下采样率 f=4 (256 -> 64)。
    """
    def __init__(self, in_channels=3, embed_dim=256, codebook_size=1024, num_quantizers=4):
        super().__init__()
        
        # ---------------- ENCODER ----------------
        # 逐层加深: 128 -> 256 -> 512
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1),
            
            # Down 1 (256 -> 128)
            ResBlock(128, 256),
            ResBlock(256, 256),
            Downsample(256),
            
            # Down 2 (128 -> 64)
            ResBlock(256, 512),
            ResBlock(512, 512),
            Downsample(512),
            
            # Bottleneck (64x64) - 最关键的部分，提取深层语义
            ResBlock(512, 512),
            AttnBlock(512), # 加入全局注意力，大幅提升重建能力
            ResBlock(512, 512),
            
            # Mapping to embed_dim
            GroupNorm(512),
            Swish(),
            nn.Conv2d(512, embed_dim, 3, 1, 1)
        )
        
        # ---------------- QUANTIZER ----------------
        # 使用您指定的配置
        self.rq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,          
            decay=0.8,                 
            commitment_weight=0.25,
            accept_image_fmap=False,   # 保持手动 reshape，防止维度错误
            shared_codebook=True       
        )
        
        # ---------------- DECODER ----------------
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 3, 1, 1),
            
            # Bottleneck
            ResBlock(512, 512),
            AttnBlock(512),
            ResBlock(512, 512),
            
            # Up 1 (64 -> 128)
            ResBlock(512, 512),
            ResBlock(512, 256),
            Upsample(256),
            
            # Up 2 (128 -> 256)
            ResBlock(256, 256),
            ResBlock(256, 128),
            Upsample(128),
            
            # Output
            ResBlock(128, 128),
            GroupNorm(128),
            Swish(),
            nn.Conv2d(128, in_channels, 3, 1, 1),
            nn.Tanh() # 归一化到 [-1, 1]
        )
        
        self.embed_dim = embed_dim
        self.num_quantizers = num_quantizers

    def forward(self, x):
        # 1. Encode
        z = self.encoder(x) # [B, embed_dim, H/4, W/4]
        B, C, H, W = z.shape

        # 2. Reshape for Quantizer: [B, N, C]
        # 这是为了适应 ResidualVQ 最稳定的输入格式，避免 broadcasting 错误
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        # 3. Quantize
        quantized_flat, indices, commit_loss = self.rq(z_flat)

        # 4. Restore feature map
        # [B, N, C] -> [B, C, H, W]
        quantized = quantized_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 5. Decode
        recon = self.decoder(quantized)

        # 6. Reshape indices: [B, N, Q] -> [B, Q, H, W]
        if indices.ndim == 3:
            indices = indices.permute(0, 2, 1).contiguous()
            indices = indices.view(B, -1, H, W)

        return recon, indices, commit_loss, quantized