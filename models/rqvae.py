# models/rqvae.py
import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

class SecretRQVAE(nn.Module):
    """
    RQ-VAE: 残差量化变分自编码器。
    用于将秘密图像离散化为多层 Token。
    """
    def __init__(self, in_channels=3, embed_dim=256, codebook_size=1024, num_quantizers=4):
        super().__init__()
        
        # Encoder: Downsample 4x (256 -> 128 -> 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )
        
        # Residual Vector Quantizer
        # 修复逻辑：启用 decay 和 kmeans_init 以确保码本能正确处理 batch 广播
        self.rq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,          # 建议开启，有利于码本初始化
            decay=0.8,                 # 必须 > 0 以启用 EMA，解决维度不匹配报错
            commitment_weight=0.25,
            accept_image_fmap=False,   # 我们在 forward 中手动重塑
            shared_codebook=True       # 强制共享码本，防止 batch size 变化导致的错误
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 1. Encode
        z = self.encoder(x)  # [B, 256, 64, 64]
        B, C, H, W = z.shape

        # 2. 重塑为 [B, N, C] 格式，这是库处理最稳定的格式
        z_flat = z.reshape(B, H * W, C).contiguous()

        # 3. Quantize
        quantized_flat, indices, commit_loss = self.rq(z_flat)

        # 4. Restore feature map
        # [B, N, C] -> [B, C, H, W]
        quantized = quantized_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 5. Decode
        recon = self.decoder(quantized)

        # 6. Reshape indices: [B, N, Q] -> [B, Q, H, W]
        if indices.ndim == 3:
            indices = indices.permute(0, 2, 1).contiguous()
            indices = indices.view(B, -1, H, W)

        return recon, indices, commit_loss, quantized