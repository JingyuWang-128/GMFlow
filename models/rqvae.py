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
        
        # Encoder: Downsample 4x (256 -> 64 -> 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )
        
        # Residual Vector Quantizer
        # [最终修复方案]
        # 1. kmeans_init=False: 禁用 Kmeans 初始化，防止 init_embed_ 广播错误
        # 2. decay=0.0: 禁用 EMA，防止 ema_inplace 广播错误
        # 3. accept_image_fmap=True: (默认值) 显式声明库会自动处理 [B, C, H, W] 输入
        # 1. 初始化时
        self.rq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=False,
            decay=0.0,
            ema_update=False,
            accept_image_fmap=False
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

        # 2. Flatten to token sequence for ResidualVQ
        # [B, C, H, W] -> [B, H, W, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous()

        # 3. Quantize (ONLY ONCE)
        quantized_flat, indices, commit_loss = self.rq(z_flat)

        # 4. Restore feature map
        # [B, H, W, C] -> [B, C, H, W]
        quantized = quantized_flat.permute(0, 3, 1, 2).contiguous()

        # 5. Decode
        recon = self.decoder(quantized)

        # 6. Reshape indices if needed
        # vector_quantize_pytorch 通常输出 [B, H*W, num_quantizers]
        if indices.ndim == 3:
            indices = indices.permute(0, 2, 1).contiguous()
            indices = indices.view(B, -1, H, W)

        return recon, indices, commit_loss, quantized