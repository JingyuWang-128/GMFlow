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
        self.rq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10
        )
        
        # Decoder: Upsample 4x
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1),
            nn.Tanh() # [-1, 1]
        )

    def forward(self, x):
        z = self.encoder(x) # [B, C, H/4, W/4]
        z_permuted = z.permute(0, 2, 3, 1) # [B, H, W, C]
        
        # quantized: 重建用的特征
        # indices: [B, H*W, Num_Quantizers] (离散码)
        quantized, indices, commit_loss = self.rq(z_permuted)
        
        quantized = quantized.permute(0, 3, 1, 2)
        recon = self.decoder(quantized)
        
        # Reshape indices for convenience: [B, Num_Q, H, W]
        B, H, W, _ = z_permuted.shape
        indices = indices.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        return recon, indices, commit_loss, quantized