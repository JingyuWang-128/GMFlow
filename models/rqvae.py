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
            kmeans_init=False,    
            kmeans_iters=10,
            decay=0.0             # 必须为 0.0 以修复 RuntimeError broadcast shape
        )
        
        # Decoder: Upsample 4x
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1),
            nn.Tanh() # [-1, 1]
        )

    def forward(self, x):
        # 1. 编码
        z = self.encoder(x) # Shape: [B, C=256, H=64, W=64]
        
        # 2. 维度置换 (Channel Last)
        # ResidualVQ 需要输入的最后一个维度是 embed_dim (256)
        # 所以我们将 [B, C, H, W] -> [B, H, W, C]
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        
        # 3. 残差量化
        # quantized: [B, H, W, C]
        # indices: [B, H*W, Num_Q] (库的输出形状可能略有不同，下面会统一处理)
        quantized, indices, commit_loss = self.rq(z_permuted)
        
        # 4. 维度还原 (Channel First)
        # [B, H, W, C] -> [B, C, H, W] 以供 Decoder 使用
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # 5. 解码
        recon = self.decoder(quantized)
        
        # 6. 处理 Indices 形状
        # 我们希望输出统一为 [B, Num_Q, H, W]
        B, H, W, C = z_permuted.shape
        
        # 如果 indices 是 3D 张量，通常是 [Batch, Sequence_Length, Num_Quantizers]
        if indices.ndim == 3:
            # 检查最后一个维度是不是 Num_Quantizers
            if indices.shape[-1] == self.rq.num_quantizers:
                # [B, Seq, Num_Q] -> [B, Num_Q, Seq] -> [B, Num_Q, H, W]
                indices = indices.permute(0, 2, 1).view(B, -1, H, W)
            else:
                # 某些版本可能是 [B, Num_Q, Seq]
                indices = indices.view(B, -1, H, W)
                
        return recon, indices, commit_loss, quantized