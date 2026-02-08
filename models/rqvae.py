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
        self.rq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            kmeans_init=False,    # 关闭 Kmeans
            kmeans_iters=10,
            decay=0.0,            # 关闭 EMA
            accept_image_fmap=True # 库会自动处理维度，不需要我们在外面 permute
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
        
        # [修复] 
        # 直接传入 [B, C, H, W]，不要使用 permute！
        # 库内部会自动识别这是图像，并将其转为 [B, H, W, C] 进行处理，最后再转回来。
        
        # quantized: [B, C, H, W] (自动转回)
        # indices: [B, H*W, Num_Q] (通常库输出 flatten 后的 indices)
        quantized, indices, commit_loss = self.rq(z)
        
        # 2. 解码
        recon = self.decoder(quantized)
        
        # 3. 处理 Indices 形状
        # 我们希望输出统一为 [B, Num_Q, H, W] 方便后续计算 Loss
        B, C, H, W = z.shape # 注意这里的 H, W 是 Feature map 的大小 (64)
        
        # vector_quantize_pytorch 输出的 indices 形状可能是 [B, H*W, Num_Q]
        if indices.ndim == 3:
            # Case: [B, Seq_Len, Num_Q] -> [B, Num_Q, Seq_Len]
            indices = indices.permute(0, 2, 1)
            # -> [B, Num_Q, H, W]
            indices = indices.view(B, -1, H, W)
                
        return recon, indices, commit_loss, quantized