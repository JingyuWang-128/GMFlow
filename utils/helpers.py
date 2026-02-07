# utils/helpers.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import torchvision

def set_seed(seed=42):
    """固定随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def denormalize(tensor):
    """将 [-1, 1] 的 Tensor 反归一化到 [0, 1] 用于保存"""
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def compute_psnr(img1, img2):
    """
    计算两个图像 Tensor 之间的 PSNR
    img1, img2: [B, C, H, W] in range [-1, 1] or [0, 1]
    """
    # 确保在 [0, 1] 范围内计算
    if img1.min() < 0:
        img1 = denormalize(img1)
    if img2.min() < 0:
        img2 = denormalize(img2)
        
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.view(mse.shape[0], -1).mean(dim=1)
    psnr = 10. * torch.log10(1. / mse)
    return psnr.mean().item()

def compute_ssim(img1, img2):
    """
    计算 SSIM。这里使用 Kornia 库，因为它高效且可微。
    如果没有安装 kornia，则需要手动实现或使用 torchmetrics。
    """
    try:
        import kornia.metrics as metrics
        # Kornia expects images in [0, 1]
        if img1.min() < 0: img1 = denormalize(img1)
        if img2.min() < 0: img2 = denormalize(img2)
        
        # window_size=11 is standard for SSIM
        ssim_val = metrics.ssim(img1, img2, window_size=11, reduction='mean')
        return ssim_val.item()
    except ImportError:
        print("Warning: Kornia not found, skipping SSIM calculation.")
        return 0.0

def save_image_grid(images, path, nrow=4):
    """保存图像网格"""
    # images: List of tensors or a single tensor [B, C, H, W]
    if isinstance(images, list):
        images = torch.stack(images, dim=0)
    
    # 反归一化
    if images.min() < 0:
        images = denormalize(images)
        
    torchvision.utils.save_image(images, path, nrow=nrow)

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count