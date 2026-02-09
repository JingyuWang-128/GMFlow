# utils/helpers.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
import torchvision
from datetime import datetime

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_dir, name="experiment"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def denormalize(tensor):
    """[-1, 1] -> [0, 1]"""
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def compute_psnr(img1, img2):
    """img1, img2: [B, C, H, W] in [-1, 1]"""
    img1 = denormalize(img1)
    img2 = denormalize(img2)
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1,2,3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()

def compute_ssim(img1, img2):
    try:
        import kornia.metrics as metrics
        img1 = denormalize(img1)
        img2 = denormalize(img2)
        
        return metrics.ssim(img1, img2, window_size=11).mean().item()
        
    except ImportError:
        return 0.0

def compute_bit_accuracy(pred_indices, target_indices):
    """计算离散 Token 的恢复准确率"""
    # pred_indices: [B, Q, H, W]
    # target_indices: [B, Q, H, W]
    correct = (pred_indices == target_indices).float()
    return correct.mean().item()

def save_image_grid(images, path, nrow=4, normalize=True):
    if normalize:
        images = denormalize(images)
    torchvision.utils.save_image(images, path, nrow=nrow)