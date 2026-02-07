# modules/interference.py
import torch
import torch.nn as nn
import kornia.augmentation as K

class InterferenceOperatorSet(nn.Module):
    """
    可微干扰算子集，用于训练中的鲁棒性引导。
    包含：噪声、模糊、模拟JPEG、遮挡
    """
    def __init__(self):
        super().__init__()
        self.aug_strong = nn.Sequential(
            K.RandomGaussianNoise(mean=0., std=0.1, p=0.8),
            K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.5),
            K.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), p=0.5),
            # 模拟 JPEG: 下采样再上采样模拟高频丢失
            K.Resize((128, 128)),
            K.Resize((256, 256))
        )
        
        self.aug_weak = nn.Sequential(
            K.RandomGaussianNoise(mean=0., std=0.02, p=0.3),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.2)
        )

    def forward(self, x, severity='strong'):
        if severity == 'strong':
            return self.aug_strong(x)
        else:
            return self.aug_weak(x)