# data/dataset.py
import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class StegoDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=256):
        """
        root_dir: 数据集根目录，例如 /home/wangjingyu/GMFlow/data
        split: 'train' | 'val' | 'test'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.image_paths = []

        # 根据你的新需求修改目录划分
        if split == 'train':
            # 训练集：只包含 train_DIV2K
            sub_dirs = [
                self.root_dir / 'train' / 'train_DIV2K'
            ]
        elif split == 'val':
            # 验证集：只包含 valid_DIV2K (新增)
            sub_dirs = [
                self.root_dir / 'train' / 'valid_DIV2K'
            ]
        elif split == 'test':
            # 测试集：包含 coco 和 imagenet
            sub_dirs = [
                self.root_dir / 'test' / 'coco',
                self.root_dir / 'test' / 'imagenet'
            ]
        else:
            raise ValueError(f"Invalid split: {split}")

        # 递归搜索图片
        for sub_dir in sub_dirs:
            if not sub_dir.exists():
                print(f"Warning: Directory {sub_dir} does not exist. Skipping.")
                continue
            
            # 支持常见图片格式
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
            for ext in extensions:
                self.image_paths.extend(list(sub_dir.rglob(ext)))

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {sub_dirs}. Using dummy data for debugging.")
            # 仅用于调试，防止报错
            self.image_paths = ["dummy.jpg"] * 100

        print(f"[{split.upper()}] Loaded {len(self.image_paths)} images from {sub_dirs}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. 目标图像
        target_path = self.image_paths[idx]
        if str(target_path) == "dummy.jpg":
            return self._get_dummy_item()

        try:
            target_img = Image.open(target_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {target_path}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

        # 2. 秘密图像 (随机选取)
        secret_idx = random.randint(0, len(self) - 1)
        try:
            secret_path = self.image_paths[secret_idx]
            if str(secret_path) == "dummy.jpg":
                secret_img = target_img
            else:
                secret_img = Image.open(secret_path).convert('RGB')
        except:
            secret_img = target_img

        target_tensor = self.transform(target_img)
        secret_tensor = self.transform(secret_img)

        # 3. Prompt (暂用固定，可扩展为读取 caption 文件)
        prompt = "A high quality photo"

        return target_tensor, secret_tensor, prompt

    def _get_dummy_item(self):
        import torch
        return torch.randn(3, self.image_size, self.image_size), \
               torch.randn(3, self.image_size, self.image_size), \
               "dummy prompt"

def get_dataloader(config, split='train'):
    dataset = StegoDataset(
        root_dir=config['data']['root_dir'],
        split=split,
        image_size=config['data']['image_size']
    )
    # 只有训练集需要 shuffle，验证和测试通常不需要
    shuffle = True if split == 'train' else False
    
    return DataLoader(
        dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=shuffle, 
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )