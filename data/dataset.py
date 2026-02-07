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
        Args:
            root_dir (str): 数据集根目录 (例如 './data')
            split (str): 'train' 或 'test'
            image_size (int): 图像统一调整的大小
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.image_paths = []

        # 定义训练和测试的文件夹路径
        if split == 'train':
            sub_dirs = [
                self.root_dir / 'train' / 'train_DIV2K',
                self.root_dir / 'train' / 'valid_DIV2K'
            ]
        elif split == 'test':
            sub_dirs = [
                self.root_dir / 'test' / 'coco',
                self.root_dir / 'test' / 'imagenet'
            ]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

        # 遍历目录收集所有 jpg 和 png 图片
        for sub_dir in sub_dirs:
            if not sub_dir.exists():
                print(f"Warning: Directory {sub_dir} does not exist. Skipping.")
                continue
            
            # 递归搜索 (rglob)
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
            for ext in extensions:
                self.image_paths.extend(list(sub_dir.rglob(ext)))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {sub_dirs}. Check your directory structure.")

        print(f"[{split.upper()}] Loaded {len(self.image_paths)} images from {sub_dirs}")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化到 [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. 加载目标图像 (Target Image - 用于生成器学习的真实分布)
        target_path = self.image_paths[idx]
        try:
            target_img = Image.open(target_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {target_path}: {e}. Skipping.")
            return self.__getitem__(random.randint(0, len(self) - 1)) # 递归重试

        # 2. 加载秘密图像 (Secret Image)
        # 在训练中，我们随机选取另一张图片作为秘密图像
        secret_idx = random.randint(0, len(self) - 1)
        secret_path = self.image_paths[secret_idx]
        try:
            secret_img = Image.open(secret_path).convert('RGB')
        except:
            secret_img = target_img # Fallback

        # 应用变换
        target_tensor = self.transform(target_img)
        secret_tensor = self.transform(secret_img)

        # 3. 文本提示 (Prompt)
        # 理想情况下，如果有对应的 caption 文件，应该读取。
        # 这里为了通用性，使用一个通用的 Prompt，或者你可以根据文件夹名生成 Prompt
        # 例如: "A high quality photo"
        prompt = "A high quality photo" 

        return target_tensor, secret_tensor, prompt

def get_dataloader(config, split='train'):
    """
    根据配置文件获取 DataLoader
    """
    dataset = StegoDataset(
        root_dir=config['data'].get('root_dir', './data'), # 默认为 ./data
        split=split,
        image_size=config['data']['image_size']
    )
    
    shuffle = True if split == 'train' else False
    
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=shuffle,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )