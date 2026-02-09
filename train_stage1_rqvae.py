# train_stage1_rqvae.py
import torch
import yaml
import os
from torch.optim import Adam
from models.rqvae import SecretRQVAE
from data.dataset import get_dataloader
from utils.helpers import set_seed, get_logger, save_image_grid, compute_psnr, compute_ssim
from tqdm import tqdm
from accelerate import Accelerator  # [新增] 引入 accelerate

def train():
    # [新增] 初始化 Accelerator
    # 它可以自动检测是单卡还是多卡，甚至自动处理混合精度（如 fp16）
    accelerator = Accelerator()
    
    # 设置随机种子 (Accelerate 会自动为不同进程设置不同的随机种子以保证数据不同)
    set_seed(42 + accelerator.process_index)

    # Load Config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # [修改] 不再从 config 读取 device，而是使用 accelerator 分配的 device
    device = accelerator.device 
    
    # [修改] 只有主进程负责创建目录，防止多进程冲突
    log_dir = os.path.join(config['experiment']['log_dir'], "stage1_rqvae")
    ckpt_dir = os.path.join(config['experiment']['checkpoint_dir'], "stage1_rqvae")
    res_dir = os.path.join(config['experiment']['result_dir'], "stage1_rqvae")
    
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        logger = get_logger(log_dir, "RQVAE_Train")
        logger.info("Configuration Loaded and Accelerator Initialized.")
    else:
        logger = None # 其他进程不需要 logger

    # Model Init
    model = SecretRQVAE(
        embed_dim=config['rqvae']['embed_dim'],
        codebook_size=config['rqvae']['codebook_size'],
        num_quantizers=config['rqvae']['num_quantizers']
    ) # [修改] 去掉 .to(device)，prepare 会自动处理
    
    optimizer = Adam(model.parameters(), lr=float(config['rqvae']['learning_rate']))
    
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # [新增] 核心步骤：Prepare
    # accelerate 会自动包装 model 为 DDP，并把数据加载器切分给不同的 GPU
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    epochs = config['rqvae']['epochs']
    best_psnr = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss_avg = 0
        
        # [修改] 只有主进程显示进度条，disable=not accelerator.is_main_process
        tqdm_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs} [Train]", 
            disable=not accelerator.is_main_process
        )
        
        for i, (_, secret, _) in enumerate(tqdm_bar):
            # secret = secret.to(device) # [修改] prepare 后的 loader 会自动把数据移到正确的 gpu，这行可以注释掉，或者保留也没事
            
            recon, _, commit_loss, _ = model(secret)
            recon_loss = torch.nn.functional.mse_loss(recon, secret)
            loss = recon_loss + commit_loss.mean()
            
            optimizer.zero_grad()
            
            # [修改] 使用 accelerator 反向传播
            accelerator.backward(loss)
            
            optimizer.step()
            
            train_loss_avg += loss.item()
            
            if accelerator.is_main_process:
                tqdm_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{train_loss_avg/(i+1):.4f}")
                if i % config['rqvae']['val_interval'] == 0:
                    tqdm_bar.write(f"[*] Step {i}: Loss {loss.item():.4f}")
        
        # 等待所有进程跑完这个 epoch
        accelerator.wait_for_everyone()
        
        train_loss_avg /= len(train_loader)
        
        # Validation
        model.eval()
        val_psnr_avg = 0
        val_ssim_avg = 0
        
        # 验证集也需要进度条控制
        val_bar = tqdm(
            val_loader, 
            desc="[Val]", 
            leave=False, 
            disable=not accelerator.is_main_process
        )
        
        with torch.no_grad():
            for i, (_, secret, _) in enumerate(val_bar):
                # secret = secret.to(device) # 自动处理
                recon, _, _, _ = model(secret)
                
                # 计算指标 (这里简单计算，严谨的做法是用 accelerator.gather 把所有卡的结果收集起来再算)
                val_psnr_avg += compute_psnr(secret, recon)
                val_ssim_avg += compute_ssim(secret, recon)
                
                # [修改] 只在主进程保存图片
                if i == 0 and accelerator.is_main_process:
                    vis = torch.cat([secret[:4], recon[:4]], dim=0)
                    save_image_grid(vis, os.path.join(res_dir, f"val_epoch_{epoch}.png"), nrow=4)
        
        val_psnr_avg /= len(val_loader)
        val_ssim_avg /= len(val_loader)
        
        # [修改] 只有主进程负责打印日志和保存模型
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Result: Loss {train_loss_avg:.4f}, PSNR {val_psnr_avg:.2f}")
            
            # 获取原始模型 (解包 DDP)
            unwrapped_model = accelerator.unwrap_model(model)
            
            if val_psnr_avg > best_psnr:
                best_psnr = val_psnr_avg
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "best_rqvae.pth"))
                
            if epoch % config['rqvae']['save_interval'] == 0:
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, f"rqvae_epoch_{epoch}.pth"))
            
            # 保存 latest
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "last_rqvae.pth"))
    
    if accelerator.is_main_process:
        logger.info("Training Completed.")

if __name__ == "__main__":
    train()