# train_stage1_rqvae.py
import torch
import yaml
import os
from torch.optim import Adam
from models.rqvae import SecretRQVAE
from data.dataset import get_dataloader
from utils.helpers import set_seed, get_logger, save_image_grid, compute_psnr, compute_ssim
from tqdm import tqdm

def train():
    set_seed(42)
    # Load Config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['experiment']['device']
    log_dir = os.path.join(config['experiment']['log_dir'], "stage1_rqvae")
    ckpt_dir = os.path.join(config['experiment']['checkpoint_dir'], "stage1_rqvae")
    res_dir = os.path.join(config['experiment']['result_dir'], "stage1_rqvae")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    logger = get_logger(log_dir, "RQVAE_Train")
    logger.info("Configuration Loaded.")
    
    model = SecretRQVAE(
        embed_dim=config['rqvae']['embed_dim'],
        codebook_size=config['rqvae']['codebook_size'],
        num_quantizers=config['rqvae']['num_quantizers']
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=float(config['rqvae']['learning_rate']))
    
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='test')
    
    epochs = config['rqvae']['epochs']
    best_psnr = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss_avg = 0
        
        # 优化点：使用 tqdm 动态显示，移除 leave=False 以外的干扰
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (_, secret, _) in enumerate(tqdm_bar):
            secret = secret.to(device)
            recon, _, commit_loss, _ = model(secret)
            recon_loss = torch.nn.functional.mse_loss(recon, secret)
            loss = recon_loss + commit_loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_avg += loss.item()
            
            # 优化点：通过 set_postfix 动态更新状态，不换行
            tqdm_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{train_loss_avg/(i+1):.4f}")
            
            # 优化点：如果需要记录日志，使用 tqdm_bar.write 避免破坏进度条
            if i % config['rqvae']['val_interval'] == 0:
                tqdm_bar.write(f"[*] Step {i}: Loss {loss.item():.4f}")
        
        train_loss_avg /= len(train_loader)
        
        # Validation
        model.eval()
        val_psnr_avg = 0
        val_ssim_avg = 0
        with torch.no_grad():
            for i, (_, secret, _) in enumerate(tqdm(val_loader, desc="[Val]", leave=False)):
                secret = secret.to(device)
                recon, _, _, _ = model(secret)
                
                val_psnr_avg += compute_psnr(secret, recon)
                val_ssim_avg += compute_ssim(secret, recon)
                
                if i == 0:
                    vis = torch.cat([secret[:4], recon[:4]], dim=0) # 仅取前4张对比
                    save_image_grid(vis, os.path.join(res_dir, f"val_epoch_{epoch}.png"), nrow=4)
        
        val_psnr_avg /= len(val_loader)
        val_ssim_avg /= len(val_loader)
        
        # Epoch 结束时记录一次完整的日志
        logger.info(f"Epoch {epoch} Result: Loss {train_loss_avg:.4f}, PSNR {val_psnr_avg:.2f}")
        
        if val_psnr_avg > best_psnr:
            best_psnr = val_psnr_avg
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_rqvae.pth"))
            
        if epoch % config['rqvae']['save_interval'] == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"rqvae_epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_rqvae.pth"))
    logger.info("Training Completed.")

if __name__ == "__main__":
    train()