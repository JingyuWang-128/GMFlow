# train_stage1_rqvae.py
import torch
import yaml
from torch.optim import Adam
from models.rqvae import SecretRQVAE
from data.dataset import get_dataloader
from utils.helpers import set_seed, save_image_grid, compute_psnr, compute_ssim, AverageMeter

def train():
    set_seed(42)
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['experiment']['device']
    model = SecretRQVAE(
        embed_dim=config['rqvae']['embed_dim'],
        codebook_size=config['rqvae']['codebook_size']
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=float(config['rqvae']['learning_rate']))
    loader = get_dataloader(config)
    
    print("Stage 1: Training RQ-VAE...")
    for epoch in range(5): # Demo epoch
        for _, secret, _ in loader:
            secret = secret.to(device)
            
            recon, _, commit_loss, _ = model(secret)
            
            recon_loss = torch.nn.functional.mse_loss(recon, secret)
            loss = recon_loss + commit_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")
        
    torch.save(model.state_dict(), "checkpoints/rqvae.pth")
    print("RQ-VAE Saved.")

if __name__ == "__main__":
    train()