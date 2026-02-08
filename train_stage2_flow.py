# train_stage2_flow.py
import torch
import yaml
import os
from torch.optim import AdamW
from modules.text_encoder import TextEncoderWrapper
from models.rqvae import SecretRQVAE
from models.gen_flow import TriStreamMambaUNet, MambaDecoderHead
from modules.losses import GenMambaLosses
from modules.interference import InterferenceOperatorSet
from data.dataset import get_dataloader
from utils.helpers import set_seed, get_logger, save_image_grid, compute_psnr, compute_bit_accuracy

def train_flow():
    set_seed(42)
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = config['experiment']['device']
    log_dir = os.path.join(config['experiment']['log_dir'], "stage2_flow")
    ckpt_dir = os.path.join(config['experiment']['checkpoint_dir'], "stage2_flow")
    res_dir = os.path.join(config['experiment']['result_dir'], "stage2_flow")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    logger = get_logger(log_dir, "GenFlow_Train")
    
    # 1. Models Initialization
    # RQ-VAE (Frozen)
    rqvae = SecretRQVAE(embed_dim=config['rqvae']['embed_dim']).to(device)
    rqvae_path = os.path.join(config['experiment']['checkpoint_dir'], "stage1_rqvae", "best_rqvae.pth")
    if os.path.exists(rqvae_path):
        rqvae.load_state_dict(torch.load(rqvae_path))
        logger.info(f"Loaded RQ-VAE from {rqvae_path}")
    else:
        logger.warning("RQ-VAE checkpoint not found! Using random initialization (Expect poor performance).")
    rqvae.eval()
    
    # Generator & Decoder
    gen_model = TriStreamMambaUNet(secret_dim=config['rqvae']['embed_dim']).to(device)
    decoder = MambaDecoderHead(num_quantizers=config['rqvae']['num_quantizers']).to(device)
    
    # Helpers
    text_enc = TextEncoderWrapper().to(device)
    interfere = InterferenceOperatorSet().to(device)
    
    # Optimizers
    opt_gen = AdamW(gen_model.parameters(), lr=float(config['gen_flow']['learning_rate']))
    opt_dec = AdamW(decoder.parameters(), lr=float(config['gen_flow']['learning_rate']))
    
    # Dataloaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='test')
    
    epochs = config['gen_flow']['epochs']
    
    # Loss Weights
    lambda_flow = config['gen_flow']['lambda_flow']
    lambda_rsmi = config['gen_flow']['lambda_rsmi']
    lambda_robust = config['gen_flow']['lambda_robust']

    for epoch in range(epochs):
        gen_model.train()
        decoder.train()
        
        total_loss_meter = 0
        robust_loss_meter = 0
        
        for i, (target_img, secret_img, prompts) in enumerate(train_loader):
            target_img = target_img.to(device)
            secret_img = secret_img.to(device)
            
            # 1. Prepare Inputs
            with torch.no_grad():
                txt_emb = text_enc(prompts, device)
                _, s_indices, _, s_feat = rqvae(secret_img)
            
            # 2. Flow Matching Training
            B = target_img.shape[0]
            t = torch.rand(B).to(device)
            x_0 = torch.randn_like(target_img).to(device) # Noise
            t_view = t.view(B, 1, 1, 1)
            x_t = t_view * target_img + (1 - t_view) * x_0 # Interpolation
            
            # 3. Forward
            v_pred, h_struc, h_tex = gen_model(x_t, t, txt_emb, s_feat)
            
            # 4. Losses
            loss_flow = GenMambaLosses.flow_matching_loss(v_pred, x_0, target_img)
            loss_rsmi = GenMambaLosses.rSMI_loss(h_struc, h_tex)
            
            # 5. Robustness Guidance (Train Decoder on attacked estimates)
            x_0_pred = x_t - (1 - t_view) * v_pred # One-step estimate
            x_attacked = interfere(x_0_pred, severity='strong') # Simulate Attack
            pred_logits = decoder(x_attacked)
            
            s_indices_flat = s_indices.view(B, config['rqvae']['num_quantizers'], -1)
            loss_robust = GenMambaLosses.robust_decode_loss(pred_logits, s_indices_flat)
            
            total_loss = lambda_flow * loss_flow + lambda_rsmi * loss_rsmi + lambda_robust * loss_robust
            
            opt_gen.zero_grad()
            opt_dec.zero_grad()
            total_loss.backward()
            opt_gen.step()
            opt_dec.step()
            
            total_loss_meter += total_loss.item()
            robust_loss_meter += loss_robust.item()
            
            if i % 50 == 0:
                logger.info(f"Epoch [{epoch}][{i}] Total: {total_loss.item():.4f} Robust: {loss_robust.item():.4f}")
        
        # Validation Loop
        if epoch % 5 == 0:
            gen_model.eval()
            decoder.eval()
            val_stego_psnr = 0
            val_bit_acc = 0
            
            with torch.no_grad():
                # Take one batch from validation
                for val_target, val_secret, val_prompts in val_loader:
                    val_target = val_target.to(device)
                    val_secret = val_secret.to(device)
                    
                    # Generate Stego (Euler 20 steps for speed in val)
                    txt_emb = text_enc(val_prompts, device)
                    _, s_indices, _, s_feat = rqvae(val_secret)
                    
                    x_gen = torch.randn_like(val_target).to(device)
                    dt = 1.0 / 20
                    for k in range(20):
                        t_val = torch.tensor([k/20.0]).to(device)
                        v_val, _, _ = gen_model(x_gen, t_val, txt_emb, s_feat)
                        x_gen = x_gen + v_val * dt
                    
                    # Metrics
                    val_stego_psnr += compute_psnr(val_target, x_gen)
                    
                    # Attack & Decode
                    x_val_attacked = interfere(x_gen, severity='weak')
                    logits = decoder(x_val_attacked)
                    pred_idx = logits.argmax(dim=2) # [B, Q, L]
                    
                    s_indices_flat = s_indices.view(s_indices.shape[0], config['rqvae']['num_quantizers'], -1)
                    val_bit_acc += compute_bit_accuracy(pred_idx, s_indices_flat)
                    
                    # Visualization (First batch only)
                    if epoch % 10 == 0:
                        # Reconstruct secret
                        H_lat = val_secret.shape[2] // 4
                        pred_map = pred_idx.view(val_target.shape[0], 4, H_lat, H_lat).permute(0, 3, 1, 2)
                        
                        # Fix for vector-quantize: need to use indices to look up codebook manually or use internal function
                        # Here assuming rqvae has a helper or we use raw indices passing if supported
                        # 简易实现：使用 RQVAE 内部的 decode
                        recon_secret = rqvae.rq.get_output_from_indices(pred_map.permute(0, 2, 3, 1))
                        recon_secret = recon_secret.permute(0, 3, 1, 2) # [B, C, H, W]
                        recon_secret = rqvae.decoder(recon_secret)
                        
                        # Save: Target | Stego | Secret | Recovered
                        vis_row = torch.cat([val_target, x_gen, val_secret, recon_secret], dim=0)
                        save_image_grid(vis_row, os.path.join(res_dir, f"val_step_{epoch}.png"), nrow=val_target.shape[0])
                    
                    break # Only validate one batch to save time
            
            logger.info(f"VAL Epoch {epoch}: Stego PSNR: {val_stego_psnr:.2f} | Bit Acc: {val_bit_acc:.4f}")
            
            torch.save(gen_model.state_dict(), os.path.join(ckpt_dir, f"gen_epoch_{epoch}.pth"))
            torch.save(decoder.state_dict(), os.path.join(ckpt_dir, f"dec_epoch_{epoch}.pth"))

    logger.info("Training Finished.")

if __name__ == "__main__":
    train_flow()