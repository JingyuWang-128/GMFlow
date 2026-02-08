# inference.py
import torch
import yaml
import os
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Metrics
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal.clip_score import CLIPScore

# Local Modules
from models.rqvae import SecretRQVAE
from models.gen_flow import TriStreamMambaUNet, MambaDecoderHead
from modules.text_encoder import TextEncoderWrapper
from modules.interference import InterferenceOperatorSet
from data.dataset import get_dataloader
from utils.helpers import set_seed, denormalize, compute_bit_accuracy

def evaluate_metrics(config, max_batches=None):
    set_seed(config['experiment']['seed'])
    device = config['experiment']['device']
    output_dir = os.path.join(config['experiment']['result_dir'], "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Evaluation on {device} ---")
    
    # 1. Load Models
    print("Loading models...")
    # RQ-VAE
    rqvae = SecretRQVAE(embed_dim=config['rqvae']['embed_dim']).to(device)
    rqvae_path = os.path.join(config['experiment']['checkpoint_dir'], "stage1_rqvae/best_rqvae.pth")
    if os.path.exists(rqvae_path):
        rqvae.load_state_dict(torch.load(rqvae_path))
    else:
        print("Warning: RQ-VAE checkpoint missing, metrics will be invalid.")
    rqvae.eval()

    # Generator
    gen_model = TriStreamMambaUNet(secret_dim=config['rqvae']['embed_dim']).to(device)
    gen_path = os.path.join(config['experiment']['checkpoint_dir'], "stage2_flow/gen_epoch_95.pth") # 需根据实际情况修改
    # 尝试加载最新的模型
    if not os.path.exists(gen_path):
         # 自动寻找最新的
         ckpt_dir = os.path.join(config['experiment']['checkpoint_dir'], "stage2_flow")
         ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('gen_epoch')]
         if ckpts:
             ckpts.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
             gen_path = os.path.join(ckpt_dir, ckpts[-1])
             print(f"Auto-selected checkpoint: {gen_path}")
    
    if os.path.exists(gen_path):
        gen_model.load_state_dict(torch.load(gen_path))
    else:
        print("Error: Generator checkpoint not found!")
        return
    gen_model.eval()

    # Decoder
    decoder = MambaDecoderHead(num_quantizers=config['rqvae']['num_quantizers']).to(device)
    dec_path = gen_path.replace("gen_epoch", "dec_epoch")
    if os.path.exists(dec_path):
        decoder.load_state_dict(torch.load(dec_path))
    decoder.eval()

    # Helpers
    text_enc = TextEncoderWrapper().to(device)
    interfere = InterferenceOperatorSet().to(device)

    # 2. Setup Metrics
    print("Initializing metrics...")
    # Fidelity Metrics
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device) # Perceptual distance
    
    # Secret Recovery Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # CLIP Score (Optional, requires CLIP weights)
    # clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # 3. Data Loader
    test_loader = get_dataloader(config, split='test')
    
    # 4. Evaluation Loop
    total_bit_acc = 0
    total_sec_psnr = 0
    total_sec_ssim = 0
    total_lpips = 0
    num_batches = 0
    
    print("Generating and evaluating...")
    with torch.no_grad():
        for i, (target_img, secret_img, prompts) in enumerate(tqdm(test_loader, desc="Evaluating", total=len(test_loader))):
            if max_batches and i >= max_batches:
                break
                
            target_img = target_img.to(device) # Real images (for FID)
            secret_img = secret_img.to(device)
            
            # --- Generate Stego ---
            txt_emb = text_enc(prompts, device)
            _, s_indices, _, s_feat = rqvae(secret_img)
            
            # Euler Solver (Generation)
            x_curr = torch.randn_like(target_img).to(device) # Noise x_0
            steps = 50
            dt = 1.0 / steps
            for k in tqdm(range(steps), desc="Euler Sampling", leave=False):
                t = torch.tensor([k / steps]).to(device)
                v_pred, _, _ = gen_model(x_curr, t, txt_emb, s_feat)
                x_curr = x_curr + v_pred * dt
            
            stego_img = x_curr
            
            # --- Attack & Decode ---
            # Paper setup usually tests with and without attack
            # Here we test with 'weak' attack (e.g. slight jpeg/noise)
            stego_attacked = interfere(stego_img, severity='weak')
            
            logits = decoder(stego_attacked)
            pred_indices = logits.argmax(dim=2) # [B, Q, L]
            
            # --- Reconstruct Secret ---
            H_lat = secret_img.shape[2] // 4
            pred_map = pred_indices.view(target_img.shape[0], 4, H_lat, H_lat).permute(0, 3, 1, 2)
            recon_secret = rqvae.rq.get_output_from_indices(pred_map.permute(0, 2, 3, 1))
            recon_secret = recon_secret.permute(0, 3, 1, 2)
            recon_secret = rqvae.decoder(recon_secret)

            # --- Normalize to [0, 1] for metrics ---
            real_norm = denormalize(target_img).clamp(0, 1)
            stego_norm = denormalize(stego_img).clamp(0, 1)
            secret_norm = denormalize(secret_img).clamp(0, 1)
            recon_norm = denormalize(recon_secret).clamp(0, 1)

            # --- Update Metrics ---
            
            # 1. FID: Update real and fake distributions
            # Convert to uint8 [0, 255] for FID metric
            real_uint8 = (real_norm * 255).to(torch.uint8)
            stego_uint8 = (stego_norm * 255).to(torch.uint8)
            fid_metric.update(real_uint8, real=True)
            fid_metric.update(stego_uint8, real=False)
            
            # 2. LPIPS (Stego vs Real) - Measures how "different" the stego is from natural images
            # Or Stego vs Generated Non-Stego (if we had that). 
            # Here we measure distance to Target to see if it captures realism.
            lpips_val = lpips_metric(stego_norm, real_norm).mean()
            total_lpips += lpips_val.item()
            
            # 3. Secret Recovery
            total_sec_psnr += psnr_metric(recon_norm, secret_norm).item()
            total_sec_ssim += ssim_metric(recon_norm, secret_norm).item()
            
            # 4. Bit Accuracy
            s_indices_flat = s_indices.view(target_img.shape[0], 4, -1)
            batch_acc = compute_bit_accuracy(pred_indices, s_indices_flat)
            total_bit_acc += batch_acc
            
            num_batches += 1
            
            # Save visual samples for the first batch
            if i == 0:
                vis = torch.cat([real_norm, stego_norm, secret_norm, recon_norm], dim=0)
                save_image(vis, os.path.join(output_dir, "visual_result.png"), nrow=target_img.shape[0])

    # 5. Compute Final Aggregates
    print("Computing final metrics...")
    final_fid = fid_metric.compute().item()
    avg_lpips = total_lpips / num_batches
    avg_sec_psnr = total_sec_psnr / num_batches
    avg_sec_ssim = total_sec_ssim / num_batches
    avg_bit_acc = total_bit_acc / num_batches
    
    print("\n========= GenMamba-Flow Evaluation Results =========")
    print(f"FID Score (Lower is better):        {final_fid:.4f}")
    print(f"LPIPS Score (Lower is better):      {avg_lpips:.4f}")
    print(f"Secret PSNR (Higher is better):     {avg_sec_psnr:.2f} dB")
    print(f"Secret SSIM (Higher is better):     {avg_sec_ssim:.4f}")
    print(f"Bit Accuracy (Higher is better):    {avg_bit_acc*100:.2f}%")
    print("=====================================================")
    
    # Save text report
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"FID: {final_fid}\n")
        f.write(f"LPIPS: {avg_lpips}\n")
        f.write(f"Secret PSNR: {avg_sec_psnr}\n")
        f.write(f"Secret SSIM: {avg_sec_ssim}\n")
        f.write(f"Bit Accuracy: {avg_bit_acc}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of batches for quick test")
    args = parser.parse_args()
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    evaluate_metrics(config, max_batches=args.max_batches)