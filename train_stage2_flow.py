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
from utils.helpers import set_seed, save_image_grid, compute_psnr, compute_ssim, AverageMeter

def train_flow():
    set_seed(42)
    # 0. Setup
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = config['experiment']['device']
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Models
    # A. 冻结的 RQ-VAE
    rqvae = SecretRQVAE(embed_dim=config['rqvae']['embed_dim']).to(device)
    if os.path.exists("checkpoints/rqvae.pth"):
        rqvae.load_state_dict(torch.load("checkpoints/rqvae.pth"))
    rqvae.eval()
    
    # B. 生成器
    gen_model = TriStreamMambaUNet(secret_dim=config['rqvae']['embed_dim']).to(device)
    
    # C. 解码器 (需联合训练以适应干扰)
    decoder = MambaDecoderHead().to(device)
    
    # D. 辅助模块
    text_enc = TextEncoderWrapper().to(device)
    interfere = InterferenceOperatorSet().to(device)
    
    # 2. Optimizer
    opt_gen = AdamW(gen_model.parameters(), lr=float(config['gen_flow']['learning_rate']))
    opt_dec = AdamW(decoder.parameters(), lr=float(config['gen_flow']['learning_rate']))
    
    loader = get_dataloader(config)
    
    print("Stage 2: Training GenMamba-Flow...")
    
    for epoch in range(10):
        for target_img, secret_img, prompts in loader:
            target_img = target_img.to(device) # x_1
            secret_img = secret_img.to(device)
            
            # --- 准备数据 ---
            with torch.no_grad():
                # 1. 文本编码
                txt_emb = text_enc(prompts, device)
                # 2. 秘密离散化 (获取 quantized feature 和 indices)
                _, s_indices, _, s_feat = rqvae(secret_img)
                # s_feat: [B, Dim, H', W']
                # s_indices: [B, Num_Q, H', W'] (Target for decoder)
            
            # --- Rectified Flow Training ---
            # 随机采样时间步 t
            B = target_img.shape[0]
            t = torch.rand(B).to(device)
            x_0 = torch.randn_like(target_img).to(device) # Noise
            
            # 插值得到 x_t
            t_view = t.view(B, 1, 1, 1)
            x_t = t_view * target_img + (1 - t_view) * x_0
            
            # --- Generator Forward ---
            # 预测速度场 v 和中间特征
            v_pred, h_struc, h_tex = gen_model(x_t, t, txt_emb, s_feat)
            
            # --- Loss 1: Flow Matching (生成质量) ---
            loss_flow = GenMambaLosses.flow_matching_loss(v_pred, x_0, target_img)
            
            # --- Loss 2: rSMI (保真度 - 结构纹理对齐) ---
            loss_rsmi = GenMambaLosses.rSMI_loss(h_struc, h_tex)
            
            # --- Loss 3: Interference Robustness (内生鲁棒性) ---
            # 1. 估计 x_0 (去噪图)
            x_0_pred = x_t - (1 - t_view) * v_pred
            
            # 2. 施加可微干扰 (模拟攻击)
            x_attacked = interfere(x_0_pred, severity='strong')
            
            # 3. 尝试解码
            pred_logits = decoder(x_attacked) # [B, Num_Q, Codebook, Seq]
            
            # 4. 计算解码 Loss
            # Flatten indices for Loss
            s_indices_flat = s_indices.view(B, 4, -1) # [B, 4, Seq]
            loss_robust = GenMambaLosses.robust_decode_loss(pred_logits, s_indices_flat)
            
            # --- Optimization ---
            total_loss = loss_flow + 0.1 * loss_rsmi + 0.5 * loss_robust
            
            opt_gen.zero_grad()
            opt_dec.zero_grad()
            total_loss.backward()
            opt_gen.step()
            opt_dec.step()
            
        print(f"Epoch {epoch}: Total {total_loss.item():.4f} | Robust {loss_robust.item():.4f}")
        
    torch.save(gen_model.state_dict(), "checkpoints/gen_flow.pth")
    torch.save(decoder.state_dict(), "checkpoints/decoder.pth")

if __name__ == "__main__":
    train_flow()