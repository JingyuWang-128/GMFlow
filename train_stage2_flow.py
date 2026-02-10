# train_stage2_flow.py
import torch
import yaml
import os
# 设置国内镜像，防止连接 HuggingFace 超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torch.optim import AdamW
from modules.text_encoder import TextEncoderWrapper
from models.rqvae import SecretRQVAE
from models.gen_flow import TriStreamMambaUNet, MambaDecoderHead
from modules.losses import GenMambaLosses
from modules.interference import InterferenceOperatorSet
from data.dataset import get_dataloader
from utils.helpers import set_seed, get_logger, save_image_grid, compute_psnr, compute_bit_accuracy
from tqdm import tqdm
from accelerate import Accelerator

def train_flow():
    # 1. 初始化 Accelerator
    accelerator = Accelerator()
    
    # 设置随机种子 (各进程不同)
    set_seed(42 + accelerator.process_index)
    
    # 2. 路径与配置 setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'configs/config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    log_dir = os.path.join(base_dir, config['experiment']['log_dir'], "stage2_flow")
    ckpt_dir = os.path.join(base_dir, config['experiment']['checkpoint_dir'], "stage2_flow")
    res_dir = os.path.join(base_dir, config['experiment']['result_dir'], "stage2_flow")
    
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        logger = get_logger(log_dir, "GenFlow_Train")
        logger.info(f"Accelerator Initialized. Device: {accelerator.device}")
    else:
        logger = None
    
    # 3. Models Initialization
    # A. RQ-VAE (Frozen)
    # [修复] 显式传入 num_quantizers
    rqvae = SecretRQVAE(
        embed_dim=config['rqvae']['embed_dim'],
        num_quantizers=config['rqvae']['num_quantizers']
    )
    rqvae.to(accelerator.device) 
    
    rqvae_path = os.path.join(base_dir, config['experiment']['checkpoint_dir'], "stage1_rqvae", "best_rqvae.pth")
    if os.path.exists(rqvae_path):
        rqvae.load_state_dict(torch.load(rqvae_path, map_location=accelerator.device))
        if accelerator.is_main_process:
            logger.info(f"Loaded RQ-VAE from {rqvae_path}")
    else:
        if accelerator.is_main_process:
            logger.warning("RQ-VAE checkpoint not found! Using random initialization.")
    rqvae.eval()
    rqvae.requires_grad_(False)
    
    # B. Generator & Decoder
    gen_model = TriStreamMambaUNet(secret_dim=config['rqvae']['embed_dim'])
    decoder = MambaDecoderHead(num_quantizers=config['rqvae']['num_quantizers'])
    
    # C. Helpers
    text_enc = TextEncoderWrapper().to(accelerator.device)
    interfere = InterferenceOperatorSet().to(accelerator.device)
    
    # 4. Optimizers
    opt_gen = AdamW(gen_model.parameters(), lr=float(config['gen_flow']['learning_rate']))
    opt_dec = AdamW(decoder.parameters(), lr=float(config['gen_flow']['learning_rate']))
    
    # 5. Dataloaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    # 6. Prepare with Accelerator
    gen_model, decoder, opt_gen, opt_dec, train_loader, val_loader = accelerator.prepare(
        gen_model, decoder, opt_gen, opt_dec, train_loader, val_loader
    )
    
    epochs = config['gen_flow']['epochs']
    
    # Loss Weights
    lambda_flow = config['gen_flow']['lambda_flow']
    lambda_rsmi = config['gen_flow']['lambda_rsmi']
    lambda_robust_base = config['gen_flow']['lambda_robust']

    # [Warmup] 定义预热轮数
    WARMUP_EPOCHS = 5

    for epoch in range(epochs):
        gen_model.train()
        decoder.train()
        
        # 动态调整 lambda_robust (Warmup)
        if epoch < WARMUP_EPOCHS:
            current_lambda_robust = 0.0
            if accelerator.is_main_process and epoch == 0:
                logger.info(f"Warmup: robust loss disabled for first {WARMUP_EPOCHS} epochs.")
        else:
            current_lambda_robust = lambda_robust_base

        total_loss_meter = 0
        robust_loss_meter = 0
        
        tqdm_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs} [Flow Train]", 
            disable=not accelerator.is_main_process,
            leave=False
        )
        
        for i, (target_img, secret_img, prompts) in enumerate(tqdm_bar):
            # 1. Prepare Inputs
            with torch.no_grad():
                txt_emb = text_enc(prompts, accelerator.device)
                _, s_indices, _, s_feat = rqvae(secret_img)
                # 获取真实的 Latent 尺寸 [B, Q, H, W]
                B_idx, Q_idx, H_lat, W_lat = s_indices.shape
            
            # 2. Flow Matching Training
            B = target_img.shape[0]
            t = torch.rand(B).to(accelerator.device)
            x_0 = torch.randn_like(target_img).to(accelerator.device)
            t_view = t.view(B, 1, 1, 1)
            x_t = t_view * target_img + (1 - t_view) * x_0
            
            # 3. Forward
            v_pred, h_struc, h_tex = gen_model(x_t, t, txt_emb, s_feat)
            
            # 4. Losses
            loss_flow = GenMambaLosses.flow_matching_loss(v_pred, x_0, target_img)
            loss_rsmi = GenMambaLosses.rSMI_loss(h_struc, h_tex)
            
            # 5. Robustness Guidance
            x_0_pred = x_t - (1 - t_view) * v_pred 
            x_attacked = interfere(x_0_pred, severity='strong') 
            
            # [关键] 传入目标分辨率，确保对齐
            # 注意：这需要 models/gen_flow.py 中的 MambaDecoderHead 已经更新支持 target_shape
            pred_logits = decoder(x_attacked, target_shape=(H_lat, W_lat))
            
            s_indices_flat = s_indices.view(B_idx, Q_idx, -1)
            loss_robust = GenMambaLosses.robust_decode_loss(pred_logits, s_indices_flat)
            
            total_loss = lambda_flow * loss_flow + lambda_rsmi * loss_rsmi + current_lambda_robust * loss_robust
            
            opt_gen.zero_grad()
            opt_dec.zero_grad()
            accelerator.backward(total_loss)
            opt_gen.step()
            opt_dec.step()
            
            total_loss_meter += total_loss.item()
            robust_loss_meter += loss_robust.item()

            if accelerator.is_main_process:
                tqdm_bar.set_postfix(
                    total=f"{total_loss.item():.4f}",
                    robust=f"{loss_robust.item():.4f}"
                )
                if i % 50 == 0:
                    logger.info(f"Epoch [{epoch}][{i}] Total: {total_loss.item():.4f} Robust: {loss_robust.item():.4f}")
        
        accelerator.wait_for_everyone()
        
        # Validation Loop
        if epoch % 5 == 0:
            gen_model.eval()
            decoder.eval()
            val_stego_psnr = 0
            val_bit_acc = 0
            
            with torch.no_grad():
                for val_batch_idx, (val_target, val_secret, val_prompts) in enumerate(val_loader):
                    # Prepare Inputs
                    txt_emb = text_enc(val_prompts, accelerator.device)
                    _, s_indices, _, s_feat = rqvae(val_secret)
                    B_val, Q_val, H_lat_val, W_lat_val = s_indices.shape
                    
                    # Euler Sampling
                    x_gen = torch.randn_like(val_target).to(accelerator.device)
                    dt = 1.0 / 20
                    for k in range(20):
                        t_val = torch.tensor([k/20.0]).to(accelerator.device)
                        v_val, _, _ = gen_model(x_gen, t_val, txt_emb, s_feat)
                        x_gen = x_gen + v_val * dt
                    
                    # Metrics
                    val_stego_psnr += compute_psnr(val_target, x_gen)
                    
                    # Attack & Decode
                    x_val_attacked = interfere(x_gen, severity='weak')
                    
                    # [关键] 对齐分辨率
                    logits = decoder(x_val_attacked, target_shape=(H_lat_val, W_lat_val))
                    pred_idx = logits.argmax(dim=2) # [B, Q, N]
                    
                    s_indices_flat = s_indices.view(B_val, Q_val, -1)
                    val_bit_acc += compute_bit_accuracy(pred_idx, s_indices_flat)
                    
                    # Visualization (Only first batch of main process)
                    if val_batch_idx == 0 and accelerator.is_main_process and epoch % 10 == 0:
                        # [修复] 动态获取 num_quantizers
                        num_q_vis = config['rqvae']['num_quantizers']
                        
                        # 安全检查
                        model_q = getattr(rqvae.rq, 'num_quantizers', num_q_vis)
                        
                        if model_q != num_q_vis:
                            logger.warning(f"Visualization Skipped: Config Q={num_q_vis} != Model Q={model_q}")
                        else:
                            try:
                                # pred_idx: [B, Q, N]
                                # 转换为 [B, N, Q] 供 RQ-VAE 解码
                                pred_indices = pred_idx.permute(0, 2, 1).contiguous()
                                
                                # RQ-VAE Decode -> 得到 [B, N, C]
                                recon_secret_feat = rqvae.rq.get_output_from_indices(pred_indices)
                                
                                # [关键修复]: 先 view 回 [B, H, W, C] 再 permute
                                # 之前的错误是直接对 3D 张量 [B, N, C] 做了 4D 的 permute
                                recon_secret_feat = recon_secret_feat.view(B_val, H_lat_val, W_lat_val, -1)
                                recon_secret_feat = recon_secret_feat.permute(0, 3, 1, 2).contiguous()
                                
                                # Decoder
                                recon_secret = rqvae.decoder(recon_secret_feat)
                                
                                vis_row = torch.cat([val_target, x_gen, val_secret, recon_secret], dim=0)
                                save_path = os.path.join(res_dir, f"val_step_{epoch}.png")
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                save_image_grid(vis_row, save_path, nrow=B_val)
                            except Exception as e:
                                # 捕获异常防止中断训练
                                logger.warning(f"Visualization Error: {e}")

                    break # 只验证一个 batch
            
            if accelerator.is_main_process:
                logger.info(f"VAL Epoch {epoch}: Stego PSNR: {val_stego_psnr:.2f} | Bit Acc: {val_bit_acc:.4f}")
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_gen = accelerator.unwrap_model(gen_model)
                unwrapped_dec = accelerator.unwrap_model(decoder)
                
                torch.save(unwrapped_gen.state_dict(), os.path.join(ckpt_dir, f"gen_epoch_{epoch}.pth"))
                torch.save(unwrapped_dec.state_dict(), os.path.join(ckpt_dir, f"dec_epoch_{epoch}.pth"))

    if accelerator.is_main_process:
        logger.info("Training Finished.")

if __name__ == "__main__":
    train_flow()