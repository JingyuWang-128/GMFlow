import torch
import yaml
import os
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
from accelerate import Accelerator # [新增]

def train_flow():
    # 1. 初始化 Accelerator
    accelerator = Accelerator()
    
    # 设置随机种子 (各进程不同)
    set_seed(42 + accelerator.process_index)
    
    # 2. 路径与配置 setup
    # 使用绝对路径，避免 accelerate 改变工作目录导致的问题
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'configs/config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 定义输出目录
    log_dir = os.path.join(base_dir, config['experiment']['log_dir'], "stage2_flow")
    ckpt_dir = os.path.join(base_dir, config['experiment']['checkpoint_dir'], "stage2_flow")
    res_dir = os.path.join(base_dir, config['experiment']['result_dir'], "stage2_flow")
    
    # 仅主进程创建目录和 Logger
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)
        logger = get_logger(log_dir, "GenFlow_Train")
        logger.info(f"Accelerator Initialized. Device: {accelerator.device}")
    else:
        logger = None
    
    # 3. Models Initialization
    # A. RQ-VAE (Frozen) - 不需要训练，所以直接移到 accelerator 的设备即可
    # 注意：不要对冻结模型使用 prepare，否则 DDP 包装会浪费显存和通信
    rqvae = SecretRQVAE(embed_dim=config['rqvae']['embed_dim'])
    rqvae.to(accelerator.device) 
    
    rqvae_path = os.path.join(base_dir, config['experiment']['checkpoint_dir'], "stage1_rqvae", "best_rqvae.pth")
    if os.path.exists(rqvae_path):
        rqvae.load_state_dict(torch.load(rqvae_path, map_location='cpu')) # 显式 map_location
        if accelerator.is_main_process:
            logger.info(f"Loaded RQ-VAE from {rqvae_path}")
    else:
        if accelerator.is_main_process:
            logger.warning("RQ-VAE checkpoint not found! Using random initialization.")
    rqvae.eval()
    rqvae.requires_grad_(False) # 彻底冻结
    
    # B. Generator & Decoder (需训练)
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
    # 使用 'val' split
    val_loader = get_dataloader(config, split='val')
    
    # 6. Prepare with Accelerator
    # 这一步会自动处理 DDP 包装和数据分片
    gen_model, decoder, opt_gen, opt_dec, train_loader, val_loader = accelerator.prepare(
        gen_model, decoder, opt_gen, opt_dec, train_loader, val_loader
    )
    
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
        
        # 仅主进程显示进度条
        tqdm_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs} [Flow Train]", 
            disable=not accelerator.is_main_process,
            leave=False
        )
        
        for i, (target_img, secret_img, prompts) in enumerate(tqdm_bar):
            # accelerator.prepare 已经处理了 target_img 和 secret_img 的设备
            
            # 1. Prepare Inputs
            with torch.no_grad():
                txt_emb = text_enc(prompts, accelerator.device)
                _, s_indices, _, s_feat = rqvae(secret_img)
            
            # 2. Flow Matching Training
            B = target_img.shape[0]
            t = torch.rand(B).to(accelerator.device)
            x_0 = torch.randn_like(target_img).to(accelerator.device) # Noise
            t_view = t.view(B, 1, 1, 1)
            x_t = t_view * target_img + (1 - t_view) * x_0 # Interpolation
            
            # 3. Forward
            v_pred, h_struc, h_tex = gen_model(x_t, t, txt_emb, s_feat)
            
            # 4. Losses
            loss_flow = GenMambaLosses.flow_matching_loss(v_pred, x_0, target_img)
            loss_rsmi = GenMambaLosses.rSMI_loss(h_struc, h_tex)
            
            # 5. Robustness Guidance
            x_0_pred = x_t - (1 - t_view) * v_pred 
            x_attacked = interfere(x_0_pred, severity='strong') 
            pred_logits = decoder(x_attacked)
            
            s_indices_flat = s_indices.view(B, config['rqvae']['num_quantizers'], -1)

            # print("[DEBUG][Train] pred_logits:", pred_logits.shape)
            # print("[DEBUG][Train] s_indices:", s_indices_flat.shape)

            # assert pred_logits.shape[-1] == s_indices_flat.shape[-1], (
            #     f"[FATAL] Token mismatch: "
            #     f"pred={pred_logits.shape[-1]} "
            #     f"target={s_indices_flat.shape[-1]}"
            # )

            loss_robust = GenMambaLosses.robust_decode_loss(pred_logits, s_indices_flat)
            
            total_loss = lambda_flow * loss_flow + lambda_rsmi * loss_rsmi + lambda_robust * loss_robust
            
            opt_gen.zero_grad()
            opt_dec.zero_grad()
            
            # [关键] 使用 accelerator 反向传播
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
        
        # 等待所有进程跑完当前 Epoch
        accelerator.wait_for_everyone()
        
        # Validation Loop
        if epoch % 5 == 0:
            gen_model.eval()
            decoder.eval()
            val_stego_psnr = 0
            val_bit_acc = 0
            
            # 验证也只跑一小部分，避免太慢
            # 注意：在多卡下，break 可能导致死锁，我们这里只取 val_loader 的第一个 batch
            # 最好的方式是设置 max_batches
            
            with torch.no_grad():
                for val_batch_idx, (val_target, val_secret, val_prompts) in enumerate(val_loader):
                    # Prepare Inputs
                    txt_emb = text_enc(val_prompts, accelerator.device)
                    _, s_indices, _, s_feat = rqvae(val_secret)
                    
                    # Euler Sampling (20 steps)
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
                    logits = decoder(x_val_attacked)
                    pred_idx = logits.argmax(dim=2)
                    
                    s_indices_flat = s_indices.view(s_indices.shape[0], config['rqvae']['num_quantizers'], -1)
                    val_bit_acc += compute_bit_accuracy(pred_idx, s_indices_flat)
                    
                    # Visualization (Only first batch of main process)
                    if val_batch_idx == 0 and accelerator.is_main_process and epoch % 10 == 0:
                        H_lat = val_secret.shape[2] // 4
                        pred_map = pred_idx.view(val_target.shape[0], 4, H_lat, H_lat).permute(0, 3, 1, 2)
                        
                        B, Q, H, W = pred_map.shape
                        assert Q == rqvae.rq.num_quantizers, "RQ层数不一致"

                        # [B, Q, H, W] → [B, Q, N]
                        pred_indices = pred_map.view(B, Q, H * W)

                        # [B, Q, N] → [B, N, Q]
                        pred_indices = pred_indices.permute(0, 2, 1).contiguous()

                        # 送入 RQ-VAE 解码
                        recon_secret = rqvae.rq.get_output_from_indices(pred_indices)

                        recon_secret = recon_secret.permute(0, 3, 1, 2)
                        recon_secret = rqvae.decoder(recon_secret)
                        
                        vis_row = torch.cat([val_target, x_gen, val_secret, recon_secret], dim=0)
                        
                        save_path = os.path.join(res_dir, f"val_step_{epoch}.png")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        save_image_grid(vis_row, save_path, nrow=val_target.shape[0])
                    
                    # 只验证一个 batch 以节省时间
                    break 
            
            # 记录日志 (仅 rank 0)
            if accelerator.is_main_process:
                logger.info(f"VAL Epoch {epoch}: Stego PSNR: {val_stego_psnr:.2f} | Bit Acc: {val_bit_acc:.4f}")
            
            # 保存模型
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Unwrap models to save clean state_dict
                unwrapped_gen = accelerator.unwrap_model(gen_model)
                unwrapped_dec = accelerator.unwrap_model(decoder)
                
                torch.save(unwrapped_gen.state_dict(), os.path.join(ckpt_dir, f"gen_epoch_{epoch}.pth"))
                torch.save(unwrapped_dec.state_dict(), os.path.join(ckpt_dir, f"dec_epoch_{epoch}.pth"))

    if accelerator.is_main_process:
        logger.info("Training Finished.")

if __name__ == "__main__":
    train_flow()