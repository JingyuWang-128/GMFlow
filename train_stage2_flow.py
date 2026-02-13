# train_stage2_flow.py
import torch
import yaml
import os
import math
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
import torch.nn.functional as F

def train_flow():
    accelerator = Accelerator()
    set_seed(42 + accelerator.process_index)
    
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
        logger.info(f"Accelerator Initialized. Device: {accelerator.device}, Num Processes: {accelerator.num_processes}")
    else:
        logger = None
    
    # --- 1. Load RQ-VAE ---
    rqvae = SecretRQVAE(
        embed_dim=config['rqvae']['embed_dim'],
        num_quantizers=config['rqvae']['num_quantizers']
    )
    
    rqvae_path = os.path.join(base_dir, config['experiment']['checkpoint_dir'], "stage1_rqvae", "best_rqvae.pth")
    if os.path.exists(rqvae_path):
        state_dict = torch.load(rqvae_path, map_location='cpu')
        rqvae.load_state_dict(state_dict)
        if accelerator.is_main_process:
            logger.info(f"Loaded RQ-VAE from {rqvae_path}")
    else:
        raise FileNotFoundError("RQ-VAE checkpoint is required!")
    
    rqvae.to(accelerator.device) 
    rqvae.eval()
    rqvae.requires_grad_(False)
    
    # 提取 Codebooks
    codebooks = []
    if hasattr(rqvae.rq, 'layers'):
        for layer in rqvae.rq.layers:
            codebooks.append(layer._codebook.embed.detach()) 
    else:
        codebooks = rqvae.rq.codebook.detach()
    
    # --- 2. Generator & Decoder ---
    gen_model = TriStreamMambaUNet(secret_dim=config['rqvae']['embed_dim'])
    
    decoder = MambaDecoderHead(
        num_quantizers=config['rqvae']['num_quantizers'],
        embed_dim=config['rqvae']['embed_dim'] 
    )
    
    text_enc = TextEncoderWrapper().to(accelerator.device)
    interfere = InterferenceOperatorSet().to(accelerator.device)
    
    opt_gen = AdamW(gen_model.parameters(), lr=float(config['gen_flow']['learning_rate']))
    opt_dec = AdamW(decoder.parameters(), lr=float(config['gen_flow']['learning_rate']))
    
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    gen_model, decoder, opt_gen, opt_dec, train_loader, val_loader = accelerator.prepare(
        gen_model, decoder, opt_gen, opt_dec, train_loader, val_loader
    )
    
    epochs = config['gen_flow']['epochs']
    
    # --- [核心修改] 三阶段课程学习参数配置 ---
    STAGE1_END = 200  # 纯生成阶段结束点
    STAGE2_END = 600  # 混合爬坡阶段结束点 (给足时间慢慢适应)
    
    # 基础权重
    base_lambda_flow = 5.0     # 始终保持较高的生成权重，防止画崩
    base_lambda_rsmi = 0.1
    target_lambda_robust = 1.0 # 最终隐写权重的目标值

    # --- Training Loop ---
    for epoch in range(epochs):
        gen_model.train()
        decoder.train()
        
        # --- [核心修改] 动态计算当前权重 ---
        if epoch < STAGE1_END:
            # 第一阶段：专注画质
            current_lambda_flow = base_lambda_flow
            current_lambda_robust = 0.0
            stage_name = "Stage 1: Image Gen Warmup"
            
        elif epoch < STAGE2_END:
            # 第二阶段：线性增加隐写权重 (Curriculum)
            # progress 从 0.0 变到 1.0
            progress = (epoch - STAGE1_END) / (STAGE2_END - STAGE1_END)
            current_lambda_flow = base_lambda_flow
            current_lambda_robust = target_lambda_robust * progress
            stage_name = f"Stage 2: Stego Curriculum ({progress*100:.1f}%)"
            
        else:
            # 第三阶段：完全体联合训练
            current_lambda_flow = base_lambda_flow
            current_lambda_robust = target_lambda_robust
            stage_name = "Stage 3: Full Joint Training"

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} | {stage_name} | Robust Weight: {current_lambda_robust:.4f}")

        tqdm_bar = tqdm(
            train_loader, 
            desc=f"Ep {epoch+1}", 
            disable=not accelerator.is_main_process,
            leave=False
        )
        
        for i, (target_img, secret_img, prompts) in enumerate(tqdm_bar):
            # Inputs
            with torch.no_grad():
                txt_emb = text_enc(prompts, accelerator.device)
                _, s_indices, _, s_feat = rqvae(secret_img)
            
            # Flow Setup
            B = target_img.shape[0]
            t = torch.rand(B).to(accelerator.device)
            x_0 = torch.randn_like(target_img).to(accelerator.device)
            x_1 = target_img
            t_view = t.view(B, 1, 1, 1)
            x_t = t_view * x_1 + (1 - t_view) * x_0 
            
            # Forward
            v_pred, h_struc, h_tex = gen_model(x_t, t, txt_emb, s_feat)
            
            # Losses
            loss_flow = GenMambaLosses.flow_matching_loss(v_pred, x_0, x_1)
            loss_rsmi = GenMambaLosses.rSMI_loss(h_struc, h_tex)
            
            # Robust Decoding Loss
            # 只有当权重 > 0 时才计算，节省 Stage 1 的算力
            if current_lambda_robust > 0:
                x_1_pred = x_t + (1 - t_view) * v_pred 
                x_attacked = interfere(x_1_pred, severity='strong')
                pred_features = decoder(x_attacked, target_shape=s_indices.shape[-2:])
                loss_robust = GenMambaLosses.hDCE_loss(
                    pred_features, s_indices, codebooks, temperature=0.1
                )
            else:
                loss_robust = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
            
            # Total Loss
            total_loss = (current_lambda_flow * loss_flow + 
                          base_lambda_rsmi * loss_rsmi + 
                          current_lambda_robust * loss_robust)
            
            opt_gen.zero_grad()
            opt_dec.zero_grad()
            accelerator.backward(total_loss)
            
            # 梯度裁剪
            accelerator.clip_grad_norm_(gen_model.parameters(), 1.0)
            accelerator.clip_grad_norm_(decoder.parameters(), 1.0)
            
            opt_gen.step()
            opt_dec.step()
            
            if accelerator.is_main_process:
                tqdm_bar.set_postfix(
                    flow=f"{loss_flow.item():.2f}",
                    robust=f"{loss_robust.item():.2f}",
                    w_rob=f"{current_lambda_robust:.2f}"
                )
        
        accelerator.wait_for_everyone()
        
        if epoch % 5 == 0:
            validate_multigpu(accelerator, gen_model, decoder, rqvae, val_loader, text_enc, interfere, codebooks, epoch, res_dir, logger, config)
            
            if accelerator.is_main_process:
                unwrapped_gen = accelerator.unwrap_model(gen_model)
                unwrapped_dec = accelerator.unwrap_model(decoder)
                torch.save(unwrapped_gen.state_dict(), os.path.join(ckpt_dir, f"gen_epoch_{epoch}.pth"))
                torch.save(unwrapped_dec.state_dict(), os.path.join(ckpt_dir, f"dec_epoch_{epoch}.pth"))
    
    if accelerator.is_main_process:
        logger.info("Training Finished.")

def validate_multigpu(accelerator, gen_model, decoder, rqvae, val_loader, text_enc, interfere, codebooks, epoch, res_dir, logger, config):
    gen_model.eval()
    decoder.eval()
    
    local_metrics = {'psnr_sum': 0.0, 'bit_acc_sum': 0.0, 'count': 0}
    
    with torch.no_grad():
        for i, (val_target, val_secret, val_prompts) in enumerate(val_loader):
            txt_emb = text_enc(val_prompts, accelerator.device)
            _, s_indices, _, s_feat = rqvae(val_secret)
            
            # Sampling
            x_curr = torch.randn_like(val_target).to(accelerator.device)
            steps = 20
            dt = 1.0 / steps
            for k in range(steps):
                t = torch.tensor([k/steps]).to(accelerator.device)
                v_pred, _, _ = gen_model(x_curr, t, txt_emb, s_feat)
                x_curr = x_curr + v_pred * dt
            
            x_stego = x_curr
            x_attacked = interfere(x_stego, severity='weak')
            
            # Decode
            pred_features = decoder(x_attacked, target_shape=s_indices.shape[-2:])
            
            # Retrieval
            B, Q, C, H, W = pred_features.shape
            pred_indices_list = []
            
            for q in range(Q):
                feat_q = pred_features[:, q].permute(0, 2, 3, 1).reshape(-1, C)
                feat_q = F.normalize(feat_q, dim=1)
                
                if isinstance(codebooks, list):
                    cb = codebooks[q]
                elif codebooks.ndim == 3:
                    cb = codebooks[q]
                else:
                    cb = codebooks
                
                if cb.ndim == 3: cb = cb.squeeze(0)
                if cb.ndim != 2 and cb.shape[-1] == C: cb = cb.view(-1, C)
                cb = F.normalize(cb, dim=1)
                
                idx = torch.matmul(feat_q, cb.transpose(0, 1)).argmax(dim=1).view(B, H, W)
                pred_indices_list.append(idx)
                
            pred_indices = torch.stack(pred_indices_list, dim=1)
            
            # Metrics
            batch_psnr = compute_psnr(val_target, x_stego)
            batch_acc = compute_bit_accuracy(pred_indices, s_indices)
            
            local_metrics['psnr_sum'] += batch_psnr
            local_metrics['bit_acc_sum'] += batch_acc
            local_metrics['count'] += 1
            
            if i == 0 and accelerator.is_main_process:
                # Vis
                idxs = pred_indices.permute(0, 2, 3, 1)
                recon_secret_feat = rqvae.rq.get_output_from_indices(idxs).permute(0, 3, 1, 2)
                recon_secret = rqvae.decoder(recon_secret_feat)
                vis = torch.cat([val_target, x_stego, val_secret, recon_secret], dim=0)
                save_image_grid(vis, os.path.join(res_dir, f"val_{epoch}.png"), nrow=B)
            
            if i > 10: break

    # Gather Metrics
    metrics_tensor = torch.tensor([
        local_metrics['psnr_sum'], 
        local_metrics['bit_acc_sum'], 
        local_metrics['count']
    ], device=accelerator.device)
    
    gathered_metrics = accelerator.gather(metrics_tensor)
    
    if accelerator.is_main_process:
        if gathered_metrics.ndim == 1:
             gathered_metrics = gathered_metrics.view(-1, 3)
        
        total_psnr = gathered_metrics[:, 0].sum().item()
        total_acc = gathered_metrics[:, 1].sum().item()
        total_count = gathered_metrics[:, 2].sum().item()
        
        if total_count > 0:
            avg_psnr = total_psnr / total_count
            avg_acc = total_acc / total_count
            logger.info(f"VAL Epoch {epoch} (Global): Stego PSNR {avg_psnr:.2f} | Bit Acc {avg_acc:.4f}")

if __name__ == "__main__":
    train_flow()