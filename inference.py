# inference.py
import torch
import yaml
from torchvision.utils import save_image
from models.rqvae import SecretRQVAE
from models.gen_flow import TriStreamMambaUNet, MambaDecoderHead
from modules.text_encoder import TextEncoderWrapper

@torch.no_grad()
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Configs & Models
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    rqvae = SecretRQVAE(embed_dim=config['rqvae']['embed_dim']).to(device)
    rqvae.load_state_dict(torch.load("checkpoints/rqvae.pth"))
    
    gen_model = TriStreamMambaUNet(secret_dim=config['rqvae']['embed_dim']).to(device)
    gen_model.load_state_dict(torch.load("checkpoints/gen_flow.pth"))
    
    decoder = MambaDecoderHead().to(device)
    decoder.load_state_dict(torch.load("checkpoints/decoder.pth"))
    
    text_enc = TextEncoderWrapper().to(device)
    
    # 1. Inputs
    prompt = ["A futuristic cyberpunk city with neon lights"]
    # 假设有一张秘密图像
    secret_img = torch.randn(1, 3, 256, 256).to(device) 
    
    # 2. Encode Secret
    _, _, _, s_feat = rqvae(secret_img)
    txt_emb = text_enc(prompt, device)
    
    # 3. Generate Stego Image (Euler Solver)
    print("Generating Stego Image...")
    x = torch.randn(1, 3, 256, 256).to(device) # Noise x_0
    dt = 1.0 / 50 # 50 steps
    
    for i in range(50):
        t = torch.tensor([i / 50.0]).to(device)
        v_pred, _, _ = gen_model(x, t, txt_emb, s_feat)
        x = x + v_pred * dt # Euler update
        
    save_image(x, "stego_output.png")
    print("Stego Image Saved to stego_output.png")
    
    # 4. Decrypt (Test)
    print("Decrypting...")
    # 模拟攻击 (可选)
    # x = some_attack(x)
    pred_logits = decoder(x)
    
    # Argmax to get indices
    # pred_logits: [1, 4, 1024, H*W]
    pred_indices = pred_logits.argmax(dim=2) # [1, 4, H*W]
    
    # Reshape back to map
    H_lat = x.shape[2] // 4
    pred_indices_map = pred_indices.view(1, 4, H_lat, H_lat).permute(0, 2, 3, 1) # [1, H, W, 4]
    
    # RQ-VAE Reconstruct
    # 注意: vector-quantize-pytorch 的 get_output_from_indices 需要 [B, Seq, Num_Q] 或做相应转换
    # 这里直接调用 rqvae 内部解码逻辑
    recon_img = rqvae.rq.get_output_from_indices(pred_indices_map)
    recon_img = recon_img.permute(0, 3, 1, 2)
    recon_img = rqvae.decoder(recon_img)
    
    save_image(recon_img, "secret_recovered.png")
    print("Secret Recovered.")

if __name__ == "__main__":
    main()