# demo_generate.py
# 这是一个纯净的“无载体”生成演示
# 证明：输入只有 Prompt 和 Noise，没有 Target Image

import torch
import yaml
import os
from torchvision.utils import save_image

# 引入模型
from models.rqvae import SecretRQVAE
from models.gen_flow import TriStreamMambaUNet, MambaDecoderHead
from modules.text_encoder import TextEncoderWrapper
from utils.helpers import set_seed, denormalize

@torch.no_grad()
def generate_from_scratch():
    # 1. 设置
    set_seed(42) # 固定种子保证复现，实际使用可去掉
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")
    
    # 加载配置
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. 加载模型 (训练好的权重)
    # RQ-VAE
    rqvae = SecretRQVAE(embed_dim=config['rqvae']['embed_dim']).to(device)
    rqvae.load_state_dict(torch.load("checkpoints/stage1_rqvae/best_rqvae.pth", map_location=device))
    rqvae.eval()

    # Generator (Flow Model)
    gen_model = TriStreamMambaUNet(secret_dim=config['rqvae']['embed_dim']).to(device)
    # 注意：这里加载的是训练好的生成器
    gen_model.load_state_dict(torch.load("checkpoints/stage2_flow/gen_epoch_95.pth", map_location=device))
    gen_model.eval()
    
    # Text Encoder
    text_enc = TextEncoderWrapper().to(device)

    # 3. 准备输入 (注意：没有 Target Image!)
    print("准备输入...")
    
    # A. 秘密信息：这里为了演示，随机生成一张秘密图
    # 在实际应用中，这是用户上传的二维码或机密图片
    secret_payload = torch.randn(1, 3, 256, 256).to(device) 
    save_image(denormalize(secret_payload), "results/demo_secret_input.png")
    
    # B. 文本提示
    prompt_text = ["A cyberpunk city with neon lights at night"]
    print(f"Prompt: {prompt_text}")

    # C. 随机噪声 (Latent Noise)
    # 这就是生成的起点，完全随机
    noise_x0 = torch.randn(1, 3, 256, 256).to(device)

    # 4. 核心过程：无载体生成
    print("开始生成 (Flow Matching)...")
    
    # Step 1: 编码秘密信息
    _, _, _, s_feat = rqvae(secret_payload) # 得到秘密特征
    
    # Step 2: 编码文本
    txt_emb = text_enc(prompt_text, device)
    
    # Step 3: ODE 积分 (Euler Solver)
    # 从纯噪声 noise_x0 开始，逐步演化成图像
    x_curr = noise_x0
    steps = 50
    dt = 1.0 / steps
    
    for i in range(steps):
        t = torch.tensor([i / steps]).to(device)
        
        # 模型预测速度场 v
        # 输入：当前图像状态 x_curr, 时间 t, 文本 txt_emb, 秘密 s_feat
        # 绝无 Target Image 参与！
        v_pred, _, _ = gen_model(x_curr, t, txt_emb, s_feat)
        
        # 更新图像状态
        x_curr = x_curr + v_pred * dt

    stego_image = x_curr
    print("生成完成！")

    # 5. 保存结果
    os.makedirs("results", exist_ok=True)
    save_image(denormalize(stego_image), "results/demo_stego_output.png")
    print("隐写图像已保存至 results/demo_stego_output.png")
    print("证明：生成过程未读取任何数据集图片，属于无载体隐写。")

if __name__ == "__main__":
    generate_from_scratch()