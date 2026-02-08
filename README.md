# ---

**论文标题：GenMamba-Flow：基于解耦 Mamba 流与干扰流形引导的鲁棒生成式隐写术**

**Title:** GenMamba-Flow: Robust Generative Steganography via Decoupled Mamba Streams and Interference Manifold Guidance

## ---

**1\. 引言与创新点 (Introduction & Contributions)**

现有的生成式隐写术（Generative Steganography）虽然避免了传统载体修改留下的统计痕迹，但仍面临三大核心痛点：(1) **生成的不可控性与保真度矛盾**：难以在嵌入高容量信息的同时保持图像纹理和语义的自然一致性；(2) **鲁棒性的滞后性**：通常依赖生成后的后处理（Post-processing）或额外的修复网络来应对攻击，缺乏内生防御能力；(3) **秘密信息的重构质量瓶颈**：难以在有损信道中实现秘密图像的像素级完美还原。

针对上述问题，本文提出 **GenMamba-Flow**，一种无需载体的、基于整流流（Rectified Flow）与状态空间模型（Mamba）的鲁棒隐写框架。本文的 **三个主要创新点** 如下：

1. **内生鲁棒性：基于干扰流形引导的生成对抗传输 (Intrinsic Robustness via Interference Manifold Guidance)**  
   * **针对痛点**：传统方法“先生成后修复”的流程割裂，且抗攻击能力有限。  
   * **创新**：我们将鲁棒性构建为生成过程中的内生属性。构建一个可微的“干扰算子集流形”（Interference Operator Manifold），在 Rectified Flow 的 ODE 积分轨迹中引入对抗性梯度场。这使得生成的图像轨迹被强制约束在“既符合语义分布又对特定干扰算子（如 JPEG、噪声）具有不变性”的流形区域，实现了“生成即防御”。  
2. **高保真性：基于 rSMI 约束的三流 Mamba 解耦注入 (High Fidelity via Tri-Stream Mamba Decoupling with rSMI)**  
   * **针对痛点**：直接嵌入信息容易破坏图像的全局结构或产生语义伪影。  
   * **创新**：利用 Mamba 的线性复杂度优势，设计了“语义-结构-纹理”三流解耦架构。利用 **相对平方互信息（rSMI）** 作为统计对齐约束，强制嵌入了秘密信息的“纹理流”在统计分布上与“结构流”保持最大互信息。这确保了隐写操作仅改变局部高频细节的微观状态，而严格保留宏观语义和结构，实现了极致的视觉保真。  
3. **高还原度：基于残差量化与对比检索的离散恢复 (High Recovery via RQ-VAE & Contrastive Retrieval)**  
   * **针对痛点**：连续值的回归预测在噪声下极不稳定，难以复原高质量秘密图像。  
   * **创新**：引入 **RQ-VAE（Residual Quantized VAE）** 将秘密图像转化为多层离散残差码本。配合 **语义辅助的硬负样本采样（Hard Negative Sampling）** 训练 Mamba 解码器，将解码问题从“像素回归”转化为“特征检索”。即使在强干扰下，只要能检索出正确的离散 Token 索引，利用残差码本的叠加特性，即可实现从语义轮廓到像素细节的分级完美重建。

## ---

**2\. 相关工作 (Related Work)**

本研究主要涉及以下四个领域的交叉融合：

* **生成式隐写术 (Generative Steganography)**：涵盖基于 GAN 和 Diffusion Model 的无载体隐写，特别是最近利用文本引导生成的隐写方法。  
* **状态空间模型 (State Space Models, SSMs)**：重点关注 Mamba 及其 2D 视觉变体（Vision Mamba/Vim）在处理长序列依赖和高分辨率特征提取中的应用。  
* **基于流的生成模型 (Flow-Based Models)**：特别是 Rectified Flow (整流流) 和 Conditional Flow Matching，它们比传统 Diffusion 提供更直且高效的生成轨迹。  
* **矢量量化与图像重建 (Vector Quantization)**：涉及 VQ-VAE 及其改进版 RQ-VAE（Residual Quantized VAE），用于将图像映射为离散的高压缩比表征。

## ---

**3\. 方法论 (Methodology)**

### **3.1 整体架构**

GenMamba-Flow 包含三个核心组件：(1) **秘密预处理器**（基于 RQ-VAE 的离散化）；(2) **生成器**（基于三流 Mamba 的 Rectified Flow）；(3) **鲁棒解码器**（基于对比检索）。

### **3.2 秘密预处理：残差离散化**

为了实现像素级还原，我们使用 RQ-VAE 将秘密图像 $I\_{sec}$ 编码为深度为 $D$ 的离散索引图 $S \\in \\{1, \\dots, K\\}^{D \\times H' \\times W'}$。

秘密特征 $Z$ 被近似为 $D$ 个残差码本向量的和：

$$Z \\approx \\sum\_{d=1}^{D} \\mathbf{e}\_{k\_d}^{(d)}, \\quad \\text{where } \\mathbf{e}\_{k\_d}^{(d)} \\in \\mathbb{R}^{C} \\text{ is the chosen code from depth } d$$  
这允许我们将复杂的图像信息分解为从“粗糙语义”（Depth 1）到“精细残差”（Depth $D$）的层级序列。

### **3.3 生成器：三流 Mamba U-Net (Tri-Stream Mamba)**

我们在 Rectified Flow 的预测网络 $v\_\\theta$ 中引入三流解耦机制。假设输入状态为 $x\_t$，时间步为 $t$，文本条件为 $c\_{txt}$，秘密特征为 $f\_{sec}$。

#### **3.3.1 状态空间方程的解耦**

Mamba 的核心是选择性状态空间模型。在我们的三流块中，隐状态 $h$ 被解耦为三个独立但交互的分量：

$$\\begin{aligned} h\_{sem} &= \\text{SSM}\_{sem}(x\_t, \\Delta, A, B, C; c\_{txt}) \\quad (\\text{Frozen, Semantic Guidance}) \\\\ h\_{struc} &= \\text{SSM}\_{struc}(x\_t \+ h\_{sem}, \\dots) \\quad (\\text{Structure Formation}) \\\\ h\_{tex} &= \\text{SSM}\_{tex}(h\_{struc}, \\dots) \\oplus \\mathcal{M}(f\_{sec}) \\quad (\\text{Texture Injection}) \\end{aligned}$$  
其中 $\\mathcal{M}(\\cdot)$ 是一个通过交叉扫描（Cross-Scan）将秘密序列注入到纹理状态空间的调制函数。$\\oplus$ 表示特征融合操作。

#### **3.3.2 损失函数 I: 结构-纹理对齐损失 (rSMI Alignment Loss)**

为了保证保真性，我们提出基于 Mamba 状态统计量的 **rSMI 损失**。我们不直接计算像素误差，而是最大化 $h\_{struc}$ 和 $h\_{tex}$ 之间的相对平方互信息，确保注入信息后的纹理统计分布不发生偏移：

$$\\mathcal{L}\_{align} \= \- \\log \\frac{\\exp(\\text{sim}(h\_{struc}, h\_{tex}) / \\tau)}{\\sum\_{j \\in \\text{batch}} \\exp(\\text{sim}(h\_{struc}, h\_{tex}^{(j)}) / \\tau)} \+ \\lambda\_{reg} \\| \\text{Cov}(h\_{tex}) \- \\text{Cov}(h\_{struc}) \\|\_F^2$$  
第一项为 InfoNCE 形式的互信息下界估计，第二项约束协方差矩阵的一致性。

### **3.4 训练与推理：干扰流形引导 (Interference Manifold Guidance)**

#### **3.4.1 干扰算子集**

定义可微干扰算子流形 $\\mathcal{M}\_{\\Pi} \= \\{ \\Pi(\\cdot; \\theta\_{\\pi}) \\mid \\Pi \\in \\{\\text{JPEG}, \\text{Crop}, \\text{Blur}, \\text{Noise}\\} \\}$。

#### **3.4.2 损失函数 II: 鲁棒解码损失 (Robust Decoding Loss)**

在训练生成器时，我们模拟攻击并计算解码梯度。对于 RQ-VAE 的每一层 $d$，我们施加分级保护：

$$\\mathcal{L}\_{robust} \= \\mathbb{E}\_{\\Pi \\sim \\mathcal{M}\_{\\Pi}} \\sum\_{d=1}^{D} w\_d \\cdot \\text{CE}(D\_\\phi(\\Pi(\\hat{x}\_0(\\theta)))\_d, S\_d)$$  
其中 $\\hat{x}\_0(\\theta) \= x\_t \- (1-t)v\_\\theta(x\_t)$ 是基于当前流场对 $x\_0$ 的单步估计，$w\_d$ 是随深度递减的权重（优先保护低层语义 Token）。

#### **3.4.3 总生成损失**

$$\\mathcal{L}\_{total} \= \\underbrace{\\| v\_\\theta(x\_t) \- (x\_1 \- x\_0) \\|^2}\_{\\mathcal{L}\_{flow}} \+ \\lambda\_1 \\mathcal{L}\_{align} \+ \\lambda\_2 \\mathcal{L}\_{robust}$$

### **3.5 解码器：语义辅助对比检索**

解码器 $D\_\\phi$ 也是一个 Mamba 网络。为了克服 RQ-VAE 累积误差，我们使用 **hDCE (Hard Decoupled Contrastive Entropy)** 损失训练它。

$$\\mathcal{L}\_{hDCE} \= \- \\log \\frac{\\exp(q \\cdot k^+ / \\tau)}{\\exp(q \\cdot k^+ / \\tau) \+ \\sum\_{n \\in \\mathcal{N}\_{hard}} \\exp(q \\cdot k\_n^- / \\tau)}$$  
其中 $k\_n^-$ 是在 RQ-VAE 码本空间中与正确 Token $k^+$ 欧氏距离最近的“硬负样本”。这使得解码器具有极强的纠错检索能力。

## ---

**4\. 实验方案 (Experiments)**

### **4.1 数据集与设置**

* **训练集**：DIV2K (高质量纹理学习), ImageNet-1K (语义多样性), COCO (文本-图像对齐)。  
* **秘密图像**：从 Paris StreetView 或 CelebA-HQ 中随机采样，测试不同分辨率 ($256^2, 512^2$)。  
* **干扰设置**：测试标准干扰（Gaussian Noise $\\sigma \\in \[0, 0.2\]$）、几何干扰（Crop ratio $\\in \[0.5, 1.0\]$）、压缩干扰（JPEG Q $\\in \[30, 90\]$）。

### **4.2 评价指标**

1. **保真性 (Fidelity)**:  
   * **FID (Fréchet Inception Distance)**: 评估生成图像的真实感。  
   * **CLIP Score**: 评估生成图像与 Prompt 的语义一致性。  
   * **Stego-LPIPS**: 隐写图与纯生成图（无隐写）的感知差异。  
2. **隐写性能 (Steganography)**:  
   * **Bit Accuracy (BA)**: 离散 Token 的恢复准确率。  
   * **Recovery PSNR/SSIM**: 重建秘密图像与原秘密图像的像素级相似度。  
3. **安全性 (Security)**:  
   * **Steganalysis Detection Accuracy**: 使用 SRNet 或 Zhu-Net 等隐写分析网络检测隐写图的准确率（越接近 50% 越好）。

### **4.3 对比基线 (Baselines)**

* **传统生成式隐写**: RivaGAN, CRoSS (CVPR'23).  
* **扩散模型隐写**: Weafusion, Diff-Stega.  
* **基于 Mamba 的基线**: 由于目前缺乏纯 Mamba 隐写，我们将构建一个 "Mamba-Base" (无三流解耦、无干扰引导) 作为消融对比。

### **4.4 预期结果分析 (Analysis Plan)**

* **主表结果**：展示 GenMamba-Flow 在 FID 和 Recovery PSNR 上双重领先，证明三流解耦解决了“保真度-容量”权衡，RQ-VAE 解决了“还原质量”瓶颈。  
* **鲁棒性曲线**：绘制在不同强度 JPEG 压缩下 Bit Accuracy 的变化曲线。预期本方法因引入干扰流形引导，曲线下降斜率显著低于 Diff-Stega。  
* **消融实验**：  
  * **w/o Tri-Stream**: 验证解耦对语义一致性（CLIP Score）的贡献。  
  * **w/o Interference Guidance**: 验证内生鲁棒性对 Bit Accuracy 的贡献。  
  * **VQ-VAE vs. RQ-VAE**: 验证残差量化对高频细节还原（SSIM）的贡献。  
* **可视化**：展示残差层级恢复效果（即 Depth 1 恢复轮廓，Depth 4 恢复细节）。

---

*(图注：GenMamba-Flow 架构图，左侧展示 RQ-VAE 分层离散化，中间展示三流 Mamba 生成器与干扰算子引导回路，右侧展示对比检索解码)*

通过上述严谨的方法论设计与实验验证，GenMamba-Flow 有望在鲁棒生成式隐写领域树立新的 SOTA 标准。