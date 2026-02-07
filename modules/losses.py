# modules/losses.py
import torch
import torch.nn.functional as F

class GenMambaLosses:
    @staticmethod
    def flow_matching_loss(v_pred, x_0, x_1):
        """
        Rectified Flow Loss: 
        目标速度 v = x_1 - x_0 (直线轨迹)
        """
        target_v = x_1 - x_0
        return F.mse_loss(v_pred, target_v)

    @staticmethod
    def rSMI_loss(feat_struc, feat_tex):
        """
        相对平方互信息 (rSMI) 的 InfoNCE 近似。
        用于对齐结构流和纹理流的统计分布，保证高保真。
        """
        # feat: [B, C, H, W] -> Flatten to [B, D]
        # 使用全局平均池化作为统计量
        f_s = F.normalize(feat_struc.mean(dim=[2, 3]), dim=1)
        f_t = F.normalize(feat_tex.mean(dim=[2, 3]), dim=1)
        
        # InfoNCE: 正样本对(同图结构与纹理)相似度高，负样本对低
        # Temperature = 0.07
        logits = torch.matmul(f_s, f_t.T) / 0.07
        labels = torch.arange(f_s.size(0), device=f_s.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def robust_decode_loss(pred_indices, target_indices):
        """
        解码分类损失 (Cross Entropy)
        pred_indices: [B, Num_Quantizers, Seq_Len, Codebook_Size] (Logits)
        target_indices: [B, Num_Quantizers, Seq_Len]
        """
        # Flatten for CE
        loss = 0
        num_q = pred_indices.shape[1]
        for i in range(num_q):
             # 简单的加权：第一层权重最大，后续递减
            weight = 1.0 / (i + 1)
            loss += weight * F.cross_entropy(
                pred_indices[:, i].permute(0, 2, 1), # [B, Classes, Seq]
                target_indices[:, i]
            )
        return loss