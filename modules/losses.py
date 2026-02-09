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
    def robust_decode_loss(pred_logits, s_indices):
        """
        pred_logits: [B, Q, K, N]
        s_indices  : [B, Q, N]
        """
        # -----------------------------
        # 1. 自检（非常关键）
        # # -----------------------------
        # assert pred_logits.dim() == 4, \
        #     f"pred_logits must be 4D [B,Q,K,N], got {pred_logits.shape}"
        # assert s_indices.dim() == 3, \
        #     f"s_indices must be 3D [B,Q,N], got {s_indices.shape}"

        B, Q, K, N = pred_logits.shape
        Bs, Qs, Ns = s_indices.shape

        # assert B == Bs, f"B mismatch: {B} vs {Bs}"
        # assert Q == Qs, f"Q mismatch: {Q} vs {Qs}"
        # assert N == Ns, f"N mismatch: {N} vs {Ns}"

        # # 索引合法性检查（防 silent bug）
        # if not torch.all((s_indices >= 0) & (s_indices < K)):
        #     raise ValueError(
        #         f"s_indices out of range [0, {K-1}]"
        #     )

        # -----------------------------
        # 2. 正确计算 CE loss
        # -----------------------------
        total_loss = 0.0

        for q in range(Q):
            # [B, K, N] → [B, N, K] → [B*N, K]
            logits_q = (
                pred_logits[:, q]
                .permute(0, 2, 1)
                .contiguous()
                .view(B * N, K)
            )

            # [B, N] → [B*N]
            targets_q = (
                s_indices[:, q]
                .contiguous()
                .view(B * N)
            )

            loss_q = F.cross_entropy(
                logits_q,
                targets_q,
                reduction="mean"
            )

            total_loss += loss_q

        return total_loss / Q