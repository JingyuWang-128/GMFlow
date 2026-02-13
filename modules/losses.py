# modules/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GenMambaLosses:
    @staticmethod
    def flow_matching_loss(v_pred, x_0, x_1):
        """
        Rectified Flow Loss: 目标速度 v = x_1 - x_0
        """
        target_v = x_1 - x_0
        return F.mse_loss(v_pred, target_v)

    @staticmethod
    def rSMI_loss(feat_struc, feat_tex):
        """
        相对平方互信息 (rSMI) 的 InfoNCE 近似
        """
        # Global Average Pooling
        f_s = F.normalize(feat_struc.mean(dim=[2, 3]), dim=1)
        f_t = F.normalize(feat_tex.mean(dim=[2, 3]), dim=1)
        
        # Temperature = 0.07
        logits = torch.matmul(f_s, f_t.T) / 0.07
        labels = torch.arange(f_s.size(0), device=f_s.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def hDCE_loss(pred_features, target_indices, codebooks, temperature=0.1):
        """
        Hard Decoupled Contrastive Entropy (hDCE) Loss
        """
        B, Q, C, H, W = pred_features.shape
        total_loss = 0.0
        
        for q in range(Q):
            # 1. Query: [N, C] (N = B*H*W)
            query = pred_features[:, q].permute(0, 2, 3, 1).reshape(-1, C)
            query = F.normalize(query, dim=1)
            
            # 2. Key Selection (Handle various formats)
            if isinstance(codebooks, list):
                codebook = codebooks[q]
            elif isinstance(codebooks, torch.Tensor):
                if codebooks.ndim == 3 and codebooks.shape[0] == Q: 
                    codebook = codebooks[q] # [K, C] from [Q, K, C]
                else:
                    codebook = codebooks # Shared or Single Tensor
            else:
                # Fallback
                codebook = codebooks

            # ---------------------------------------------------------
            # [关键修复] 强力维度清洗，防止 RuntimeError: input.dim()=3
            # ---------------------------------------------------------
            if codebook.ndim == 3:
                # 常见情况: [1, K, C] -> Squeeze to [K, C]
                codebook = codebook.squeeze(0)
            
            # 如果经过 squeeze 还是不对（比如 [1, 1, K, C]），强制 reshape
            if codebook.ndim != 2:
                # 假设最后一维是 Embedding Dim (C)
                if codebook.shape[-1] == C:
                    codebook = codebook.view(-1, C)
                else:
                    # 如果最后一维不对，可能是转置了 [C, K] -> 转置回 [K, C]
                    if codebook.shape[0] == C:
                         codebook = codebook.view(C, -1).t()
            # ---------------------------------------------------------

            key = F.normalize(codebook, dim=1)
            
            # 3. Cosine Similarity [N, K]
            # 现在 key 保证是 2D [K, C]，可以使用 .t() 或 .permute(1, 0)
            # logits = query @ key.T
            logits = torch.matmul(query, key.transpose(0, 1)) / temperature
            
            # 4. Target Preparation
            target = target_indices[:, q].reshape(-1)
            
            # 5. Cross Entropy
            loss_q = F.cross_entropy(logits, target)
            total_loss += loss_q
            
        return total_loss / Q