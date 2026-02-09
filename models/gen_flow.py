# models/gen_flow.py
import torch
import torch.nn as nn
from .mamba_block import VisionMambaBlock

# print(">>> IMPORTING GEN_FLOW FROM:", __file__)


class TriStreamMambaUNet(nn.Module):
    def __init__(self, in_channels=3, dim=128, secret_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        self.text_proj = nn.Linear(768, dim)

        self.secret_proj = nn.Linear(secret_dim, dim * 4)

        self.inc = nn.Conv2d(in_channels, dim, 3, padding=1)

        self.down1 = nn.Sequential(
            VisionMambaBlock(dim),
            nn.Conv2d(dim, dim * 2, 4, 2, 1)
        )

        self.down2 = nn.Sequential(
            VisionMambaBlock(dim * 2),
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)
        )

        self.bot = VisionMambaBlock(dim * 4)

        self.up1_conv = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.up1_block = VisionMambaBlock(dim * 2)

        self.up2_conv = nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1)
        self.up2_block = VisionMambaBlock(dim)

        self.outc = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x_t, t, text_emb, secret_emb):

        t_emb = self.time_mlp(t.view(-1, 1))
        txt_emb = self.text_proj(text_emb.mean(dim=1))
        cond = (t_emb + txt_emb).unsqueeze(-1).unsqueeze(-1)

        x1 = self.inc(x_t) + cond
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # print("[DEBUG][Gen] x1:", x1.shape)
        # print("[DEBUG][Gen] x2:", x2.shape)
        # print("[DEBUG][Gen] x3:", x3.shape)

        # assert x3.shape[2] == 64, \
        #     f"[FATAL] Bottleneck must be 64x64 but got {x3.shape}"

        h_struc = x3

        s_feat = torch.nn.functional.interpolate(
            secret_emb,
            size=x3.shape[2:]
        )

        s_feat = self.secret_proj(
            s_feat.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        h_tex = h_struc + s_feat

        x = self.bot(h_tex)

        x = self.up1_conv(x)
        x = x + x2
        x = self.up1_block(x)

        x = self.up2_conv(x)
        x = x + x1
        x = self.up2_block(x)

        v_pred = self.outc(x)

        return v_pred, h_struc, h_tex

class MambaDecoderHead(nn.Module):
    def __init__(self, in_channels=3, dim=128,
                 num_quantizers=4, codebook_size=1024):
        super().__init__()

        self.num_q = num_quantizers
        self.cb_size = codebook_size

        self.net = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, dim, 4, 2, 1),
            nn.ReLU(),
            VisionMambaBlock(dim),

            # 128 -> 64  ✅ stop here
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.ReLU(),
            VisionMambaBlock(dim * 2),
        )

        self.head = nn.Conv2d(
            dim * 2,
            num_quantizers * codebook_size,
            1
        )

    def forward(self, x):
        feat = self.net(x)
        # print("[DEBUG][Decoder] feat:", feat.shape)

        # ===== 自检 =====
        # assert feat.shape[2:] == (64, 64), (
        #     f"[FATAL] Decoder must output 64x64, got {feat.shape}"
        # )

        logits = self.head(feat)
        B, _, H, W = logits.shape

        logits = logits.view(
            B, self.num_q, self.cb_size, H * W
        )

        # print("[DEBUG][Decoder] logits:", logits.shape)

        # assert logits.shape[-1] == 4096
        return logits
