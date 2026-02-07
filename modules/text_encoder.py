# modules/text_encoder.py
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoderWrapper(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.text_encoder.requires_grad_(False) # 冻结

    def forward(self, prompts, device):
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            embeddings = self.text_encoder(**inputs).last_hidden_state
        return embeddings # [B, Seq, Dim]