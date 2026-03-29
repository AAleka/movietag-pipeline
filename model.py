import torch
import torch.nn as nn

from transformers import AutoModel


class VisualTransformer(nn.Module):
    def __init__(self, dim, depth=2, heads=8, dropout=0.2):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 2048, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dropout=dropout
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_emb[:, :N+1]
        x = self.encoder(x)

        x = self.norm(x)
        return x


class MultiModalSceneClassifier(nn.Module):
    def __init__(self, clip_dim, num_tags):
        super().__init__()
        
        self.frame_encoder = VisualTransformer(clip_dim)
        self.text_encoder = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, clip_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=clip_dim,
            num_heads=8,
            batch_first=True
        )

        self.norm = nn.LayerNorm(clip_dim)

        self.head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tags),
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, frames, input_ids, attention_mask):
        v_tokens = self.frame_encoder(frames)

        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        t_tokens = text_out.last_hidden_state 
        t_tokens = self.text_proj(t_tokens)

        fused, _ = self.cross_attn(query=t_tokens, key=v_tokens, value=v_tokens)
        fused = fused.mean(dim=1)
        logits = self.head(fused)

        return logits / self.temperature