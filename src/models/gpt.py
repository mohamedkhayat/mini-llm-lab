import torch
import torch.nn as nn
from models.blocks.transformerBlock import TransformerBlock
from models.normalization.layerNorm import LayerNorm

# GPT-2 style model


class GptModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.fc = nn.Linear(cfg.emb_dim, cfg.vocab_size)

    def forward(self, x):
        # x : B X SEQ_LEN
        _, seq_len = x.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds model context length "
                f"{self.pos_emb.num_embeddings}"
            )

        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embs + pos_embs
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.fc(x)
        return logits
