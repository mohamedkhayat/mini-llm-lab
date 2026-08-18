import torch.nn as nn
from models.normalization.layerNorm import LayerNorm
from models.attention.mha import MultiHeadAttention
from models.attention.fastmha import FastMultiHeadAttention
from models.feed_forward.ffn import FeedForward


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal attention and a GELU FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)

        attention_name = str(getattr(cfg, "attention", "mha")).lower()
        attention_impls = {
            "mha": MultiHeadAttention,
            "fastmha": FastMultiHeadAttention,
        }
        if attention_name not in attention_impls:
            choices = ", ".join(sorted(attention_impls))
            raise ValueError(
                f"Unknown attention implementation {attention_name!r}; "
                f"choose one of: {choices}"
            )

        self.mha = attention_impls[attention_name](
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.context_length,
            cfg.drop_rate,
            cfg.n_heads,
            cfg.qkv_bias,
        )
        self.ffn = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
