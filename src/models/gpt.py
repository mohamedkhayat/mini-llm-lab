from contextlib import nullcontext

import torch
import torch.nn as nn

from models.blocks.transformerBlock import TransformerBlock
from models.normalization.normalization import get_norm

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

        # Older configs/checkpoints without a normalization key keep the
        # original layer-norm behavior so they stay resumable.
        self.final_norm = get_norm(getattr(cfg, "normalization", "layer_norm"))(
            cfg.emb_dim
        )
        self.fc = nn.Linear(cfg.emb_dim, cfg.vocab_size)
        # Older configs/checkpoints do not have this key and are treated as
        # untied so they remain compatible with the original training setup.
        self.tie_embeddings = bool(getattr(cfg, "tie_embeddings", False))
        if self.tie_embeddings:
            self.fc.weight = self.tok_emb.weight
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

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

    def generate(
        self,
        token_idx,
        context_length,
        autocast_context=nullcontext(),
        max_new_tokens=10,
        temperature=1.0,
        top_k=None,
        eos_id=None,
    ):
        for _ in range(max_new_tokens):
            idx_cond = token_idx[:, -context_length:]
            with torch.no_grad(), autocast_context:
                logits = self.forward(idx_cond)

            logits = logits[:, -1, :]  # take last position
            if top_k is not None:
                top_logits, _ = torch.topk(logits, k=top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            if temperature > 0.0:
                logits /= temperature
                probas = torch.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probas, num_samples=1)
            else:
                next_token_idx = torch.argmax(logits, dim=-1, keepdim=True)

            if next_token_idx == eos_id:
                break

            token_idx = torch.cat((token_idx, next_token_idx), dim=-1)
        return token_idx
