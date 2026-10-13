from contextlib import nullcontext

import torch
import torch.nn as nn

from models.blocks.transformerBlock import TransformerBlock
from models.normalization.normalization import get_norm
import math
# GPT-2 style model


class GptModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)

        self.context_length = cfg.context_length

        if not cfg.position_embedding == "rope":
            self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        else:
            self.pos_emb = None
        self.dropout = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        )

        # Older configs/checkpoints without a normalization key keep the
        # original layer-norm behavior so they stay resumable.
        self.final_norm = get_norm(getattr(cfg, "normalization", "layer_norm"))(
            cfg.emb_dim
        )
        # F4: the key's absence defaults to no bias (current behavior);
        # build_model infers the key from the checkpoint's tensors when a
        # saved model_cfg predates it.
        self.lm_head = nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=bool(getattr(cfg, "lm_head_bias", False))
        )

        self.tie_embeddings = bool(getattr(cfg, "tie_embeddings", False))
        if self.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers)
                )

    def forward(self, x, return_logits=True):
        # x : B X SEQ_LEN
        _, seq_len = x.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds model context length "
                f"{self.context_length}"
            )

        tok_embs = self.tok_emb(x)
        if self.pos_emb is not None:
            pos_embs = self.pos_emb(torch.arange(seq_len, device=x.device))
            x = tok_embs + pos_embs
        else:
            x = tok_embs
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        if return_logits:
            return self.lm_head(x)

        return x

    def generate(
        self,
        token_idx,
        context_length,
        max_new_tokens=10,
        temperature=1.0,
        top_k=None,
        eos_id=None,
        autocast_context=None,
    ):
        if autocast_context is None:
            autocast_context = nullcontext()
        vocab_size = self.lm_head.out_features
        # [B] mask of rows that have hit EOS; they stay pinned at eos_id.
        done = torch.zeros(
            token_idx.shape[0], dtype=torch.bool, device=token_idx.device
        )
        for _ in range(max_new_tokens):
            idx_cond = token_idx[:, -context_length:]
            with torch.no_grad(), autocast_context:
                logits = self.forward(idx_cond)

            logits = logits[:, -1, :]  # take last position
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, vocab_size))[0][:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if temperature > 0.0:
                logits /= temperature
                probas = torch.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probas, num_samples=1)
            else:
                next_token_idx = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None:
                next_token_idx = torch.where(done.unsqueeze(-1), eos_id, next_token_idx)
                done = done | (next_token_idx.squeeze(-1) == eos_id)
                if done.all():
                    token_idx = torch.cat((token_idx, next_token_idx), dim=-1)
                    break

            token_idx = torch.cat((token_idx, next_token_idx), dim=-1)
        return token_idx

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
