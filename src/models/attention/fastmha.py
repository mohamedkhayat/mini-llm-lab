import torch.nn as nn
import torch.nn.functional as F


class FastMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        num_kv_groups,
        qkv_bias=False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.d_in = d_in
        self.context_length = context_length

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        if num_tokens > self.context_length:
            raise ValueError(
                f"sequence length {num_tokens} exceeds attention context length "
                f"{self.context_length}"
            )

        qkv = self.qkv(x)

        # shape is B x num_tokens x d_out
        # now we need to reshape it so its
        # B x num_heads x head_dim x tokens

        # first lets split d_out into num_heads x head_dim

        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # now lets transpose dim 1 and 2

        qkv = qkv.permute(2, 0, 3, 1, 4)

        queries, keys, values = qkv.unbind(0)

        dropout_p = self.dropout.p if self.training else 0.0
        context_vec = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )
        out_vec = self.out_proj(context_vec)

        return out_vec
