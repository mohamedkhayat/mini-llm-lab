import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupQueryAttention(nn.Module):
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
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.group_size = num_heads // num_kv_groups
        self.d_out = d_out
        self.d_in = d_in
        self.context_length = context_length
        self.dropout = dropout
        self.W_query = nn.Linear(
            d_in, d_out, bias=qkv_bias
        )  # d_out = n_heads x head_dim
        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=qkv_bias
        )  # here we no longer use n_heads, but num_kv_groups
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        # we first do the big mat muls once
        # then we split the output into multiple heads
        batch_size, num_tokens, _ = x.shape
        if num_tokens > self.context_length:
            raise ValueError(
                f"sequence length {num_tokens} exceeds attention context length "
                f"{self.context_length}"
            )

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # shape fo Q : is B x num_tokens x d_out
        # now we need to reshape it so its
        # B x n_kv_groups x head_dim x tokens
        # shape for K,V is : B x num_tokens x n_kv_groups * head_dim
        # now we need to reshape it so its
        # B x n_kv_groups x head_dim x tokens

        # first lets split d_out into num_heads x head_dim for Q
        # and num_kv_groups * self.head_dim in 2

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)

        # now lets transpose dim 1 and 2

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # shape is now b x num_heads x num_tokens x head dim for Q
        # and b x num_kv_groups x num_tokens x head dim for KV

        # I could do this but FA can do it for me
        # keys = keys.repeat_interleave(self.group_size, dim=1)
        # values = values.repeat_interleave(self.group_size, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        context_vec = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
            enable_gqa=True,
        )
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )
        out_vec = self.out_proj(context_vec)

        return out_vec
