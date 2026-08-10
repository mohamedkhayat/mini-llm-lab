import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias = False
        
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.d_in = d_in

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        self.out_proj = nn.Linear(d_out, d_out)
        
    def forward(self, x):
        # we first do the big mat muls once
        # then we split the output into multiple heads
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # shape is B x num_tokens x d_out
        # now we need to reshape it so its 
        # B x num_heads x head_dim x tokens
        
        # first lets split d_out into num_heads x head_dim
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # now lets transpose dim 1 and 2
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # shape is now b x num_heads x num_tokens x head dim
        
        # attn scores = queries @ keys and shape should be B x num_heads x num_tokens x num_tokens
        # (B x num_heads x num_tokens x head dim) x (B x num_heads x head_dim, num_tokens)
        # so we need to reshape keys
        
        attn_scores = queries @ keys.transpose(2, 3)

        # now we apply mask
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # make a 2d boolean mask from 0 to seq_len, because seq length is not always == context_size

        attn_scores.masked_fill_(mask_bool, -torch.inf) # replace Truethy values with -inf so softmax makes them = 0

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attn_weights shape is B x n_heads x num_tokens x num_tokens
        # values shape is B x num_heads x num_tokens x head_dim
        context_vec = (attn_weights @ values) # shape is B x n_heads x n_tokens, head_dim
        # we need shape to be B x num_tokens x n_heads x head_dim
        context_vec = context_vec.transpose(1, 2)
        # now need to reshape it back to B x num_tokens x d_out
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        out_vec = self.out_proj(context_vec)

        return out_vec