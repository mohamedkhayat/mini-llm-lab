import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift parameters."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.shift = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift
