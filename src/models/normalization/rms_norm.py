import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        mean_squared = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(self.eps + mean_squared)
        return x_normed * self.scale
