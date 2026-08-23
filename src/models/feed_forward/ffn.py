import torch.nn as nn
from models.activations.gelu import GeLU
from models.feed_forward.utils import equalize_params


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        activation_impls = {"gelu" : GeLU(), "silu": nn.SiLU(), "sigmoid": nn.Sigmoid()}
        hidden_dim = equalize_params(cfg.hidden_dim, cfg.gated, do_equalize= cfg.equalize_params)
        self.fc1 = nn.Linear(cfg.emb_dim, hidden_dim, bias=cfg.ffn_bias)
        self.activation = activation_impls[cfg.activation]
        self.out = nn.Linear(hidden_dim, cfg.emb_dim, bias=cfg.ffn_bias)
        self.gated = cfg.gated
        if self.gated:
            self.fc2 = nn.Linear(cfg.emb_dim, hidden_dim, bias=cfg.ffn_bias)

    def forward(self, x):
        shortcut = x
        x = self.activation(self.fc1(x))
        if self.gated:
            x_proj = self.fc2(shortcut)
            x = x_proj * x
        x = self.out(x)
        return x
