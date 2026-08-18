import torch.nn as nn
from models.activations.gelu import GeLU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = cfg.emb_dim * int(getattr(cfg, "ff_mult", 4))
        self.fc1 = nn.Linear(cfg.emb_dim, hidden_dim)
        self.gelu = GeLU()
        self.fc2 = nn.Linear(hidden_dim, cfg.emb_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
