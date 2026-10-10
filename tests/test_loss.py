"""T3.1: build_loss — one shared loss path, FLCE resolved once, Liger
imported lazily only in the FLCE branch."""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.loss import build_loss


class _StandIn(nn.Module):
    """GptModel stand-in: forward(x, return_logits) + an lm_head with bias,
    so the FLCE branch's attribute reads are exercised structurally."""

    def __init__(self, vocab=32, dim=8):
        super().__init__()
        torch.manual_seed(0)
        self.lm_head = nn.Linear(dim, vocab, bias=True)
        # Deterministic hidden states (a token->vector lookup), so repeated
        # forwards of the same input agree. Plain attribute (not a buffer):
        # the stand-in is never saved/loaded or moved.
        self._embeddings = torch.randn(vocab, dim)

    def forward(self, x, return_logits=True):
        h = self._embeddings[x]
        return self.lm_head(h) if return_logits else h


def test_flce_fallback_on_cpu(capsys):
    """use_flce=True on CPU warns once and falls back to cross-entropy;
    Liger is NOT imported (lazy import stays in the FLCE branch)."""
    assert "liger_kernel" not in sys.modules
    model = _StandIn()
    loss_fn = build_loss(model, model, use_flce=True, device=torch.device("cpu"))

    out = capsys.readouterr().out
    assert "FLCE" in out and "Falling back" in out
    assert "liger_kernel" not in sys.modules, "Liger must stay unimported on the fallback path"

    x = torch.randint(0, 32, (2, 6))
    y = torch.randint(0, 32, (2, 6))
    loss = loss_fn(x, y)
    assert loss.dim() == 0 and loss.requires_grad


def test_loss_callable_matches_manual_cross_entropy():
    """The CE callable is exactly cross_entropy over the model's logits
    (the golden literals depend on this path being unchanged)."""
    torch.manual_seed(0)
    model = _StandIn()
    loss_fn = build_loss(model, model, use_flce=False, device=torch.device("cpu"))

    x = torch.randint(0, 32, (2, 6))
    y = torch.randint(0, 32, (2, 6))
    expected = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    torch.testing.assert_close(loss_fn(x, y), expected)


def test_no_warning_when_flce_disabled(capsys):
    model = _StandIn()
    build_loss(model, model, use_flce=False, device=torch.device("cpu"))
    assert "FLCE" not in capsys.readouterr().out
