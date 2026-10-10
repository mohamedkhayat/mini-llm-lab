"""T1.4: GptModel.generate top-k masking and batch EOS (P13).

``forward`` is monkeypatched to return controlled logits so the masking,
sampling, and EOS-stop logic in ``generate`` is exercised deterministically.
"""

import torch

from models.gpt import GptModel

EOS = 7
# _tiny_model uses vocab_size=32.
MODEL_VOCAB = 32


def _tiny_model():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(
        {
            "name": "gpt2",
            "attention": "mha",
            "tie_embeddings": False,
            "vocab_size": 32,
            "context_length": 16,
            "emb_dim": 16,
            "hidden_dim": 32,
            "n_heads": 2,
            "n_kv_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": True,
            "ffn_bias": True,
            "normalization": "layernorm",
            "activation": "gelu",
            "gated": False,
            "equalize_params": False,
            "position_embedding": "absolute",
            "residual_style": "serial",
            "logit_softcap": None,
            "rope_theta": None,
        }
    )
    return GptModel(cfg)


def test_generate_batch_eos(monkeypatch):
    """Batch of 2 where one row hits EOS early: no crash (the old
    tensor-to-bool on a [B,1] tensor raised), the early-done row stays at
    EOS, and generation stops as soon as BOTH rows are done."""
    model = _tiny_model()
    state = {"step": 0}

    def fake_forward(self, x, return_logits=True):
        s = state["step"]
        state["step"] += 1
        batch, seq = x.shape
        vocab = 32
        logits = torch.full((batch, seq, vocab), -10.0)
        # Row 0: emits token 5 for its first two generated tokens, then EOS.
        logits[0, -1, 5 if s < 2 else EOS] = 10.0
        # Row 1: EOS from the very first generated token.
        logits[1, -1, EOS] = 10.0
        return logits

    monkeypatch.setattr(GptModel, "forward", fake_forward)
    prompt = torch.tensor([[1, 2], [3, 4]])

    out = model.generate(
        prompt,
        context_length=16,
        max_new_tokens=5,
        temperature=0.0,
        top_k=None,
        eos_id=EOS,
    )

    # Both rows done after 3 generated tokens -> early stop (2 + 3, not 2+5).
    assert out.shape == (2, 5)
    assert out[0].tolist() == [1, 2, 5, 5, EOS]
    assert out[1].tolist() == [3, 4, EOS, EOS, EOS]


def test_generate_batch_eos_none_keeps_generating(monkeypatch):
    """eos_id=None: no EOS handling, full max_new_tokens for every row."""
    model = _tiny_model()
    state = {"step": 0}

    def fake_forward(self, x, return_logits=True):
        state["step"] += 1
        batch, seq = x.shape
        vocab = 32
        logits = torch.full((batch, seq, vocab), -10.0)
        logits[0, -1, 5] = 10.0
        logits[1, -1, EOS] = 10.0  # EOS-looking token, but eos_id is None
        return logits

    monkeypatch.setattr(GptModel, "forward", fake_forward)
    prompt = torch.tensor([[1, 2], [3, 4]])

    out = model.generate(
        prompt,
        context_length=16,
        max_new_tokens=3,
        temperature=0.0,
        eos_id=None,
    )

    assert out.shape == (2, 5)
    assert out[1].tolist() == [3, 4, EOS, EOS, EOS]


def test_generate_top_k_support_and_oversized_k(monkeypatch):
    """Tokens outside the top-k are never sampled (seeded); top_k larger
    than the vocab is clamped instead of crashing topk."""
    model = _tiny_model()

    def fake_forward(self, x, return_logits=True):
        batch, seq = x.shape
        vocab = 32
        logits = torch.full((batch, seq, vocab), -100.0)
        logits[:, -1, 1] = 100.0  # in top-3
        logits[:, -1, 2] = 50.0  # in top-3
        logits[:, -1, 3] = 0.0  # in top-3
        # tokens 0, 4..31 are strictly outside the top-3 support
        return logits

    monkeypatch.setattr(GptModel, "forward", fake_forward)
    prompt = torch.tensor([[1, 2], [3, 4]])
    torch.manual_seed(1234)

    out = model.generate(
        prompt,
        context_length=16,
        max_new_tokens=20,
        temperature=1.0,  # multinomial path
        top_k=MODEL_VOCAB + 10,  # > vocab: must clamp, not crash
    )

    support = {1, 2, 3}
    generated = out[:, 2:].flatten().tolist()
    assert len(generated) == 2 * 20
    assert set(generated) <= support, f"sampled outside top-k: {set(generated) - support}"
    assert 1 in generated  # the dominant logit must have been sampled
