"""F4: lm_head bias config key + checkpoint inference.

Three checkpoint generations must all load with strict=True:
- before the bias removal: the state dict carries lm_head.bias;
- between the removal and this fix: no bias tensor AND no lm_head_bias
  key in the saved model_cfg — the key's absence cannot decide, the
  checkpoint's tensors are the source of truth;
- after this fix: the saved model_cfg carries the key explicitly.
"""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from models.gpt import GptModel
from training.assembly import _Runtime, build_model
from training.checkpointing import restore_model


def _tiny_model_cfg(**overrides):
    values = {
        "vocab_size": 64,
        "context_length": 16,
        "emb_dim": 16,
        "n_heads": 2,
        "n_kv_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "attention": "mha",
        "hidden_dim": 32,
        "activation": "gelu",
        "gated": False,
        "ffn_bias": False,
        "equalize_params": False,
        "tie_embeddings": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runtime(tmp_path):
    return _Runtime(
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        run_name="f4",
        use_bf16=False,
        use_compile=False,
        compile_mode="default",
    )


def test_lm_head_bias_key_builds_bias_and_absence_defaults_to_none():
    assert GptModel(_tiny_model_cfg()).lm_head.bias is None
    assert GptModel(_tiny_model_cfg(lm_head_bias=False)).lm_head.bias is None
    assert GptModel(_tiny_model_cfg(lm_head_bias=True)).lm_head.bias is not None


def test_checkpoint_with_bias_loads_and_rebuilds_with_bias(tiny_cfg, tmp_path):
    """A pre-change checkpoint (bias tensor, no lm_head_bias key in
    model_cfg) must load with strict=True, and the rebuilt model must
    have the bias."""
    biased = GptModel(_tiny_model_cfg(lm_head_bias=True))
    checkpoint = {
        "model": biased.state_dict(),
        # The saved cfg predates the key entirely.
        "model_cfg": {
            "vocab_size": 64,
            "context_length": 16,
            "emb_dim": 16,
            "n_heads": 2,
            "n_kv_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "attention": "mha",
            "hidden_dim": 32,
            "activation": "gelu",
            "gated": False,
            "ffn_bias": False,
            "equalize_params": False,
            "tie_embeddings": False,
        },
    }
    cfg = OmegaConf.create(
        {"model": OmegaConf.to_container(tiny_cfg.model, resolve=True)}
    )

    model, model_cfg, _ = build_model(cfg, checkpoint, _runtime(tmp_path))

    assert model.lm_head.bias is not None
    assert bool(model_cfg.lm_head_bias) is True


def test_checkpoint_with_orig_mod_prefixed_bias_loads(tiny_cfg, tmp_path):
    """Compiled-wrapper checkpoints carry _orig_mod.-prefixed keys; the
    inference must strip the prefix before deciding."""
    biased = GptModel(_tiny_model_cfg(lm_head_bias=True))
    state = {f"_orig_mod.{key}": value for key, value in biased.state_dict().items()}
    checkpoint = {
        "model": state,
        "model_cfg": {
            "vocab_size": 64,
            "context_length": 16,
            "emb_dim": 16,
            "n_heads": 2,
            "n_kv_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "attention": "mha",
            "hidden_dim": 32,
            "activation": "gelu",
            "gated": False,
            "ffn_bias": False,
            "equalize_params": False,
            "tie_embeddings": False,
        },
    }
    cfg = OmegaConf.create(
        {"model": OmegaConf.to_container(tiny_cfg.model, resolve=True)}
    )

    model, _, _ = build_model(cfg, checkpoint, _runtime(tmp_path))

    assert model.lm_head.bias is not None


def test_bias_free_checkpoint_without_cfg_key_loads(tiny_cfg, tmp_path):
    """A checkpoint written after the bias removal but before the config
    key exists: no bias tensor, no key in model_cfg — must load as
    bias-free."""
    bias_free = GptModel(_tiny_model_cfg())
    assert "lm_head.bias" not in bias_free.state_dict()
    checkpoint = {
        "model": bias_free.state_dict(),
        "model_cfg": {
            "vocab_size": 64,
            "context_length": 16,
            "emb_dim": 16,
            "n_heads": 2,
            "n_kv_heads": 2,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "attention": "mha",
            "hidden_dim": 32,
            "activation": "gelu",
            "gated": False,
            "ffn_bias": False,
            "equalize_params": False,
            "tie_embeddings": False,
        },
    }
    cfg = OmegaConf.create(
        {"model": OmegaConf.to_container(tiny_cfg.model, resolve=True)}
    )

    model, model_cfg, _ = build_model(cfg, checkpoint, _runtime(tmp_path))

    assert model.lm_head.bias is None
    assert bool(model_cfg.lm_head_bias) is False


def test_fresh_run_checkpoint_roundtrips_strict():
    """A fresh run (no bias, cfg carries the key) loads its own
    checkpoint with strict=True."""
    model = GptModel(_tiny_model_cfg(lm_head_bias=False))
    assert model.lm_head.bias is None

    fresh = GptModel(_tiny_model_cfg(lm_head_bias=False))
    restore_model(fresh, {"model": model.state_dict()})
