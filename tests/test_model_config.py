from types import SimpleNamespace

import pytest

from models.gpt import GptModel
from models.normalization.layerNorm import LayerNorm
from models.normalization.normalization import get_norm
from models.normalization.rms_norm import RMSNorm


def make_model_config(**overrides):
    values = {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 4,
        "n_kv_heads": 4,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "attention": "mha",
        "ff_mult": 2,
        "hidden_dim": 32,
        "activation": "gelu",
        "gated": False,
        "ffn_bias": False,
        "equalize_params": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tied_embeddings_are_enabled_explicitly():
    model = GptModel(make_model_config(tie_embeddings=True))

    assert model.tie_embeddings is True
    assert model.fc.weight is model.tok_emb.weight


def test_missing_tied_embeddings_key_defaults_to_untied():
    model = GptModel(make_model_config())

    assert model.tie_embeddings is False
    assert model.fc.weight is not model.tok_emb.weight


@pytest.mark.parametrize("spelling", ["layer_norm", "layernorm"])
def test_layer_norm_spellings_resolve_to_layer_norm(spelling):
    assert get_norm(spelling) is LayerNorm
    model = GptModel(make_model_config(normalization=spelling))

    assert isinstance(model.final_norm, LayerNorm)


@pytest.mark.parametrize("spelling", ["rms_norm", "rmsnorm"])
def test_rms_norm_spellings_resolve_to_rms_norm(spelling):
    assert get_norm(spelling) is RMSNorm
    model = GptModel(make_model_config(normalization=spelling))

    assert isinstance(model.final_norm, RMSNorm)


def test_normalization_lookup_is_case_insensitive():
    assert get_norm("LAYER_NORM") is LayerNorm
    assert get_norm("RmsNorm") is RMSNorm


def test_unknown_normalization_lists_the_valid_choices():
    with pytest.raises(KeyError) as excinfo:
        get_norm("batch_norm")

    message = str(excinfo.value)
    assert "layernorm" in message
    assert "rmsnorm" in message


def test_missing_normalization_key_defaults_to_layer_norm():
    model = GptModel(make_model_config())

    assert isinstance(model.final_norm, LayerNorm)


def test_gpt2_preset_shaped_config_uses_layer_norm():
    model = GptModel(make_model_config(name="gpt2", normalization="layernorm"))

    assert isinstance(model.final_norm, LayerNorm)


@pytest.mark.parametrize("preset", ["qwen", "moe"])
def test_qwen_and_moe_preset_shaped_configs_use_rms_norm(preset):
    model = GptModel(make_model_config(name=preset, normalization="rmsnorm"))

    assert isinstance(model.final_norm, RMSNorm)
