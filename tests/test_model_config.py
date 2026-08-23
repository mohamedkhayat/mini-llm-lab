from types import SimpleNamespace

from models.gpt import GptModel


def make_model_config(**overrides):
    values = {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "attention": "mha",
        "ff_mult": 2,
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
