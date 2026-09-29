import json
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from data.budget import minimum_cache_tokens, split_cache_tokens
from data.prepare import _verify_prefix_preserved


_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_entrypoint", Path(__file__).parents[1] / "train.py"
)
train_entrypoint = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(train_entrypoint)
is_dataset_valid = train_entrypoint.is_dataset_valid
prepare_cache_for_run = train_entrypoint.prepare_cache_for_run


def make_cfg(*, max_tokens="1k", val_ratio=0.1, start_decay=False):
    return SimpleNamespace(
        data=SimpleNamespace(
            max_tokens=max_tokens,
            val_ratio=val_ratio,
            tokenizer_name="gpt2",
            seq_len=8,
            hf_dataset="stub",
            hf_config=None,
            text_column="text",
            file_format="parquet",
        ),
        training=SimpleNamespace(
            batch_size=2,
            accum_steps=1,
            resume_mode="exact",
            start_decay=start_decay,
            decay_tokens="1k" if start_decay else None,
        ),
    )


def write_cache(cache_dir, total_tokens=1_000, val_ratio=0.1, *, dtype="uint16"):
    train_tokens, val_tokens = split_cache_tokens(total_tokens, val_ratio)
    itemsize = 2 if dtype == "uint16" else 4
    cache_dir.mkdir()
    (cache_dir / "train.bin").write_bytes(b"\0" * (train_tokens * itemsize))
    (cache_dir / "eval.bin").write_bytes(b"\0" * (val_tokens * itemsize))
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "tokenizer_name": "gpt2",
                "hf_dataset": "stub",
                "hf_config": None,
                "text_column": "text",
                "file_format": "parquet",
                "dtype": dtype,
                "max_tokens": total_tokens,
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
                "val_ratio": val_ratio,
            }
        ),
        encoding="utf-8",
    )


def test_cache_metadata_and_file_sizes_match_the_requested_split(tmp_path):
    cache_dir = tmp_path / "cache"
    write_cache(cache_dir)

    assert is_dataset_valid(cache_dir, make_cfg()) is True

    metadata_path = cache_dir / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["val_ratio"] = 0.2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert is_dataset_valid(cache_dir, make_cfg()) is False


def test_cache_file_size_mismatch_forces_reprocessing(tmp_path):
    cache_dir = tmp_path / "cache"
    write_cache(cache_dir)
    with (cache_dir / "train.bin").open("ab") as file:
        file.write(b"\0\0")

    assert is_dataset_valid(cache_dir, make_cfg()) is False


def test_previous_memmap_metadata_without_identity_fields_remains_reusable(tmp_path):
    cache_dir = tmp_path / "cache"
    write_cache(cache_dir)
    metadata_path = cache_dir / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in ("hf_dataset", "hf_config", "text_column", "file_format"):
        metadata.pop(field)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert is_dataset_valid(cache_dir, make_cfg()) is True


def test_prefix_verification_rejects_changed_regenerated_tokens(tmp_path):
    old_path = tmp_path / "old.bin"
    new_path = tmp_path / "new.bin"
    np.arange(10, dtype=np.uint16).tofile(old_path)
    np.concatenate((np.arange(5), np.arange(100, 110))).astype(np.uint16).tofile(
        new_path
    )

    with pytest.raises(RuntimeError, match="do not match"):
        _verify_prefix_preserved(old_path, new_path, "uint16")


def test_fresh_train_budget_must_fit_the_configured_cache():
    cfg = make_cfg(max_tokens="1k")
    cfg.training.train_tokens = "2k"

    with pytest.raises(ValueError, match="train_tokens"):
        prepare_cache_for_run(cfg)


def test_resume_data_identity_change_is_rejected_before_preparation(tmp_path):
    cfg = make_cfg()
    checkpoint = {
        "cfg": {
            "data": {
                "hf_dataset": "different-dataset",
                "hf_config": None,
                "text_column": "text",
                "file_format": "parquet",
                "tokenizer_name": "gpt2",
            }
        }
    }

    with pytest.raises(ValueError, match="data identity changed"):
        prepare_cache_for_run(cfg, checkpoint)


def test_decay_endpoint_expands_total_cache_budget(monkeypatch, tmp_path):
    cfg = make_cfg(max_tokens="1k", start_decay=True)
    cache_dir = tmp_path / "cache"
    prepared_budgets = []

    monkeypatch.setattr(train_entrypoint, "get_data_dir", lambda _cfg: cache_dir)

    def fake_prepare_dataset(prepared_cfg, **_kwargs):
        prepared_budgets.append(prepared_cfg.data.max_tokens)

    monkeypatch.setattr(train_entrypoint, "prepare_dataset", fake_prepare_dataset)

    budget = prepare_cache_for_run(cfg, {"step": 10})

    required_train = budget.target_step * 16 + 1
    assert cfg.data.max_tokens >= minimum_cache_tokens(required_train, 0.1)
    assert prepared_budgets == [cfg.data.max_tokens]


def test_cache_budget_parser_is_used_for_phase_preflight(monkeypatch, tmp_path):
    cfg = make_cfg(max_tokens="1.2k", start_decay=True)
    cache_dir = tmp_path / "cache"
    prepared_budgets = []

    monkeypatch.setattr(train_entrypoint, "get_data_dir", lambda _cfg: cache_dir)
    monkeypatch.setattr(
        train_entrypoint,
        "prepare_dataset",
        lambda prepared_cfg, **_kwargs: prepared_budgets.append(
            prepared_cfg.data.max_tokens
        ),
    )

    prepare_cache_for_run(cfg, {"step": 10})

    assert prepared_budgets
    assert isinstance(cfg.data.max_tokens, int)
