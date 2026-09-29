"""Tests for the sequential memmap dataloader (train.bin / eval.bin / meta.json)."""

import json

import numpy as np
import pytest
import torch

from data.dataloader import MemmapDataLoader


def make_data_dir(
    tmp_path,
    train_tokens=None,
    eval_tokens=None,
    dtype: np.dtype = np.dtype(np.uint16),
):
    """Write a minimal prepared dataset: train.bin, eval.bin, meta.json."""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    if train_tokens is None:
        train_tokens = np.arange(96, dtype=dtype)
    if eval_tokens is None:
        eval_tokens = np.arange(1000, 1000 + len(train_tokens), dtype=dtype)
    train_tokens = np.ascontiguousarray(train_tokens, dtype=dtype)
    eval_tokens = np.ascontiguousarray(eval_tokens, dtype=dtype)
    train_tokens.tofile(data_dir / "train.bin")
    eval_tokens.tofile(data_dir / "eval.bin")
    meta = {
        "tokenizer_name": "gpt2",
        "dtype": "uint16" if dtype == np.dtype(np.uint16) else "uint32",
        "max_tokens": int(len(train_tokens) + len(eval_tokens)),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(eval_tokens)),
        "val_ratio": 0.5,
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return data_dir


def make_loader(data_dir, split="train", batch_size=2, block_size=3, device="cpu"):
    return MemmapDataLoader(
        data_dir, split, batch_size, block_size, device
    )


def test_first_batch_is_a_contiguous_window_with_shifted_targets(tmp_path):
    # batch_tokens = 2 x 3 = 6; batch 0 covers file positions 0..6.
    data_dir = make_data_dir(tmp_path)
    loader = make_loader(data_dir)
    x, y = next(iter(loader))
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert x.dtype == torch.long
    assert x[0].tolist() == [0, 1, 2]
    assert x[1].tolist() == [3, 4, 5]
    assert y[0].tolist() == [1, 2, 3]
    assert y[1].tolist() == [4, 5, 6]


def test_uint32_meta_reads_a_uint32_file(tmp_path):
    tokens = np.arange(100000, 100096, dtype=np.uint32)
    data_dir = make_data_dir(tmp_path, train_tokens=tokens, dtype=np.dtype(np.uint32))
    loader = make_loader(data_dir)
    x, y = next(iter(loader))
    assert x[0].tolist() == [100000, 100001, 100002]
    assert y[0].tolist() == [100001, 100002, 100003]


def test_eval_split_reads_eval_bin_from_data_dir(tmp_path):
    data_dir = make_data_dir(tmp_path)
    loader = make_loader(data_dir, split="eval")
    x, _ = next(iter(loader))
    assert x[0].tolist() == [1000, 1001, 1002]


def test_state_dict_round_trip_resumes_from_saved_position(tmp_path):
    data_dir = make_data_dir(tmp_path)
    first = make_loader(data_dir)
    it = iter(first)
    next(it)  # batch 1
    next(it)  # batch 2
    next(it)  # batch 3
    state = first.state_dict()  # current_step == 3
    batch4 = next(it)  # what the next consumer sees

    resumed = make_loader(data_dir)
    resumed.load_state_dict(state)
    x, _ = next(iter(resumed))
    assert x.tolist() == batch4[0].tolist()
    assert x is not batch4[0]  # a fresh tensor, not the same object


def test_val_iterator_starts_at_batch_zero_on_every_iter(tmp_path):
    data_dir = make_data_dir(tmp_path)
    loader = make_loader(data_dir, split="eval")
    first = next(iter(loader))
    second = next(iter(loader))
    assert first[0][0].tolist() == second[0][0].tolist() == [1000, 1001, 1002]


def test_train_pass_stops_at_exhaustion(tmp_path):
    # 17 tokens -> (17 - 1) // 6 = 2 batches; the train split is a single
    # pass: iteration ends at the file's end (the trainer rejects budgets
    # that would rescan).
    data_dir = make_data_dir(tmp_path, train_tokens=np.arange(17, dtype=np.uint16))
    loader = make_loader(data_dir)
    assert len(loader) == 2
    batches = [b[0].tolist() for b in iter(loader)]
    assert batches == [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
    # A fresh iterator resumes from the loader's cursor, not the top of
    # the file.
    assert list(iter(loader)) == []


def test_total_batches_drops_the_incomplete_tail(tmp_path):
    # 15 tokens -> 2 full 6-token batches; the trailing 3 tokens are unusable.
    data_dir = make_data_dir(tmp_path, train_tokens=np.arange(15, dtype=np.uint16))
    loader = make_loader(data_dir)
    assert len(loader) == 2
    assert len(list(iter(loader))) == 2


def test_loader_accepts_a_torch_device(tmp_path):
    data_dir = make_data_dir(tmp_path)
    loader = MemmapDataLoader(data_dir, "train", 2, 3, torch.device("cpu"))
    x, _ = next(iter(loader))
    assert x.device.type == "cpu"


def test_missing_bin_raises_a_clear_error(tmp_path):
    data_dir = tmp_path / "empty"
    data_dir.mkdir()
    (data_dir / "meta.json").write_text(json.dumps({"dtype": "uint16"}))
    with pytest.raises(FileNotFoundError, match="train.bin"):
        make_loader(data_dir)
