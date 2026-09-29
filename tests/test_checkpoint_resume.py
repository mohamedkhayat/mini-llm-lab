"""Tests for the checkpoint/resume seam with the memmap dataloader:
cursor capture/restore, and the consistency checks."""

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from data.dataloader import MemmapDataLoader
from training.checkpointing import (
    RunState,
    atomic_torch_save,
    build_checkpoint_payload,
    capture_data_state,
    check_resume_consistency,
    load_checkpoint,
    restore_data_state,
    restore_training_state,
)
from training.trainer import get_model_parameter_metrics

# 81 tokens, batch 2 x 8 = 16 -> (81 - 1) // 16 = 5 batches per pass.
BATCHES_PER_PASS = 5


def make_loader(tmp_path, split="train"):
    data_dir = tmp_path / "cache"
    if not (data_dir / "meta.json").is_file():
        data_dir.mkdir(exist_ok=True)
        train = (np.arange(81) % 31) + 1
        eval_tokens = ((np.arange(49) + 13) % 31) + 1
        train.astype(np.uint16).tofile(data_dir / "train.bin")
        eval_tokens.astype(np.uint16).tofile(data_dir / "eval.bin")
        with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_name": "gpt2",
                    "dtype": "uint16",
                    "max_tokens": 130,
                    "train_tokens": 81,
                    "val_tokens": 49,
                    "val_ratio": 0.375,
                },
                f,
            )
    return MemmapDataLoader(data_dir, split, 2, 8, torch.device("cpu"))


def test_capture_data_state_saves_the_memmap_cursor(tmp_path):
    loader = make_loader(tmp_path)
    it = iter(loader)
    next(it)
    next(it)

    state = capture_data_state(loader)

    assert state == {"kind": "memmap", "state": {"current_step": 2}}


def test_restore_data_state_resumes_from_the_saved_cursor(tmp_path):
    source = make_loader(tmp_path)
    it = iter(source)
    next(it)
    next(it)
    saved = capture_data_state(source)

    fresh = make_loader(tmp_path)
    restore_data_state(fresh, saved)
    assert fresh.current_step == 2
    x, _ = next(iter(fresh))

    # The first batch after resume is the 3rd batch of the file (file
    # positions 32..47; tokens are (i % 31) + 1).
    assert x[0].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]


def test_restore_data_state_rejects_non_memmap_checkpoints(tmp_path):
    # Checkpoints from the old trainers (streaming runs saved kind
    # "dataset"; older runs saved no data state) are not resumable.
    fresh = make_loader(tmp_path)
    with pytest.raises(ValueError, match="not resumable"):
        restore_data_state(fresh, {"kind": "dataset", "state": {}})

    fresh2 = make_loader(tmp_path)
    with pytest.raises(ValueError, match="not resumable"):
        restore_data_state(fresh2, None)


def test_check_resume_consistency_rejects_a_pass_length_change():
    with pytest.raises(ValueError, match="batches per"):
        check_resume_consistency(
            saved_total_steps=100,
            total_steps=100,
            saved_steps_per_pass=5,
            steps_per_pass=4,
            saved_tokens_per_step=16,
            tokens_per_step=16,
            decay_run=False,
        )


def test_check_resume_consistency_rejects_a_tokens_per_step_change():
    # A changed tokens-per-step (batch_size, seq_len, or accum_steps)
    # invalidates the saved cursor and the budget math.
    with pytest.raises(ValueError, match="tokens-per-step"):
        check_resume_consistency(
            saved_total_steps=100,
            total_steps=100,
            saved_steps_per_pass=40,
            steps_per_pass=40,
            saved_tokens_per_step=16,
            tokens_per_step=32,
            decay_run=False,
        )


def test_check_resume_consistency_rejects_a_missing_steps_per_pass():
    # Only checkpoints carrying a steps_per_pass value are resumable.
    with pytest.raises(ValueError, match="steps_per_pass"):
        check_resume_consistency(
            saved_total_steps=100,
            total_steps=100,
            saved_steps_per_pass=None,
            steps_per_pass=4,
            saved_tokens_per_step=16,
            tokens_per_step=16,
            decay_run=False,
        )


def test_restore_training_state_round_trip(tmp_path):
    """A payload built from a running position restores step, counters,
    W&B identity, and the memmap data cursor into a fresh loader bundle."""
    loader = make_loader(tmp_path)
    it = iter(loader)
    next(it)
    next(it)
    next(it)  # current_step == 3

    payload = build_checkpoint_payload(
        model_state={},
        model_cfg={},
        optimizer_state={},
        scheduler_state={},
        step=3,
        best_val_loss=1.5,
        tokens_seen=48,
        run_elapsed_seconds=10.0,
        steps_per_pass=BATCHES_PER_PASS,
        tokens_per_step=16,
        total_steps=100,
        wandb_run_id="stub-run-id",
        rng_state={
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
        },
        cfg_container={},
        wsd_state={"triggered": False, "step": None},
        data_state=capture_data_state(loader),
    )
    assert payload["checkpoint_version"] == 4

    fresh = make_loader(tmp_path)
    state = RunState(
        step=0,
        best_val_loss=float("inf"),
        wandb_run_id=None,
        run_elapsed_seconds=0.0,
        tokens_seen=0,
        tokens_per_step=16,
        train_loader=fresh,
    )
    restore_training_state(state, payload)

    assert state.step == 3
    assert state.best_val_loss == 1.5
    assert state.wandb_run_id == "stub-run-id"
    assert state.tokens_seen == 48
    assert fresh.current_step == 3


def test_checkpoint_write_is_atomic_and_readable(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_torch_save({"step": 12, "tensor": torch.tensor([1, 2, 3])}, path)

    checkpoint = load_checkpoint(path)
    assert checkpoint["step"] == 12
    torch.testing.assert_close(checkpoint["tensor"], torch.tensor([1, 2, 3]))
    assert not list(tmp_path.glob(".latest.pt.tmp.*"))


def test_parameter_metrics_count_tied_parameters_once():
    model = nn.Module()
    model.first = nn.Linear(4, 4, bias=False)
    model.second = nn.Linear(4, 4, bias=False)
    model.second.weight = model.first.weight
    model.frozen = nn.Parameter(torch.ones(3), requires_grad=False)

    metrics = get_model_parameter_metrics(model)

    assert metrics["model/parameters_total"] == 19
    assert metrics["model/parameters_trainable"] == 16
    assert metrics["model/parameters_non_trainable"] == 3
