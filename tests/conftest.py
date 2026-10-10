"""Shared fixtures for the trainer tests.

``tiny_cfg`` + ``tiny_memmap`` drive the REAL trainer end-to-end on CPU/fp32
with a terminal logger: 2 layers, emb_dim=32, 2 heads, vocab=256,
seq_len=16, batch_size=2, accum_steps=2. The memmap cache is written under
``tmp_path/data/<hf_dataset>/<tokenizer>/`` (the layout ``get_data_dir``
resolves after ``monkeypatch.chdir(tmp_path)``).
"""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from training.assembly import build_trainer
from training.log_backend import RunLogger

VOCAB = 256
SEQ_LEN = 16
BATCH_SIZE = 2
ACCUM_STEPS = 2
# Tokens consumed by one optimizer step (batch_size * seq_len * accum_steps).
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN * ACCUM_STEPS

TRAIN_TOKENS = 4000
EVAL_TOKENS = 512


class RecordingLogger(RunLogger):
    """Test logger implementing the run-logger interface: records calls."""

    def __init__(self):
        self.init_calls = []
        self.metric_batches = []
        self.samples = []
        self.checkpoints = []
        self.finished = False

    def init(self, info) -> None:
        self.init_calls.append(
            {
                "name": info.name,
                "config": info.config,
                "computed": info.computed,
                "stage_marker": info.stage_marker,
                "parameter_metrics": info.parameter_metrics,
                "step": info.step,
                "saved_config": info.saved_config,
                "resumed_run_id": info.resumed_run_id,
            }
        )

    def log_metrics(self, metrics, step) -> None:
        self.metric_batches.append(metrics)

    def log_sample(self, text, step, loss=None) -> None:
        self.samples.append({"text": text, "step": step, "loss": loss})

    def log_checkpoint(self, kind, name, path, metadata) -> None:
        self.checkpoints.append({"kind": kind, "name": name, "path": path})

    @property
    def run_url(self):
        return None

    def finish(self) -> None:
        self.finished = True

    # --- conveniences for the assertions below ---

    def logged(self, key):
        return [m[key] for m in self.metric_batches if key in m]

    def final_run_metrics(self):
        """The metrics batch with the closing run/progress key."""
        for metrics in reversed(self.metric_batches):
            if "run/progress" in metrics:
                return metrics
        return None

    def best_saves(self):
        return [c for c in self.checkpoints if c["kind"] == "best"]


@pytest.fixture
def tiny_memmap(tmp_path):
    """Write a prepared memmap cache (train.bin / eval.bin / meta.json) with
    seeded random tokens. Returns the data directory."""
    data_dir = tmp_path / "data" / "smoke" / "gpt2"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    train = rng.integers(1, VOCAB, size=TRAIN_TOKENS)
    eval_tokens = rng.integers(1, VOCAB, size=EVAL_TOKENS)
    train.astype(np.uint16).tofile(data_dir / "train.bin")
    eval_tokens.astype(np.uint16).tofile(data_dir / "eval.bin")
    meta = {
        "tokenizer_name": "gpt2",
        "dtype": "uint16",
        "max_tokens": TRAIN_TOKENS + EVAL_TOKENS,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": EVAL_TOKENS,
        "val_ratio": EVAL_TOKENS / (TRAIN_TOKENS + EVAL_TOKENS),
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return data_dir


@pytest.fixture
def tiny_cfg(tmp_path, tiny_memmap, monkeypatch):
    """A full config driving the real trainer over ``tiny_memmap``.

    CPU, fp32, terminal logging, no warmup, no eval/save during the run
    (eval_interval is huge; the end-of-run final eval lands in Phase 3).
    The GPU is stubbed out (same guard as ``tests/test_training_loop.py``):
    these tests must never touch the shared, possibly memory-starved GPU.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    return OmegaConf.create(
        {
            "data": {
                "hf_dataset": "smoke",
                "tokenizer_name": "gpt2",
                "seq_len": SEQ_LEN,
                "max_tokens": TRAIN_TOKENS + EVAL_TOKENS,
            },
            "model": {
                "name": "gpt2",
                "attention": "mha",
                "tie_embeddings": False,
                "vocab_size": VOCAB,
                "context_length": 128,
                "emb_dim": 32,
                "hidden_dim": 64,
                "n_heads": 2,
                "n_kv_heads": 2,
                "n_layers": 2,
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
                "temperature": 1.0,
                "top_k": 25,
            },
            "training": {
                "exp_name": "tiny",
                "device": "cpu",
                "compile": False,
                "compile_mode": "default",
                "use_bf16": False,
                "use_tensor_cores": False,
                "log_backend": "terminal",
                "upload_artifacts": False,
                "lr": 5e-4,
                "min_lr": 5e-5,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.95,
                "max_grad_norm": 1.0,
                "batch_size": BATCH_SIZE,
                "accum_steps": ACCUM_STEPS,
                "train_tokens": None,
                "decay_tokens": None,
                "continue_tokens": None,
                "warmup_fraction": 0.0,
                "eval_interval": 10**9,
                "eval_batches": 2,
                "log_interval": 2,
                "save_interval": None,
                "resume_from": None,
                "resume_mode": "exact",
                "start_decay": False,
                "seed": 42,
                "use_flce": False,
                # "a" encodes to token 64 (GPT-2) — inside the tiny model's
                # VOCAB (256). "stub" encodes to [301, 549], which would
                # index past the tiny embedding table in the generation
                # sample. The real configs pair a full-size vocab with the
                # real tokenizer, so this only constrains test fixtures.
                "start_context": "a",
                "max_new_tokens": 2,
            },
        }
    )


@pytest.fixture
def make_trainer(tiny_cfg):
    """Factory for REAL trainers: ``build_trainer`` over ``tiny_cfg`` /
    ``tiny_memmap`` with a ``RecordingLogger`` injected (terminal backend —
    no W&B, no GPU).

    ``training_overrides`` sets real config keys before assembly; a step
    budget is expressed as ``train_tokens = N * TOKENS_PER_STEP``. The
    checkpoint (an already-loaded dict, P9) is passed straight through.
    Returns ``(trainer, logger)``.
    """

    def factory(training_overrides=None, checkpoint=None):
        for key, value in dict(training_overrides or {}).items():
            setattr(tiny_cfg.training, key, value)
        logger = RecordingLogger()
        trainer = build_trainer(tiny_cfg, checkpoint=checkpoint, logger=logger)
        return trainer, logger

    return factory
