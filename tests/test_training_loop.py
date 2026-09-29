"""Behavioral tests for the training loop, driven through the public
``Trainer.train()`` entry on a stub-constructed trainer.

The stub skips the config-driven constructor (``Trainer.__new__``) and sets
the attributes the loop touches: a tiny real model (with a stubbed
``generate`` call), a real optimizer and scheduler, real
``MemmapDataLoader`` instances over a small tokenized ``.bin`` cache,
a recording logger, and a temp save directory. No GPU, no W&B, no corpora —
the loop, the data stream, the checkpoints on disk, and the logger calls
are all real.
"""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from data.dataloader import MemmapDataLoader
from models.gpt import GptModel
from training.checkpointing import load_checkpoint
from training.log_backend import RunLogger
from training.trainer import Trainer, get_model_parameter_metrics


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """CPU-only test file: never query the shared (possibly memory-starved)
    GPU for RNG state or timers."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

VOCAB = 32
SEQ_LEN = 8
BATCH_SIZE = 2
BATCH_TOKENS = BATCH_SIZE * SEQ_LEN  # 16 tokens per optimizer step
# 201 tokens -> (201 - 1) // 16 = 12 batches per data pass.
TRAIN_TOKENS = 201
STEPS_PER_PASS = 12
# Eval file: 49 tokens -> 3 batches (>= eval_batches below).
EVAL_TOKENS = 49


def write_cache(tmp_path):
    """Write a minimal prepared dataset (train.bin / eval.bin / meta.json).

    Token ids stay below VOCAB so the stub model can embed them.
    """
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    train = (np.arange(TRAIN_TOKENS) % (VOCAB - 1)) + 1
    eval_tokens = ((np.arange(EVAL_TOKENS) + 13) % (VOCAB - 1)) + 1
    train.astype(np.uint16).tofile(data_dir / "train.bin")
    eval_tokens.astype(np.uint16).tofile(data_dir / "eval.bin")
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_name": "gpt2",
                "dtype": "uint16",
                "max_tokens": TRAIN_TOKENS + EVAL_TOKENS,
                "train_tokens": TRAIN_TOKENS,
                "val_tokens": EVAL_TOKENS,
                "val_ratio": 0.375,
            },
            f,
        )
    return data_dir


class StubTokenizer:
    """Just enough of the tiktoken.Encoding interface for sample logging."""

    def encode(self, text, allowed_special=None):
        return [1, 2]

    def decode(self, ids):
        return "stub sample text"


class RecordingLogger(RunLogger):
    """Test logger implementing the run-logger interface: records calls."""

    def __init__(self):
        self.init_calls = []
        self.metric_batches = []
        self.samples = []
        self.checkpoints = []
        self.finished = False

    def init(
        self,
        *,
        name,
        config,
        computed,
        stage_marker,
        parameter_metrics,
        step,
        saved_config=None,
        resumed_run_id=None,
    ) -> None:
        self.init_calls.append(
            {
                "name": name,
                "config": config,
                "computed": computed,
                "stage_marker": stage_marker,
                "parameter_metrics": parameter_metrics,
                "step": step,
                "saved_config": saved_config,
                "resumed_run_id": resumed_run_id,
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


class StoppingLoader:
    """MemmapDataLoader wrapper that can trip the stop flag between steps.

    Delegates iteration to the wrapped loader; after ``trip_after`` batches
    have been yielded it sets ``stop_requested`` on the trainer, and can
    instead raise ``KeyboardInterrupt`` mid-iteration (a signal-free way to
    drive the interrupt path deterministically).
    """

    def __init__(self, loader, trainer, trip_after=None, interrupt_after=None):
        self._loader = loader
        self._trainer = trainer
        self._trip_after = trip_after
        self._interrupt_after = interrupt_after

    def __iter__(self):
        for batch_idx, batch in enumerate(self._loader):
            if self._interrupt_after is not None and batch_idx == self._interrupt_after:
                raise KeyboardInterrupt
            yield batch
            if self._trip_after is not None and batch_idx + 1 == self._trip_after:
                self._trainer.stop_requested = True

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        # Forward state_dict / current_step and friends to the loader.
        return getattr(self._loader, name)


def build_stub_trainer(tmp_path, training_overrides=None, **attribute_overrides):
    """Assemble a real, runnable trainer without the config constructor.

    Tiny real model (stubbed ``generate``), real AdamW + LambdaLR, real
    memmap dataloaders over a small tokenized cache, a recording logger,
    temp save dir. ``training_overrides`` patches the training config
    section (the loop reads its intervals from there); remaining keyword
    arguments override plain trainer attributes.
    """
    torch.manual_seed(0)

    data_dir = write_cache(tmp_path)

    model_cfg = OmegaConf.create(
        {
            "name": "gpt2",
            "attention": "mha",
            "tie_embeddings": False,
            "vocab_size": VOCAB,
            "context_length": SEQ_LEN,
            "emb_dim": 32,
            "hidden_dim": 64,
            "n_heads": 2,
            "n_kv_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": True,
            "ffn_bias": True,
            "norm": "layernorm",
            "activation": "gelu",
            "gated": False,
            "equalize_params": False,
            "position_embedding": "absolute",
            "residual_style": "serial",
            "logit_softcap": None,
            "rope_theta": None,
            "temperature": 1.0,
            "top_k": 0,
        }
    )
    model = GptModel(model_cfg)
    model.generate = staticmethod(lambda *a, **k: torch.tensor([[1, 2]]))

    training = {
        "seed": 0,
        "device": "cpu",
        "batch_size": BATCH_SIZE,
        "accum_steps": 1,
        "lr": 1e-2,
        "min_lr": 1e-3,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_fraction": 0.0,
        "decay_tokens": None,
        "start_decay": False,
        "log_interval": 2,
        "eval_interval": 4,
        "eval_batches": 2,
        "save_interval": 0,
        "max_new_tokens": 2,
        "start_context": "stub",
        "exp_name": "stub-run",
        "log_backend": "terminal",
        "use_bf16": False,
        "use_tensor_cores": False,
        "compile": False,
        "compile_mode": "default",
    }
    if training_overrides:
        training.update(training_overrides)
    cfg = OmegaConf.create(
        {
            "training": training,
            "data": {
                "seq_len": SEQ_LEN,
                "tokenizer_name": "gpt2",
                "hf_dataset": "stub",
            },
            "model": model_cfg,
        }
    )

    train_loader = MemmapDataLoader(
        data_dir, "train", BATCH_SIZE, SEQ_LEN, torch.device("cpu")
    )
    val_loader = MemmapDataLoader(
        data_dir, "eval", BATCH_SIZE, SEQ_LEN, torch.device("cpu")
    )
    train_eval_loader = MemmapDataLoader(
        data_dir, "train", BATCH_SIZE, SEQ_LEN, torch.device("cpu")
    )

    assert len(train_loader) == STEPS_PER_PASS

    # Mirror the real _setup_data accounting: the loaders are built with the
    # micro-batch geometry; steps_per_pass and tokens_per_step count
    # optimizer steps over the accumulated micro-batches.
    accum_steps = int(training["accum_steps"])

    trainer = Trainer.__new__(Trainer)
    attributes = {
        "cfg": cfg,
        "model_cfg": model_cfg,
        "model": model,
        "tokenizer": StubTokenizer(),
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_eval_loader": train_eval_loader,
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-2),
        "scheduler": None,  # created below: must wrap the same optimizer
        "criterion": torch.nn.CrossEntropyLoss(),
        "device": torch.device("cpu"),
        "use_bf16": False,
        "use_tensor_cores": False,
        "use_compile": False,
        "compile_mode": "default",
        "model_parameter_metrics": get_model_parameter_metrics(model),
        "log_backend": "terminal",
        "logger": RecordingLogger(),
        "step": 0,
        "best_val_loss": float("inf"),
        "tokens_seen": 0,
        "run_elapsed_seconds": 0.0,
        "stop_requested": False,
        "wandb_run_id": None,
        "accum_steps": accum_steps,
        "steps_per_pass": max(1, len(train_loader) // accum_steps),
        "tokens_per_step": BATCH_TOKENS * accum_steps,
        "total_steps": 10,
        "budget_name": "max_steps",
        "wsd_decay": {"triggered": False, "step": None},
        "stage_label": "stage-1-stable",
        "save_dir": str(tmp_path),
        "run_name": "stub-run",
        "_manifest_git_commit": "deadbeef",
        "_manifest_config_digest": "cfg123",
    }
    attributes["total_train_tokens"] = (
        attributes["total_steps"] * attributes["tokens_per_step"]
    )
    for name, value in attributes.items():
        setattr(trainer, name, value)
    # The scheduler must wrap the same optimizer the loop steps.
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda step: 1.0
    )
    for name, value in attribute_overrides.items():
        setattr(trainer, name, value)
    return trainer


def test_build_scheduler_rejects_checkpoints_missing_scheduler_or_optimizer_state(
    tmp_path,
):
    # Resume requires the full optimizer/scheduler state; checkpoints from
    # the old trainers are not resumable.
    trainer = build_stub_trainer(tmp_path, continuation_run=False)
    trainer.resume_checkpoint = {
        "scheduler": None,
        "optimizer": trainer.optimizer.state_dict(),
    }
    with pytest.raises(ValueError, match="scheduler"):
        trainer._build_scheduler()

    trainer.resume_checkpoint = {
        "scheduler": trainer.scheduler.state_dict(),
        "optimizer": None,
    }
    with pytest.raises(ValueError, match="optimizer"):
        trainer._build_scheduler()

    # A complete state restores in the load-bearing order (scheduler first,
    # so the checkpoint's exact current LR wins).
    trainer.resume_checkpoint = {
        "scheduler": trainer.scheduler.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
    }
    trainer._build_scheduler()


def test_budget_stop_writes_final_and_latest_checkpoints(tmp_path):
    """A run that hits its step budget ends cleanly at exactly the total
    steps: final + latest checkpoints on disk, closing run metrics with
    progress 1.0, and the run finished. The budget (10 steps) fits in the
    file's 12-step capacity (single pass)."""
    trainer = build_stub_trainer(
        tmp_path, training_overrides={"eval_interval": 999}, total_steps=10
    )

    trainer.train()

    assert trainer.step == 10
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "final_model.pt").is_file()
    assert not (tmp_path / "best.pt").exists()

    final = trainer.logger.metric_batches[-1]
    assert final["run/progress"] == 1.0
    assert final["run/steps_remaining"] == 0
    assert final["step"] == 10
    assert trainer.logger.finished is True


def test_stop_flag_between_steps_writes_latest_and_logs_the_interrupt(tmp_path):
    """When the stop flag is tripped between steps the loop stops after the
    current step: latest checkpoint written, the interrupted-run metric
    logged, and a clean exit (no final checkpoint)."""
    trainer = build_stub_trainer(
        tmp_path, training_overrides={"eval_interval": 999}, total_steps=10
    )
    trainer.train_loader = StoppingLoader(trainer.train_loader, trainer, trip_after=4)

    trainer.train()

    assert trainer.step == 4
    assert (tmp_path / "latest.pt").is_file()
    assert not (tmp_path / "final_model.pt").exists()

    last = trainer.logger.metric_batches[-1]
    assert last["run/interrupted"] == 1
    assert last["run/steps_remaining"] == 6
    assert trainer.logger.finished is True


def test_keyboard_interrupt_from_the_data_iterator_takes_the_graceful_path(tmp_path):
    """A KeyboardInterrupt raised by the data iterator takes the same
    graceful path as a signal stop: latest checkpoint written, the
    interrupted-run metric logged, and no re-raise."""
    trainer = build_stub_trainer(
        tmp_path, training_overrides={"eval_interval": 999}, total_steps=10
    )
    trainer.train_loader = StoppingLoader(
        trainer.train_loader, trainer, interrupt_after=3
    )

    trainer.train()  # must not re-raise

    assert trainer.step == 3
    assert (tmp_path / "latest.pt").is_file()
    assert not (tmp_path / "final_model.pt").exists()

    assert any("run/interrupted" in m for m in trainer.logger.metric_batches)
    assert trainer.logger.finished is True


def test_best_checkpoint_written_only_on_improvement_and_rewritten_later(tmp_path):
    """Each evaluation saves best.pt only when the val loss improves, and a
    later improvement rewrites it: the file on disk carries the lowest val
    loss seen, and exactly one best save happened per improvement. The
    evaluations also exercise the new train-loss shadow loader and the
    deterministic eval split."""
    trainer = build_stub_trainer(tmp_path, total_steps=12)

    trainer.train()

    assert (tmp_path / "best.pt").is_file()
    val_losses = trainer.logger.logged("val/loss")
    assert len(val_losses) == 3

    improvements = 0
    best = float("inf")
    for val_loss in val_losses:
        if val_loss < best:
            best = val_loss
            improvements += 1

    best_saves = trainer.logger.best_saves()
    assert len(best_saves) == improvements
    # The fixture (seeded, dropout-free) improves at every evaluation, so a
    # later evaluation rewrites the file saved by an earlier one.
    assert improvements >= 2

    saved = load_checkpoint(tmp_path / "best.pt")
    assert saved["best_val_loss"] == best


def test_save_interval_writes_periodic_step_checkpoints_and_latest(tmp_path):
    """At each save interval the loop writes a periodic step checkpoint and
    refreshes latest.pt (plus the final pair at completion). The periodic
    checkpoint carries the memmap data cursor."""
    trainer = build_stub_trainer(
        tmp_path,
        training_overrides={"eval_interval": 999, "save_interval": 4},
        total_steps=8,
    )

    trainer.train()

    for name in ("step_4.pt", "step_8.pt", "latest.pt", "final_model.pt"):
        assert (tmp_path / name).is_file(), name

    # Periodic saves hit the manifest as "periodic" entries; only best/final
    # checkpoints are uploaded as W&B artifacts, so the final one is the
    # only checkpoint the run's logger records.
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    checkpoints = manifest["checkpoints"]
    assert checkpoints["step_4.pt"]["kind"] == "periodic"
    assert checkpoints["step_4.pt"]["step"] == 4
    assert checkpoints["step_8.pt"]["kind"] == "periodic"
    assert checkpoints["step_8.pt"]["step"] == 8
    assert checkpoints["latest.pt"]["kind"] == "latest"

    assert [c["kind"] for c in trainer.logger.checkpoints] == ["final"]

    saved_step_4 = load_checkpoint(tmp_path / "step_4.pt")
    assert saved_step_4["step"] == 4
    assert saved_step_4["steps_per_pass"] == trainer.steps_per_pass
    assert saved_step_4["data_state"] == {
        "kind": "memmap",
        "state": {"current_step": 4},
    }
    # Single pass: 8 steps = 8 batches, so the data cursor sits at
    # batch 8 — always step * accum_steps (accum=1 today).
    saved_final = load_checkpoint(tmp_path / "final_model.pt")
    assert saved_final["data_state"] == {
        "kind": "memmap",
        "state": {"current_step": 8},
    }


def test_resume_starts_from_the_saved_batch_position(tmp_path):
    """A resumed run continues from the loader's saved position (not from
    the top of the file) and runs to the budget without rescan: with the
    cursor at batch 3 and a budget of 12, the run consumes batches 3..11
    and ends exactly at the file's end."""
    trainer = build_stub_trainer(
        tmp_path,
        training_overrides={"eval_interval": 999},
        total_steps=12,
    )
    trainer.step = 3
    trainer.tokens_seen = 3 * BATCH_TOKENS
    trainer.train_loader.load_state_dict({"current_step": 3})

    trainer.train()

    assert trainer.step == 12
    assert trainer.tokens_seen == 12 * BATCH_TOKENS
    assert trainer.train_loader.current_step == 12


def test_gradient_accumulation_steps_once_per_micro_batch_group(tmp_path):
    """With accum_steps=2 each optimizer step consumes 2 micro-batches:
    the loader cursor advances in micro-batch units, tokens count the
    effective batch (batch_size x seq_len x accum_steps), and the loss
    line appears once per optimizer step."""
    trainer = build_stub_trainer(
        tmp_path,
        training_overrides={"accum_steps": 2, "log_interval": 1},
        total_steps=4,
    )

    trainer.train()

    assert trainer.step == 4
    assert trainer.tokens_seen == 4 * 2 * BATCH_TOKENS
    assert trainer.train_loader.current_step == 8
    assert len(trainer.logger.logged("train/loss")) == 4


def test_budget_larger_than_one_pass_is_rejected(tmp_path):
    """The trainer does not rescan the data: a budget beyond the file's
    capacity raises before any step, pointing at data.max_tokens."""
    trainer = build_stub_trainer(tmp_path, total_steps=13)

    with pytest.raises(ValueError, match="data.max_tokens"):
        trainer.train()

    assert trainer.step == 0
    assert trainer.tokens_seen == 0
    assert trainer.train_loader.current_step == 0
    assert not (tmp_path / "final_model.pt").exists()
