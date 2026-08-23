"""Behavioral tests for the training loop, driven through the public
``Trainer.train()`` entry on a stub-constructed trainer.

The stub skips the config-driven constructor (``Trainer.__new__``) and sets
the attributes the loop touches: a tiny real model (with a stubbed
``generate`` call), a real optimizer and scheduler, in-memory dataloaders,
a recording logger, and a temp save directory. No GPU, no W&B, no corpora —
the loop, the checkpoints on disk, and the logger calls are all real.
"""

import json

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from data.dataloader import EpochRandomSampler
from models.gpt import GptModel
from training.checkpointing import load_checkpoint
from training.log_backend import RunLogger
from training.trainer import Trainer, get_model_parameter_metrics

VOCAB = 32
SEQ_LEN = 8
BATCH_SIZE = 2
DATASET_SIZE = 16


class RangeTokenDataset(Dataset):
    """In-memory token pairs: sample ``i`` is a short run of token ids."""

    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, index):
        tokens = torch.tensor(
            [(index + k) % VOCAB for k in range(SEQ_LEN)], dtype=torch.long
        )
        return tokens, tokens


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
    """DataLoader wrapper that can trip the stop flag between steps.

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
        # Forward sampler / generator / batch_size and friends to the loader.
        return getattr(self._loader, name)


def build_stub_trainer(tmp_path, training_overrides=None, **attribute_overrides):
    """Assemble a real, runnable trainer without the config constructor.

    Tiny real model (stubbed ``generate``), real AdamW + LambdaLR, in-memory
    dataloaders with the epoch sampler, a recording logger, temp save dir.
    ``training_overrides`` patches the training config section (the loop
    reads its intervals from there); remaining keyword arguments override
    plain trainer attributes.
    """
    torch.manual_seed(0)

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
        "lr": 1e-2,
        "min_lr": 1e-3,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_fraction": 0.0,
        "lr_decay_fraction": 0.2,
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
            "data": {"batch_size": BATCH_SIZE, "seq_len": SEQ_LEN},
            "model": model_cfg,
        }
    )

    def make_loader():
        return DataLoader(
            RangeTokenDataset(),
            batch_size=BATCH_SIZE,
            sampler=EpochRandomSampler(RangeTokenDataset(), seed=0),
        )

    train_loader = make_loader()
    val_loader = make_loader()

    trainer = Trainer.__new__(Trainer)
    attributes = {
        "cfg": cfg,
        "model_cfg": model_cfg,
        "model": model,
        "tokenizer": StubTokenizer(),
        "train_loader": train_loader,
        "val_loader": val_loader,
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
        "epoch": 0,
        "batch_in_epoch": 0,
        "tokens_seen": 0,
        "run_elapsed_seconds": 0.0,
        "stop_requested": False,
        "wandb_run_id": None,
        "steps_per_epoch": len(train_loader),
        "tokens_per_step": BATCH_SIZE * SEQ_LEN,
        "total_steps": 10,
        "budget_name": "max_steps",
        "wsd_decay": {"triggered": False, "step": None},
        "stage_label": "stage-1-stable",
        "save_dir": str(tmp_path),
        "run_name": "stub-run",
        "_manifest_git_commit": "deadbeef",
        "_manifest_config_digest": "cfg123",
    }
    attributes["total_train_tokens"] = attributes["total_steps"] * attributes[
        "tokens_per_step"
    ]
    for name, value in attributes.items():
        setattr(trainer, name, value)
    # The scheduler must wrap the same optimizer the loop steps.
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda step: 1.0
    )
    for name, value in attribute_overrides.items():
        setattr(trainer, name, value)
    return trainer


def test_budget_stop_writes_final_and_latest_checkpoints(tmp_path):
    """A run that hits its step budget ends cleanly at exactly the total
    steps: final + latest checkpoints on disk, closing run metrics with
    progress 1.0, and the run finished."""
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
    trainer.train_loader = StoppingLoader(
        trainer.train_loader, trainer, trip_after=4
    )

    trainer.train()

    assert trainer.step == 4
    assert (tmp_path / "latest.pt").is_file()
    assert not (tmp_path / "final_model.pt").exists()

    last = trainer.logger.metric_batches[-1]
    assert last["run/interrupted"] == 1
    assert last["run/steps_remaining"] == 6
    assert trainer.logger.finished is True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The KeyboardInterrupt path does not log the interrupted-run metric "
        "yet; ticket 06 unifies both stop paths on one handler, which this "
        "assertion turns green (the marker is removed then)."
    ),
)
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
    loss seen, and exactly one best save happened per improvement."""
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
    refreshes latest.pt (plus the final pair at completion)."""
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
    assert saved_step_4["cursor"] == {"epoch": 0, "batch_in_epoch": 4}


def test_resume_cursor_skips_consumed_batches_and_advances_the_epoch(tmp_path):
    """A resumed run starts mid-pass: already-consumed batches are skipped,
    the sampler's epoch setter is called for each logical data pass, and a
    full pass advances the epoch cursor."""
    trainer = build_stub_trainer(
        tmp_path,
        training_overrides={"eval_interval": 999},
        total_steps=13,
        step=3,
        epoch=0,
        batch_in_epoch=3,
        tokens_seen=3 * BATCH_SIZE * SEQ_LEN,
    )

    # Record the sampler's epoch setter calls across the whole run.
    sampler = trainer.train_loader.sampler
    epoch_calls = []
    original_set_epoch = sampler.set_epoch

    def recording_set_epoch(epoch):
        epoch_calls.append(epoch)
        original_set_epoch(epoch)

    sampler.set_epoch = recording_set_epoch

    trainer.train()

    # Pass 1 (resumed): only batches 3..7 consumed -> 5 steps; pass 2: the
    # budget runs out after 5 more steps, so the cursor points at batch 5.
    assert trainer.step == 13
    assert trainer.epoch == 1
    assert trainer.batch_in_epoch == 5
    assert trainer.tokens_seen == 13 * trainer.tokens_per_step
    assert epoch_calls == [0, 1]
    assert (tmp_path / "final_model.pt").is_file()