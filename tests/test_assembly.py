"""T6.2: build_trainer assembly (spec 5.11).

- A bad budget raises BEFORE the logger is initialised (zero W&B calls).
- ``resume_from`` without a loaded checkpoint refuses (no silent fresh start).
- The trainer never loads checkpoints itself (train.py is the only caller
  of load_checkpoint — P9).
"""

import pytest

from conftest import TOKENS_PER_STEP
from training import checkpointing
from training.assembly import build_trainer
from training import assembly as assembly_mod  # noqa: E402  (monkeypatch seam)


class _RecordingLogger:
    """Counts init() calls: tracker.start() is the W&B-side-effect seam."""

    def __init__(self):
        self.inits = 0

    def init(self, info):
        self.inits += 1

    def log_metrics(self, metrics, step=None):
        pass

    def log_checkpoint(self, *args, **kwargs):
        pass

    def finish(self):
        pass

    @property
    def is_terminal(self):
        return True


def _capture_logger(monkeypatch):
    logger = _RecordingLogger()

    def fake_create(backend, upload_artifacts=False):
        return logger

    monkeypatch.setattr(assembly_mod, "create_logger", fake_create)
    return logger


def test_budget_validated_before_logger(tiny_cfg, monkeypatch):
    """An over-one-pass budget raises in build_trainer BEFORE the logger
    is initialised (P3: a bad config produces zero W&B calls)."""
    logger = _capture_logger(monkeypatch)
    tiny_cfg.training.train_tokens = 100 * TOKENS_PER_STEP  # 100 > 62 steps/pass
    with pytest.raises(ValueError, match="exceeds one pass"):
        build_trainer(tiny_cfg)
    assert logger.inits == 0, "no logger init (W&B start) for a bad budget"


def test_resume_from_without_checkpoint_raises(tiny_cfg):
    """training.resume_from set but no checkpoint loaded: raise instead of
    silently starting a fresh run."""
    tiny_cfg.training.resume_from = "runs/2026-01-01/latest.pt"
    with pytest.raises(ValueError, match="resume_from"):
        build_trainer(tiny_cfg)


def test_trainer_never_loads_a_checkpoint(tiny_cfg, monkeypatch):
    """P9: the trainer never calls load_checkpoint — train.py is the only
    caller. build_trainer receives the already-loaded checkpoint."""
    calls = []
    real_load = checkpointing.load_checkpoint

    def spy(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(checkpointing, "load_checkpoint", spy)

    trainer = build_trainer(tiny_cfg)
    trainer.train()
    assert calls == [], "the trainer path must never load a checkpoint"
