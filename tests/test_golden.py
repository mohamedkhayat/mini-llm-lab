"""T0.2: golden characterization of the training math.

These tests pin invariants 1-3 of the trainer-refactor spec: the loss/LR
trajectory of a seeded CPU/fp32 run, the v4 checkpoint key set, and
resume-equals-uninterrupted. Every later phase must keep them green.

The run is driven through the REAL trainer (``Trainer`` + ``train()``) over
the ``tiny_cfg``/``tiny_memmap`` fixtures; per-step losses and LRs are read
from the terminal progress line (the public console seam), which the
terminal logger prints at ``log_interval``.
"""

import random
import re
from pathlib import Path

import numpy as np
import torch

from conftest import TOKENS_PER_STEP
from training.checkpointing import load_checkpoint
from training.assembly import build_trainer

# The terminal progress line: ``step N | tokens T | loss L | lr R | tok/s S``
LINE = re.compile(
    r"^step (\d+) \| tokens [\d.]+[KMB]? \| loss (\d+\.\d{4}) \| lr (\S+) \| tok/s",
    re.M,
)

# Golden trajectory: first 6 steps of a fresh 6-step run, seed 42, CPU/fp32,
# tiny_cfg defaults + train_tokens=6*64, warmup_fraction=0.5, log_interval=1.
# Recorded 2026-10-09 from the T0.1-fixed code (the pre-refactor baseline).
# LR is the .2e-formatted string (3-step linear warmup to 5e-4).
GOLDEN_LOSSES = [5.7827, 5.619, 5.7772, 5.6959, 5.7679, 5.773]
GOLDEN_LRS = ["3.33e-04", "5.00e-04", "5.00e-04", "5.00e-04", "5.00e-04", "5.00e-04"]

# The v4 checkpoint payload key set (spec section 5.8: unchanged by the
# refactor; ``checkpoint_version`` stays 4).
CHECKPOINT_KEYS_V4 = {
    "checkpoint_version",
    "model",
    "model_cfg",
    "optimizer",
    "scheduler",
    "step",
    "best_val_loss",
    "tokens_seen",
    "run_elapsed_seconds",
    "steps_per_pass",
    "tokens_per_step",
    "total_steps",
    "wandb_run_id",
    "rng",
    "cfg",
    "wsd",
    "data_state",
}


def _progress_steps(out):
    """Parse {step: (loss, lr_string)} from terminal progress lines."""
    return {
        int(m.group(1)): (float(m.group(2)), m.group(3)) for m in LINE.finditer(out)
    }


def _run(cfg, capsys):
    trainer = build_trainer(cfg)
    trainer.train()
    out = capsys.readouterr().out
    return trainer, _progress_steps(out)


def _assert_same_weights(state_a, state_b):
    assert set(state_a) == set(state_b)
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"weights differ: {key}"


def test_golden_trajectory(tiny_cfg, capsys):
    """Invariant 1: the first 6 losses and LRs match the recorded values."""
    tiny_cfg.training.train_tokens = 6 * TOKENS_PER_STEP
    tiny_cfg.training.warmup_fraction = 0.5
    tiny_cfg.training.log_interval = 1
    trainer, steps = _run(tiny_cfg, capsys)

    assert trainer.state.step == 6
    assert [steps[s][0] for s in range(1, 7)] == GOLDEN_LOSSES
    assert [steps[s][1] for s in range(1, 7)] == GOLDEN_LRS


def test_checkpoint_keys_v4(tiny_cfg, capsys):
    """Invariant 2: the payload key set and version are the v4 contract."""
    tiny_cfg.training.train_tokens = 2 * TOKENS_PER_STEP
    trainer, _ = _run(tiny_cfg, capsys)

    for name in ("final_model.pt", "latest.pt"):
        checkpoint = load_checkpoint(Path(trainer.ckpt.save_dir) / name)
        assert set(checkpoint) == CHECKPOINT_KEYS_V4, name
        assert checkpoint["checkpoint_version"] == 4, name


class _Rng:
    """All RNG streams captured after a run (the checkpoint restores all
    of them, so a bit-identical continuation ends with identical streams)."""

    def __init__(self):
        self.torch = torch.get_rng_state()
        self.numpy = np.random.get_state()[1].copy()
        self.python = list(random.getstate()[1])

    def equals(self, other):
        return (
            torch.equal(self.torch, other.torch)
            and (self.numpy == other.numpy).all()
            and self.python == other.python
        )


def test_resume_equivalence_stable(tiny_cfg, capsys):
    """Invariant 3 (stable): run(8) == run(4) + resume + run(4).

    The split checkpoint is the periodic save at the step-4 group boundary
    (clean cursor: no mid-accumulation group is skipped — that stop-path
    behavior is pinned separately by T1.1). Weights, per-step losses, and
    RNG state must all be identical.
    """
    tiny_cfg.training.train_tokens = 8 * TOKENS_PER_STEP
    tiny_cfg.training.log_interval = 1

    # Uninterrupted 8-step run.
    trainer_a, steps_a = _run(tiny_cfg, capsys)
    weights_a = load_checkpoint(Path(trainer_a.ckpt.save_dir) / "final_model.pt")[
        "model"
    ]
    rng_a = _Rng()
    assert set(steps_a) == set(range(1, 9))

    # First 4 steps with a periodic save at step 4 (then let it finish so
    # the periodic file stays on disk; it is never modified after write).
    tiny_cfg.training.save_interval = 4
    trainer_b1, steps_b1 = _run(tiny_cfg, capsys)
    assert [steps_b1[s][0] for s in range(1, 9)] == [steps_a[s][0] for s in range(1, 9)]
    split = Path(trainer_b1.ckpt.save_dir) / "step_4.pt"
    assert split.is_file()
    capsys.readouterr()

    # Resume from the step-4 checkpoint to the same total budget.
    trainer_b2 = build_trainer(tiny_cfg, checkpoint=load_checkpoint(split))
    trainer_b2.train()
    assert trainer_b2.state.step == 8
    out_b2 = capsys.readouterr().out

    steps_b2 = _progress_steps(out_b2)
    assert set(steps_b2) == {5, 6, 7, 8}
    resumed_losses = [steps_b2[s][0] for s in range(5, 9)]
    uninterrupted_losses = [steps_a[s][0] for s in range(5, 9)]
    assert resumed_losses == uninterrupted_losses

    weights_b2 = load_checkpoint(Path(trainer_b2.ckpt.save_dir) / "final_model.pt")[
        "model"
    ]
    _assert_same_weights(weights_a, weights_b2)

    assert _Rng().equals(rng_a)
