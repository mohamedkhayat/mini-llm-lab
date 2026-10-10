"""T6.1: persisted warmup — resume continues the ramp from the stored
``warmup_steps`` (independent of later config edits); old checkpoints
without the key fall back to the historical rule."""

import glob
import re
from pathlib import Path

import torch

from conftest import TOKENS_PER_STEP
from data.dataloader import MemmapDataLoader
from training.checkpointing import load_checkpoint
from training.assembly import build_trainer

LINE = re.compile(
    r"^step (\d+) \| tokens [\d.]+[KMB]? \| loss \S+ \| lr (\S+) \| tok/s",
    re.M,
)

STEPS = 10
RUN_TOKENS = STEPS * TOKENS_PER_STEP


def _lrs(out):
    """{step: lr_float} from the terminal progress lines."""
    return {int(m.group(1)): float(m.group(2)) for m in LINE.finditer(out)}


def _run(cfg, capsys, checkpoint=None):
    trainer = build_trainer(cfg, checkpoint=checkpoint)
    trainer.train()
    return _lrs(capsys.readouterr().out)


def test_warmup_resume(tiny_cfg, tmp_path, capsys):
    """Resume at a step inside warmup continues the ramp with the PERSISTED
    warmup_steps: changing warmup_fraction on relaunch must not change the
    LR curve."""
    tiny_cfg.training.train_tokens = RUN_TOKENS
    tiny_cfg.training.warmup_fraction = 0.5
    tiny_cfg.training.save_interval = 2
    tiny_cfg.training.log_interval = 1
    original = _run(tiny_cfg, capsys)

    assert original[3] < 5e-4, "step 3 must still be inside the warmup ramp"

    # Relaunch from step 2 with a different warmup fraction: the persisted
    # warmup (5 of 10 steps) must win over the new fraction (0.2 -> 2 steps).
    # P9: the checkpoint reaches the trainer through the constructor —
    # train.py is the only caller of load_checkpoint.
    step_2 = glob.glob(str(tmp_path / "**" / "step_2.pt"), recursive=True)
    assert step_2, "the periodic step_2.pt checkpoint must exist"
    tiny_cfg.training.warmup_fraction = 0.2
    tiny_cfg.training.save_interval = None

    resumed = _run(tiny_cfg, capsys, checkpoint=load_checkpoint(Path(step_2[0])))

    for step in range(3, STEPS + 1):
        assert resumed[step] == original[step], (
            f"step {step}: resumed LR must continue the original ramp"
        )


def test_old_checkpoint_without_warmup_steps_still_resumes(
    tiny_cfg, tmp_path, capsys
):
    """A checkpoint whose wsd entry has no ``warmup_steps`` key (written
    before this ticket) resumes via the historical rule: warmup is
    recomputed from the relaunch config's warmup_fraction."""
    tiny_cfg.training.train_tokens = RUN_TOKENS
    tiny_cfg.training.warmup_fraction = 0.5
    tiny_cfg.training.save_interval = 2
    tiny_cfg.training.log_interval = 1
    reference = _run(tiny_cfg, capsys)

    import glob

    step_2 = glob.glob(str(tmp_path / "**" / "step_2.pt"), recursive=True)
    assert step_2
    old_format = torch.load(step_2[0], weights_only=False)
    assert "warmup_steps" in old_format["wsd"], "new checkpoints persist it"
    del old_format["wsd"]["warmup_steps"]
    old_path = tmp_path / "step_2_old_format.pt"
    torch.save(old_format, old_path)

    # Relaunch from the old-format checkpoint with fraction 0.2: the
    # fallback rule recomputes warmup = int(0.2 * 10) = 2, so the resumed
    # curve is the one a fresh 0.2-fraction run would produce from step 3.
    tiny_cfg.training.warmup_fraction = 0.2
    tiny_cfg.training.save_interval = None

    resumed = _run(
        tiny_cfg, capsys, checkpoint=torch.load(old_path, weights_only=False)
    )
    assert resumed[3] == 5e-4, "warmup=2 ends at step 3: LR reaches peak"
    for step in range(4, STEPS + 1):
        assert resumed[step] == 5e-4


def test_phase_budget_carries_warmup_steps_in_as_dict():
    from training.schedule import PhaseBudget

    budget = PhaseBudget(
        mode="train_tokens",
        stage="stable",
        start_step=0,
        target_step=10,
        requested_tokens=RUN_TOKENS,
        effective_tokens=RUN_TOKENS,
        decay_start_step=None,
        decay_steps=None,
        warmup_steps=5,
    )
    assert budget.as_dict()["warmup_steps"] == 5
    # F9: the default is the None sentinel (resumes, decay runs, fresh
    # runs before resolution); resolve_budget fills it for fresh stable
    # starts and as_dict persists None as null.
    assert PhaseBudget(
        mode="one_pass", stage="stable", start_step=0, target_step=None,
        requested_tokens=None, effective_tokens=None,
    ).warmup_steps is None


def test_one_pass_resume_uses_the_persisted_warmup_steps(
    tiny_cfg, tmp_path, capsys, monkeypatch
):
    """F9: a one-pass checkpoint (wsd target_step None) carries its
    resolved warmup_steps; resuming after a warmup_fraction edit must
    keep the original LR curve (pre-F9 the exact_resume branch rebuilt
    the budget with warmup 0 and resolve_budget refilled it from the
    NEW config's fraction)."""
    tiny_cfg.training.train_tokens = None  # one-pass budget
    tiny_cfg.training.warmup_fraction = 0.5
    tiny_cfg.training.save_interval = 2
    tiny_cfg.training.log_interval = 1

    reference = build_trainer(tiny_cfg)
    reference.train()
    original = _lrs(capsys.readouterr().out)
    assert len(original) == 62, "the tiny cache is 62 optimizer steps"

    # First leg: stop after step 2 (the trip lands inside step 3's pull).
    trainer_box = {}
    pulled = [0]
    real_iter = MemmapDataLoader.__iter__

    def counting_iter(loader):
        for batch in real_iter(loader):
            pulled[0] += 1
            yield batch
            if pulled[0] == 4:
                trainer_box["trainer"].shutdown.stop.requested = True

    monkeypatch.setattr(MemmapDataLoader, "__iter__", counting_iter)
    leg1 = build_trainer(tiny_cfg)
    trainer_box["trainer"] = leg1
    leg1.train()
    monkeypatch.setattr(MemmapDataLoader, "__iter__", real_iter)
    assert leg1.state.step == 2
    latest = glob.glob(str(tmp_path / "**" / "latest.pt"), recursive=True)
    assert latest, "the stop checkpoint must exist"

    # Relaunch with a different warmup fraction: the persisted warmup
    # (int(0.5 * 62) = 31) must win over the new fraction (0.2 -> 12).
    tiny_cfg.training.warmup_fraction = 0.2
    tiny_cfg.training.save_interval = None
    resumed = build_trainer(
        tiny_cfg, checkpoint=load_checkpoint(Path(latest[0]))
    )
    resumed.train()
    resumed_lrs = _lrs(capsys.readouterr().out)

    for step in range(3, 63):
        assert resumed_lrs[step] == original[step], (
            f"step {step}: the resumed LR must keep the original ramp"
        )
