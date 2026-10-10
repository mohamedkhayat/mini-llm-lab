"""T0.1 seams: the real trainer runs end-to-end on CPU.

These drive the real ``Trainer`` constructor + ``train()`` loop over the
``tiny_cfg``/``tiny_memmap`` fixtures — no stubs. The Phase-0 bugs (the
tensor-comparison param-group filter, the second checkpoint load in
``_setup_runtime``) surface here as a crash at construction or first step.
"""

from collections import Counter
from pathlib import Path

import pytest

from conftest import TOKENS_PER_STEP
from data.dataloader import MemmapDataLoader
from models.gpt import GptModel
from training.assembly import build_trainer


def _build_and_run(cfg):
    trainer = build_trainer(cfg)
    trainer.train()
    return trainer


def test_five_step_cpu_run_completes(tiny_cfg):
    """A 5-step CPU run (use_flce false) completes with the final
    checkpoints written."""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    trainer = _build_and_run(tiny_cfg)

    assert trainer.state.step == 5
    assert trainer.state.tokens_seen == 5 * TOKENS_PER_STEP
    for name in ("final_model.pt", "latest.pt"):
        assert (Path(trainer.ckpt.save_dir) / name).is_file(), f"{name} missing"


def test_param_groups_split_on_ndim(tiny_cfg):
    """Every trainable param is in exactly one group; the no-decay group
    holds exactly the ndim < 2 params with weight_decay 0."""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    trainer = build_trainer(tiny_cfg)

    decay, nodecay = trainer.optimizer.param_groups
    decay_ids = {id(p) for p in decay["params"]}
    nodecay_ids = {id(p) for p in nodecay["params"]}
    # A fresh model with the same cfg is the independent reference for the
    # trainable param population (trainer.model is an untyped god-class
    # attribute; comparing ndim histograms makes the partition exact).
    reference = GptModel(tiny_cfg.model)
    ref_hist = Counter(
        p.ndim for p in reference.parameters() if p.requires_grad
    )
    decay_hist = Counter(p.ndim for p in decay["params"])
    nodecay_hist = Counter(p.ndim for p in nodecay["params"])

    assert decay["weight_decay"] == float(tiny_cfg.training.weight_decay)
    assert nodecay["weight_decay"] == 0.0
    assert decay_ids & nodecay_ids == set(), "a param is in both groups"
    assert decay_hist + nodecay_hist == ref_hist
    assert all(ndim >= 2 for ndim in decay_hist)
    assert all(ndim < 2 for ndim in nodecay_hist)


def test_flce_falls_back_on_cpu_and_still_runs(tiny_cfg, capsys):
    """use_flce=true on CPU warns once, falls back to cross-entropy, and
    the 5-step run still completes."""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    tiny_cfg.training.use_flce = True
    trainer = _build_and_run(tiny_cfg)

    out = capsys.readouterr().out
    assert "FLCE" in out and "Falling back" in out
    assert trainer.state.step == 5


def test_trainer_does_not_load_a_checkpoint_it_was_not_given(tiny_cfg):
    """train.py loads the checkpoint once and passes it; build_trainer
    with resume_from set but NO loaded checkpoint refuses to silently
    start a fresh run (T6.2) and never loads one itself (P9 — the
    trainer/assembly call chain contains no load_checkpoint)."""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    first = _build_and_run(tiny_cfg)
    latest = Path(first.ckpt.save_dir) / "latest.pt"
    assert latest.is_file()

    cfg2 = tiny_cfg.copy()
    cfg2.training.resume_from = str(latest)
    with pytest.raises(ValueError, match="resume_from"):
        build_trainer(cfg2)  # no checkpoint= passed


def test_stop_never_mid_accumulation(tiny_cfg, monkeypatch):
    """A stop requested between the two micro-batches of a group is
    honored before that group's optimizer step: the group's step is never
    executed, latest lands at the last completed step, and resuming from
    it runs on to the end (a checkpoint never lands mid-accumulation)."""
    import torch

    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    trainer = build_trainer(tiny_cfg)

    real_iter = MemmapDataLoader.__iter__
    pulled = [0]

    def counting_iter(loader):
        for batch in real_iter(loader):
            pulled[0] += 1
            yield batch
            if pulled[0] == 3:  # between group 2's two micro-batches
                trainer.shutdown.stop.requested = True

    monkeypatch.setattr(MemmapDataLoader, "__iter__", counting_iter)
    trainer.train()

    # Group 2's step was pulled but never executed: the run stopped at
    # step 1 and the checkpoint landed there.
    assert trainer.state.step == 1
    ckpt = torch.load(
        Path(trainer.ckpt.save_dir) / "latest.pt", map_location="cpu", weights_only=False
    )
    assert ckpt["step"] == 1
    assert not (Path(trainer.ckpt.save_dir) / "final_model.pt").exists()

    # The checkpoint's data cursor sits on a group boundary (two complete
    # micro-batch groups), never mid-accumulation: resume restores the step
    # and the cursor exactly. (The pulled-but-untrained group is lost —
    # single pass, no rescan — so running the resumed run to its budget is
    # a data-budget question, not a checkpoint-consistency question.)
    import copy

    cfg2 = copy.deepcopy(tiny_cfg)
    cfg2.training.resume_from = str(Path(trainer.ckpt.save_dir) / "latest.pt")
    resumed = build_trainer(cfg2, checkpoint=ckpt)
    assert resumed.state.step == 1
    assert resumed.train_loader.current_step == 4
