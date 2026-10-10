"""Behavioral tests for the training loop, driven through the public
``Trainer.train()`` entry on REAL trainers built by ``build_trainer``
(the ``make_trainer`` fixture over ``tiny_cfg``/``tiny_memmap`` — spec
5.12; the stub constructor seam was retired by T6.3).

Everything is real: a tiny GPT-2 model, a real optimizer and scheduler,
real ``MemmapDataLoader`` instances over the seeded ``.bin`` cache, a
recording logger, and a temp save directory. No GPU, no W&B, no corpora.
"""

import json
import warnings
from pathlib import Path

import pytest
import torch

from conftest import TOKENS_PER_STEP
from training.checkpointing import check_cursor_consistency, load_checkpoint


class StoppingLoader:
    """MemmapDataLoader wrapper that can trip the stop flag between steps.

    Delegates iteration to the wrapped loader; after ``trip_after`` batches
    have been yielded it requests a stop on the trainer's shutdown object, and
    can instead raise ``KeyboardInterrupt`` mid-iteration (a signal-free way
    to drive the force-kill path deterministically).
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
                self._trainer.shutdown.stop.requested = True

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        # Forward state_dict / current_step and friends to the loader.
        return getattr(self._loader, name)


def test_build_scheduler_rejects_checkpoints_missing_scheduler_or_optimizer_state(
    make_trainer, tmp_path
):
    """Resume requires the full optimizer/scheduler state; checkpoints
    missing either are not resumable. A complete checkpoint restores in
    the load-bearing order (scheduler first, so the checkpoint's exact
    current LR wins)."""
    trainer, logger = make_trainer(
        {"save_interval": 2, "eval_interval": 999, "train_tokens": 4 * TOKENS_PER_STEP}
    )
    trainer.train()
    step_2 = load_checkpoint(Path(trainer.ckpt.save_dir) / "step_2.pt")

    broken = {**step_2, "scheduler": None}
    with pytest.raises(ValueError, match="scheduler"):
        make_trainer({}, checkpoint=broken)

    broken = {**step_2, "optimizer": None}
    with pytest.raises(ValueError, match="optimizer"):
        make_trainer({}, checkpoint=broken)

    # A complete checkpoint resumes: the state, loader cursor, and the
    # scheduler's exact LR are all restored.
    resumed, _ = make_trainer(
        {"eval_interval": 999, "train_tokens": 4 * TOKENS_PER_STEP},
        checkpoint=step_2,
    )
    assert resumed.state.step == 2
    # The data cursor is in MICRO-batch units: step 2 x accum 2 = batch 4.
    assert resumed.train_loader.current_step == 4


def test_budget_stop_writes_final_and_latest_checkpoints(make_trainer):
    """A run that hits its step budget ends cleanly at exactly the total
    steps: final + latest checkpoints on disk, closing run metrics with
    progress 1.0, and the run finished."""
    trainer, logger = make_trainer(
        {"eval_interval": 999, "train_tokens": 10 * TOKENS_PER_STEP}
    )

    trainer.train()

    assert trainer.state.step == 10
    save_dir = Path(trainer.ckpt.save_dir)
    assert (save_dir / "latest.pt").is_file()
    assert (save_dir / "final_model.pt").is_file()
    # F7: the final eval no longer saves best.pt — with eval_interval
    # disabled there is no periodic eval, so no best checkpoint at all.
    assert not (save_dir / "best.pt").exists()

    final = logger.metric_batches[-1]
    assert final["run/progress"] == 1.0
    assert final["run/steps_remaining"] == 0
    assert final["step"] == 10
    assert logger.finished is True


def test_stop_flag_between_steps_writes_latest_and_logs_the_interrupt(make_trainer):
    """When the stop flag is tripped between steps the loop stops after the
    current step: latest checkpoint written, the interrupted-run metric
    logged, and a clean exit (no final checkpoint)."""
    trainer, logger = make_trainer(
        {"eval_interval": 999, "train_tokens": 10 * TOKENS_PER_STEP}
    )
    trainer.train_loader = StoppingLoader(
        trainer.train_loader, trainer, trip_after=8  # 4 steps x 2 micro-batches
    )

    trainer.train()

    assert trainer.state.step == 4
    save_dir = Path(trainer.ckpt.save_dir)
    assert (save_dir / "latest.pt").is_file()
    assert not (save_dir / "final_model.pt").exists()

    last = logger.metric_batches[-1]
    assert last["run/interrupted"] == 1
    assert last["run/steps_remaining"] == 6
    assert logger.finished is True


def test_stop_mid_pull_keeps_the_cursor_consistent_with_the_step(make_trainer):
    """F2: a stop requested during a pull discards the pulled micro-batches
    and must restore the data cursor to its pre-pull position, so the
    checkpoint's cursor equals (step - start_step) * accum_steps + the
    phase's start cursor."""
    trainer, logger = make_trainer(
        {"eval_interval": 999, "train_tokens": 10 * TOKENS_PER_STEP}
    )
    trainer.train_loader = StoppingLoader(
        trainer.train_loader, trainer, trip_after=8  # trips inside a pull
    )

    trainer.train()

    assert trainer.state.step == 4
    # The discarded pull must not count: the cursor sits exactly at
    # step * accum_steps (4 x 2 = 8, not 10).
    assert trainer.train_loader.current_step == 8
    saved = load_checkpoint(Path(trainer.ckpt.save_dir) / "latest.pt")
    assert saved["data_state"] == {
        "kind": "memmap",
        "state": {"current_step": 8},
    }
    check_cursor_consistency(
        trainer.state, trainer.train_loader, trainer.accum_steps,
        trainer.budget.phase,
    )


def test_stop_mid_pull_then_resume_matches_uninterrupted_run(make_trainer):
    """F2 resume-equivalence: a stop injected mid-pull checkpoints the
    pre-pull cursor; resuming from it must consume exactly the remaining
    data and end with weights identical to an uninterrupted run of the
    same budget (no skipped or duplicated micro-batches)."""
    reference, _ = make_trainer(
        {"eval_interval": 999, "train_tokens": 10 * TOKENS_PER_STEP}
    )
    reference.train()

    trainer, _ = make_trainer(
        {"eval_interval": 999, "train_tokens": 10 * TOKENS_PER_STEP}
    )
    trainer.train_loader = StoppingLoader(
        trainer.train_loader, trainer, trip_after=8
    )
    trainer.train()
    assert trainer.state.step == 4

    checkpoint = load_checkpoint(Path(trainer.ckpt.save_dir) / "latest.pt")
    resumed, _ = make_trainer(
        {"eval_interval": 999, "train_tokens": 10 * TOKENS_PER_STEP},
        checkpoint=checkpoint,
    )
    resumed.train()

    reference_state = reference.model.state_dict()
    resumed_state = resumed.model.state_dict()
    for key in reference_state:
        torch.testing.assert_close(
            resumed_state[key], reference_state[key], atol=0, rtol=0
        )


def test_stop_mid_pull_then_resume_exhausts_the_file_without_ran_out(
    make_trainer
):
    """F2: a one-pass budget that fills the file exactly (62 steps x 2
    micro-batches = 124 = the whole tiny train file). Pre-F2 the
    discarded pull skipped 2 batches, so the resumed run hit
    'Ran out of data' before the budget was reached."""
    trainer, _ = make_trainer(
        {"eval_interval": 999, "train_tokens": 62 * TOKENS_PER_STEP}
    )
    trainer.train_loader = StoppingLoader(
        trainer.train_loader, trainer, trip_after=8
    )
    trainer.train()
    assert trainer.state.step == 4

    checkpoint = load_checkpoint(Path(trainer.ckpt.save_dir) / "latest.pt")
    resumed, _ = make_trainer(
        {"eval_interval": 999, "train_tokens": 62 * TOKENS_PER_STEP},
        checkpoint=checkpoint,
    )
    resumed.train()
    assert resumed.state.step == 62


def test_keyboard_interrupt_from_the_iterator_propagates(make_trainer):
    """A KeyboardInterrupt from the data iterator is no longer swallowed:
    after T1.1 the graceful stop comes from GracefulShutdown (the signal
    handler requests it), and a raw interrupt takes the force-kill path —
    the run is closed, the handlers are restored, and the exception
    propagates (see tests/test_shutdown.py for the second-signal case)."""
    trainer, logger = make_trainer(
        {"eval_interval": 999, "train_tokens": 10 * TOKENS_PER_STEP}
    )
    trainer.train_loader = StoppingLoader(
        trainer.train_loader, trainer, interrupt_after=3
    )

    with pytest.raises(KeyboardInterrupt):
        trainer.train()

    # The loop was closed (tracking finished) despite the abort.
    assert logger.finished is True
    # No checkpoint was written for the aborted step.
    save_dir = Path(trainer.ckpt.save_dir)
    assert not (save_dir / "latest.pt").exists()
    assert not (save_dir / "final_model.pt").exists()


def test_best_checkpoint_written_only_on_improvement_and_rewritten_later(make_trainer):
    """Each evaluation saves best.pt only when the val loss improves, and a
    later improvement rewrites it: the file on disk carries the lowest val
    loss seen, and exactly one best save happened per improvement."""
    trainer, logger = make_trainer(
        {"eval_interval": 4, "train_tokens": 12 * TOKENS_PER_STEP}
    )

    trainer.train()

    save_dir = Path(trainer.ckpt.save_dir)
    assert (save_dir / "best.pt").is_file()
    val_losses = logger.logged("val/loss")
    # 3 periodic evals (interval 4 over 12 steps). The final full-scan eval
    # logs final/* and is not a val eval (F7).
    assert len(val_losses) == 3
    assert len(logger.logged("final/loss")) == 1

    improvements = 0
    best = float("inf")
    for val_loss in val_losses:
        if val_loss < best:
            best = val_loss
            improvements += 1

    best_saves = logger.best_saves()
    assert len(best_saves) == improvements
    # The fixture (seeded, dropout-free) improves at every evaluation, so a
    # later evaluation rewrites the file saved by an earlier one.
    assert improvements >= 2

    saved = load_checkpoint(save_dir / "best.pt")
    assert saved["best_val_loss"] == best


def test_save_interval_writes_periodic_step_checkpoints_and_latest(make_trainer):
    """At each save interval the loop writes a periodic step checkpoint and
    republishes latest.pt from it (one write, P4); the run ends with the
    final checkpoint event."""
    trainer, logger = make_trainer(
        {"eval_interval": 999, "save_interval": 4, "train_tokens": 8 * TOKENS_PER_STEP}
    )

    trainer.train()

    save_dir = Path(trainer.ckpt.save_dir)
    for name in ("step_4.pt", "step_8.pt", "latest.pt", "final_model.pt"):
        assert (save_dir / name).is_file(), name

    # Periodic saves hit the manifest as "periodic" entries; only best/final
    # checkpoints are uploaded as W&B artifacts. With eval_interval disabled
    # there is no periodic eval, and F7 removed the final eval's best save,
    # so the final checkpoint is the only artifact the run records.
    manifest = json.loads((save_dir / "run_manifest.json").read_text())
    checkpoints = manifest["checkpoints"]
    assert checkpoints["step_4.pt"]["kind"] == "periodic"
    assert checkpoints["step_4.pt"]["step"] == 4
    assert checkpoints["step_8.pt"]["kind"] == "periodic"
    assert checkpoints["step_8.pt"]["step"] == 8
    assert checkpoints["latest.pt"]["kind"] == "latest"

    assert [c["kind"] for c in logger.checkpoints] == ["final"]

    saved_step_4 = load_checkpoint(save_dir / "step_4.pt")
    assert saved_step_4["step"] == 4
    assert saved_step_4["steps_per_pass"] == trainer.budget.steps_per_pass
    assert saved_step_4["data_state"] == {
        "kind": "memmap",
        "state": {"current_step": 8},  # 4 steps x 2 micro-batches
    }
    # Single pass: 8 steps x 2 micro-batches = 16 batches, so the data
    # cursor sits at batch 16 — always step * accum_steps (accum=2).
    saved_final = load_checkpoint(save_dir / "final_model.pt")
    assert saved_final["data_state"] == {
        "kind": "memmap",
        "state": {"current_step": 16},
    }

def test_resume_starts_from_the_saved_batch_position(make_trainer):
    """A resumed run continues from the loader's saved position (not from
    the top of the file) and runs to the budget without rescan: with the
    cursor at batch 3 and a budget of 12, the run consumes the remaining
    batches and ends exactly at the budget."""
    trainer, logger = make_trainer(
        {"eval_interval": 999, "train_tokens": 12 * TOKENS_PER_STEP}
    )
    trainer.state.step = 3
    trainer.state.tokens_seen = 3 * TOKENS_PER_STEP
    trainer.train_loader.load_state_dict({"current_step": 6})

    trainer.train()

    assert trainer.state.step == 12
    assert trainer.state.tokens_seen == 12 * TOKENS_PER_STEP
    # Cursor started at micro-batch 6 (= step 3) and consumed 9 steps x
    # 2 micro-batches: 6 + 18 = 24.
    assert trainer.train_loader.current_step == 24


def test_gradient_accumulation_steps_once_per_micro_batch_group(make_trainer):
    """With accum_steps=2 (the tiny fixture default) each optimizer step
    consumes 2 micro-batches: the loader cursor advances in micro-batch
    units, tokens count the effective batch (batch_size x seq_len x
    accum_steps), and the loss line appears once per optimizer step."""
    trainer, logger = make_trainer(
        {"log_interval": 1, "train_tokens": 4 * TOKENS_PER_STEP}
    )

    trainer.train()

    assert trainer.state.step == 4
    assert trainer.state.tokens_seen == 4 * TOKENS_PER_STEP
    assert trainer.train_loader.current_step == 8
    assert len(logger.logged("train/loss")) == 4


def test_train_loss_is_the_interval_mean(make_trainer):
    """F6: the logged train/loss is the arithmetic mean over the log
    interval (add() every step, mean_and_reset at log steps) — not a
    single-step sample."""
    import torch

    trainer, logger = make_trainer(
        {"log_interval": 4, "eval_interval": 999, "train_tokens": 8 * TOKENS_PER_STEP}
    )
    known = [4.0, 2.0, 6.0, 1.0, 2.0, 8.0, 6.0, 4.0]
    it = iter(known)

    def stub_train_step(micro_batches):
        return torch.tensor(next(it)), torch.tensor(1.0)

    trainer.train_step = stub_train_step
    trainer.train()

    logged = logger.logged("train/loss")
    assert len(logged) == 2
    assert logged[0] == pytest.approx((4.0 + 2.0 + 6.0 + 1.0) / 4)  # 3.25
    assert logged[1] == pytest.approx((2.0 + 8.0 + 6.0 + 4.0) / 4)  # 5.0


def test_cpu_run_with_bf16_requested_enters_no_autocast(make_trainer):
    """F8: use_bf16=true on a CPU device resolves to False in the
    runtime — the loss path must never enter a CUDA autocast (which
    warns on every call when CUDA is unavailable), so a full run stays
    clean with that warning escalated to an error."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*CUDA is not available.*"
        )
        trainer, _ = make_trainer(
            {
                "use_bf16": True,
                "eval_interval": 999,
                "train_tokens": 3 * TOKENS_PER_STEP,
            }
        )
        trainer.train()

    assert trainer.state.step == 3


def test_budget_larger_than_one_pass_is_rejected(make_trainer, tmp_path):
    """The trainer does not rescan the data: a budget beyond the file's
    capacity (62 steps for the tiny cache) raises BEFORE the trainer is
    even built — and therefore before any logging/W&B side effect (P3),
    pointing at data.max_tokens."""
    with pytest.raises(ValueError, match="data.max_tokens"):
        make_trainer({"train_tokens": 100 * TOKENS_PER_STEP})

    assert not (tmp_path / "runs" / "final_model.pt").exists()
