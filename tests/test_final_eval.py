"""F7: the end-of-run final evaluation is separate from best-tracking.

The final full-scan eval logs final/*, records itself in the
final_model.pt manifest entry, and never touches best_val_loss,
best.pt, or the sample generator. An optional token cap bounds the
scan.
"""

import json
from pathlib import Path

import pytest

from conftest import BATCH_SIZE, SEQ_LEN, TOKENS_PER_STEP
from training.checkpointing import load_checkpoint


def test_final_eval_is_separate_from_best_tracking(make_trainer):
    """The final eval logs final/*, takes no sample, and leaves
    best_val_loss / best.pt at the periodic evals' result."""
    trainer, logger = make_trainer(
        {"eval_interval": 4, "train_tokens": 8 * TOKENS_PER_STEP}
    )
    trainer.train()

    # Two periodic evals (steps 4, 8) — and the final eval is NOT one of
    # them: it logs its own namespace.
    val_losses = logger.logged("val/loss")
    assert len(val_losses) == 2
    final_losses = logger.logged("final/loss")
    assert len(final_losses) == 1
    # One sample per periodic eval; the final eval does not sample.
    assert len(logger.samples) == 2

    # best_val_loss is the periodic best, untouched by the final eval.
    periodic_best = min(val_losses)
    assert trainer.state.best_val_loss == periodic_best
    improvements = 0
    best = float("inf")
    for loss in val_losses:
        if loss < best:
            best = loss
            improvements += 1
    assert len(logger.best_saves()) == improvements
    saved = load_checkpoint(Path(trainer.ckpt.save_dir) / "best.pt")
    assert saved["best_val_loss"] == periodic_best


def test_final_manifest_entry_records_the_final_eval(make_trainer):
    """The final_model.pt manifest entry carries an optional final_eval
    field (loss, tokens); the latest.pt alias entry does not."""
    trainer, logger = make_trainer(
        {"eval_interval": 999, "train_tokens": 4 * TOKENS_PER_STEP}
    )
    trainer.train()

    manifest = json.loads(
        (Path(trainer.ckpt.save_dir) / "run_manifest.json").read_text()
    )
    checkpoints = manifest["checkpoints"]
    assert checkpoints["final_model.pt"]["final_eval"] == {
        "loss": logger.logged("final/loss")[0],
        "tokens": logger.logged("final/tokens")[0],
    }
    assert "final_eval" not in checkpoints["latest.pt"]


def test_final_eval_cap_converts_to_a_batch_budget(make_trainer):
    """training.final_eval_max_tokens bounds the final scan at
    tokens // (batch_size * seq_len) batches (null = full scan);
    decimal-suffix values are accepted."""
    capped, _ = make_trainer(
        {
            "eval_interval": 999,
            "train_tokens": 4 * TOKENS_PER_STEP,
            "final_eval_max_tokens": 4 * BATCH_SIZE * SEQ_LEN,
        }
    )
    assert capped.final_evaluator._num_batches == 4

    suffixed, _ = make_trainer(
        {
            "eval_interval": 999,
            "train_tokens": 4 * TOKENS_PER_STEP,
            "final_eval_max_tokens": "512",
        }
    )
    assert suffixed.final_evaluator._num_batches == 512 // (
        BATCH_SIZE * SEQ_LEN
    )

    # make_trainer mutates one shared cfg: reset the cap explicitly for
    # the uncapped case.
    uncapped, _ = make_trainer(
        {
            "eval_interval": 999,
            "train_tokens": 4 * TOKENS_PER_STEP,
            "final_eval_max_tokens": None,
        }
    )
    assert uncapped.final_evaluator._num_batches is None
