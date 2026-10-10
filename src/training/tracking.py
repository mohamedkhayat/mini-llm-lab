"""Run tracking with no trainer reference: all inputs are explicit values
(spec 5.7). The trainer (until T6.2: assembly) builds the ``RunInfo`` and
the ``ProgressSnapshot`` values; the tracker derives the run metrics,
owns the terminal console lines, and forwards everything to a
``RunLogger``.
"""

import math
from dataclasses import dataclass

from training.log_backend import RunInfo, RunLogger
from training.run_manifest import make_entry, upsert_manifest, utc_timestamp
from training.utils import format_count


@dataclass(frozen=True)
class ProgressSnapshot:
    """One progress log point. The caller owns the timing/syncing
    (``LossAccumulator``, ``ThroughputMeter``, ``RunClock``); the tracker
    derives the run/* metrics from these values."""

    step: int
    total_steps: int
    steps_per_pass: int
    tokens_seen: int
    tokens_per_step: int
    loss: float
    grad_norm: float
    lr: float
    tok_per_sec: float
    interval_seconds: float
    elapsed_seconds: float
    gpu_metrics: dict


class RunTracker:
    """Run events + checkpoint reporting for one run, behind an
    explicit-value interface."""

    def __init__(self, logger: RunLogger, *, save_dir, run_name: str,
                 git_commit, config_digest, terminal: bool = True):
        self._logger = logger
        self._save_dir = save_dir
        self._run_name = run_name
        self.git_commit = git_commit
        self.config_digest = config_digest
        self._terminal = terminal

    @property
    def logger(self):
        """The underlying RunLogger (inspection / test seam)."""
        return self._logger

    def start(self, info: RunInfo) -> None:
        self._logger.init(info)

    def log_metrics(self, metrics, step) -> None:
        self._logger.log_metrics(metrics, step)

    def log_progress(self, snap: ProgressSnapshot) -> None:
        """One progress point: the derived run/* metrics are computed here
        from the snapshot; the terminal console line is unchanged."""
        progress = min(snap.step / max(snap.total_steps, 1), 1.0)
        steps_remaining = max(snap.total_steps - snap.step, 0)
        steps_per_second = snap.step / max(snap.elapsed_seconds, 1e-9)
        estimated_remaining_seconds = steps_remaining / max(steps_per_second, 1e-9)
        metrics = {
            "train/loss": snap.loss,
            "train/ppl": math.exp(snap.loss),
            "train/lr": snap.lr,
            "train/tokens_seen": snap.tokens_seen,
            "train/tok_per_sec": snap.tok_per_sec,
            "train/interval_seconds": snap.interval_seconds,
            "run/elapsed_seconds": snap.elapsed_seconds,
            "run/elapsed_minutes": snap.elapsed_seconds / 60,
            "run/elapsed_hours": snap.elapsed_seconds / 3600,
            "run/steps_per_epoch": snap.steps_per_pass,
            "run/total_steps": snap.total_steps,
            "run/steps_remaining": steps_remaining,
            "run/tokens_remaining": steps_remaining * snap.tokens_per_step,
            "run/progress": progress,
            "run/estimated_remaining_seconds": estimated_remaining_seconds,
            "step": snap.step,
            **snap.gpu_metrics,
        }
        if snap.grad_norm is not None:
            metrics["train/grad_norm"] = snap.grad_norm
        self.log_metrics(metrics, snap.step)
        print(
            f"step {snap.step} | tokens {format_count(snap.tokens_seen)} | "
            f"loss {snap.loss:.4f} | lr {snap.lr:.2e} | tok/s {snap.tok_per_sec:,.0f}"
        )

    def log_evaluation(self, step, result) -> None:
        """Log an ``EvalResult``; the terminal line is unchanged."""
        self.log_metrics(
            {
                "val/loss": result.loss,
                "val/ppl": result.ppl,
                "val/stderr": result.stderr,
                "val/tokens": result.tokens,
                "step": step,
            },
            step,
        )
        if self._terminal:
            print(f"eval | step {step} | val loss {result.loss:.4f}")

    def log_sample(self, step, text, loss=None) -> None:
        self._logger.log_sample(text, step=step, loss=loss)

    def log_interrupted(self, step, total_steps, tokens_per_step) -> None:
        steps_remaining = total_steps - step
        self.log_metrics(
            {
                "run/interrupted": 1,
                "run/steps_remaining": steps_remaining,
                "run/tokens_remaining": max(steps_remaining, 0) * tokens_per_step,
            },
            step,
        )

    def log_finished(self, snap: ProgressSnapshot) -> None:
        self.log_metrics(
            {
                "run/elapsed_seconds": snap.elapsed_seconds,
                "run/elapsed_minutes": snap.elapsed_seconds / 60,
                "run/elapsed_hours": snap.elapsed_seconds / 3600,
                "run/steps_per_epoch": snap.steps_per_pass,
                "run/total_steps": snap.total_steps,
                "run/steps_remaining": 0,
                "run/tokens_remaining": 0,
                "run/progress": 1.0,
                "run/estimated_remaining_seconds": 0.0,
                "step": snap.step,
            },
            snap.step,
        )

    def record_checkpoint(self, kind, filename, *, step, tokens_seen, stage,
                          best_val_loss, wandb_run_id) -> None:
        """Update the local run manifest; failures never stop training."""
        try:
            wandb_url = self._logger.run_url
            entry = make_entry(
                file=filename,
                kind=kind,
                step=step,
                tokens_seen=tokens_seen,
                stage=stage,
                best_val_loss=best_val_loss,
                timestamp=utc_timestamp(),
                wandb_run_id=wandb_run_id,
                wandb_url=wandb_url,
            )
            upsert_manifest(
                self._save_dir,
                run_name=self._run_name,
                wandb_run_id=wandb_run_id,
                wandb_url=wandb_url,
                git_commit=self.git_commit,
                config_digest=self.config_digest,
                entry=entry,
            )
        except Exception as error:
            print(f"Warning: failed to update the run manifest: {error}")

    def log_checkpoint_artifact(self, kind, name, path, metadata) -> None:
        self._logger.log_checkpoint(kind, name, path, metadata)

    @property
    def run_url(self):
        return self._logger.run_url

    @property
    def run_id(self) -> str | None:
        """The underlying logger's run id (F3: forwarded so assembly can
        persist it in ``state.wandb_run_id`` after ``start``)."""
        return self._logger.run_id

    def finish(self) -> None:
        self._logger.finish()
