"""The training loop only (spec 5.12).

``Trainer`` receives ready-made components from ``build_trainer``
(assembly) and runs the loop.  This module holds no config, no omegaconf,
no signal handling, and no file paths: everything config-derived is
built once in assembly before the loop starts.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch.nn.utils import clip_grad_norm_

from training.meter import LossAccumulator, gpu_memory_metrics
from training.shutdown import GracefulShutdown
from training.tracking import ProgressSnapshot


@dataclass(frozen=True)
class Intervals:
    """The loop's cadence in optimizer steps."""

    log_interval: int
    eval_interval: int
    save_interval: int | None


class Trainer:
    """The loop-only trainer (spec 5.12): all components arrive ready-made.

    ``self.model`` may be a ``torch.compile`` wrapper; the wrapper
    preserves the module interface, and raw parameter access goes through
    ``_orig_mod`` — assembly handles that on save/restore, the loop only
    ever calls ``forward``/``parameters``/``train``.
    """

    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        train_loader,
        loss_fn,
        evaluator,
        final_evaluator,
        sampler,
        ckpt,
        tracker,
        state,
        budget,
        clock,
        meter,
        intervals,
        accum_steps,
        max_grad_norm,
        autocast_ctx,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        # The once-at-completion eval scans the FULL eval split
        # (num_batches=None) — a separate Evaluator built by assembly.
        self.final_evaluator = final_evaluator
        self.sampler = sampler
        self.ckpt = ckpt
        self.tracker = tracker
        self.state = state
        self.budget = budget
        self.clock = clock
        self.meter = meter
        self.intervals = intervals
        self.accum_steps = accum_steps
        self.max_grad_norm = max_grad_norm
        self.autocast_ctx = autocast_ctx
        # Derived from the model rather than a config: the loop only needs
        # it for the throughput meter and the GPU-memory metrics.
        self.device = next(model.parameters()).device
        # Held on the instance (T1.1) so a data seam (e.g. a test loader)
        # can request a stop between micro-batches.
        self.shutdown = GracefulShutdown()

    def get_lr(self):
        """Current learning rate, as set by the scheduler."""
        return self.optimizer.param_groups[0]["lr"]

    def train_step(self, micro_batches):
        """One optimizer step over a group of micro-batches (gradient
        accumulation): losses are scaled by 1/accum_steps, gradients
        clipped, then one optimizer + LR step. Returns
        (loss_tensor, grad_norm_tensor) with no host sync (spec #2)."""
        self.optimizer.zero_grad()
        total = 0.0
        for x, y in micro_batches:
            loss = self.loss_fn(x, y) / self.accum_steps
            loss.backward()
            total += loss.detach()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        # total is a Tensor here: the pull loop always supplies at least
        # one micro-batch, so the first += replaces the 0.0 seed.
        return cast("torch.Tensor", total), grad_norm

    def _log_progress(self, grad_norm):
        # The single .item() on the log path (the interval mean from the
        # GPU-side accumulator); the meter owns the interval timing.
        loss = self.loss_accum.mean_and_reset()
        elapsed, tok_per_sec = self.meter.tick(self.state.tokens_seen)
        self.state.run_elapsed_seconds = self.clock.elapsed()
        self.tracker.log_progress(
            ProgressSnapshot(
                step=self.state.step,
                total_steps=self.budget.total_steps,
                steps_per_pass=self.budget.steps_per_pass,
                tokens_seen=self.state.tokens_seen,
                tokens_per_step=self.budget.tokens_per_step,
                loss=loss,
                grad_norm=grad_norm,
                lr=self.get_lr(),
                tok_per_sec=tok_per_sec,
                interval_seconds=elapsed,
                elapsed_seconds=self.state.run_elapsed_seconds,
                gpu_metrics=gpu_memory_metrics(self.device),
            )
        )

    def _log_evaluation(self, result):
        """Log one evaluation + sample; save the best checkpoint on
        improvement."""
        self.tracker.log_evaluation(self.state.step, result)
        self.tracker.log_sample(
            self.state.step, self.sampler.sample(), loss=result.loss
        )
        if result.loss < self.state.best_val_loss:
            self.state.best_val_loss = result.loss
            self.ckpt.save("best", self.state)

    def _run_evaluation(self):
        """Every eval_interval: periodic eval over the batch-capped eval
        split (spec #4)."""
        self._log_evaluation(self.evaluator.run())

    def _run_final_evaluation(self):
        """Once at completion: a full scan of the eval split, logged
        through the same path."""
        self._log_evaluation(self.final_evaluator.run())

    def _handle_stop(self):
        """The unified stop path (SIGINT/SIGTERM/SIGHUP): checkpoint the
        current step, then log the interrupted-run metrics (spec #6)."""
        print(f"Saving restart checkpoint at step {self.state.step}...")
        self.ckpt.save("latest", self.state)
        self.tracker.log_interrupted(
            self.state.step, self.budget.total_steps, self.budget.tokens_per_step
        )

    def _finish_run(self):
        """Run completion: final full-scan eval, closing run metrics, and
        the single final checkpoint event (spec #6)."""
        self._run_final_evaluation()
        self.state.run_elapsed_seconds = self.clock.elapsed()
        self.tracker.log_finished(
            ProgressSnapshot(
                step=self.state.step,
                total_steps=self.budget.total_steps,
                steps_per_pass=self.budget.steps_per_pass,
                tokens_seen=self.state.tokens_seen,
                tokens_per_step=self.budget.tokens_per_step,
                loss=0.0,
                grad_norm=0.0,
                lr=self.get_lr(),
                tok_per_sec=0.0,
                interval_seconds=0.0,
                elapsed_seconds=self.state.run_elapsed_seconds,
                gpu_metrics={},
            )
        )
        self.ckpt.save("final", self.state)

    def train(self):
        """The outer training loop (the seven semantics of spec 5.12)."""
        self.model.train()
        self.meter.start(self.state.tokens_seen)
        self.loss_accum = LossAccumulator()

        with self.shutdown as stop:
            try:
                batch_iter = iter(self.train_loader)
                while (
                    self.state.step < self.budget.total_steps
                    and not stop.requested
                ):
                    # #1: pull one group of micro-batches. A stop requested
                    # during the pull is honored before the step: the pulled
                    # micro-batches are discarded and the data cursor is
                    # restored to its pre-pull position, so the stop
                    # checkpoint's cursor never lands ahead of state.step
                    # (a resume would otherwise skip data, and a one-pass
                    # run could end with 'Ran out of data').
                    cursor = self.train_loader.state_dict()
                    micro_batches = []
                    for _ in range(self.accum_steps):
                        try:
                            micro_batches.append(next(batch_iter))
                        except StopIteration:
                            raise RuntimeError(
                                "Ran out of data before the budget was reached; "
                                "the trainer does not rescan the file. This "
                                "should be impossible: assembly rejects budgets "
                                "larger than one pass over the data."
                            )
                    if stop.requested:
                        self.train_loader.load_state_dict(cursor)
                        break

                    # #2: one optimizer step, no host sync.
                    loss, grad_norm = self.train_step(micro_batches)
                    self.state.step += 1
                    self.state.tokens_seen += self.budget.tokens_per_step

                    # #3: one sync per log interval.
                    if self.state.step % self.intervals.log_interval == 0:
                        self.loss_accum.add(loss)
                        self._log_progress(grad_norm.item())
                    # #4: periodic evaluation + best checkpoint.
                    if self.state.step % self.intervals.eval_interval == 0:
                        self._run_evaluation()
                    # #5: periodic restart checkpoint.
                    if (
                        self.intervals.save_interval
                        and self.state.step % self.intervals.save_interval == 0
                    ):
                        self.ckpt.save("periodic", self.state)
                    if stop.requested:
                        break

                # #6: stop -> latest + interrupted metrics; else complete.
                if stop.requested:
                    self._handle_stop()
                else:
                    self._finish_run()
            finally:
                # #7: flush the log backend no matter how the loop exits.
                self.close()

    def close(self):
        """Flush and close the run's log backend."""
        self.tracker.finish()
