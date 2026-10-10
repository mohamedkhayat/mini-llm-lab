"""Run metrics: timing, throughput, and the GPU-side loss accumulator.

Plain objects owned by the trainer (``clock``, ``meter``, ``loss_accum``) —
nothing writes timer bookkeeping onto another object. The loss accumulator
is the only place ``train/loss`` syncs to the host: ``add`` stashes a
detached tensor every optimizer step, ``mean_and_reset`` is the single
``.item()`` (the interval mean, read at log steps).
"""

import time

import torch


class RunClock:
    """Wall-clock elapsed seconds, offset by ``elapsed_before``.

    A resumed run keeps counting from where the checkpoint's
    ``run_elapsed_seconds`` left off.
    """

    def __init__(self, elapsed_before: float):
        self._elapsed_before = float(elapsed_before)
        self._start = time.perf_counter()

    def restart(self, elapsed_before: float) -> None:
        """Restart the wall-clock window at ``elapsed_before`` (F9): time
        spent before the call — assembly, W&B init, compile — is not run
        time. The clock is shared (trainer loop + checkpoint before_save),
        so it is reset in place."""
        self._elapsed_before = float(elapsed_before)
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return self._elapsed_before + (time.perf_counter() - self._start)


class ThroughputMeter:
    """Interval timing: CUDA events on cuda, perf_counter otherwise.

    ``tick`` keeps the ``torch.cuda.synchronize`` so GPU timing stays
    correct — it is the only intended sync on log steps.
    """

    def __init__(self, device):
        self._device = device
        self._use_cuda = device.type == "cuda"
        self._timer_start: torch.cuda.Event | None = None
        self._timer_end: torch.cuda.Event | None = None
        self._start_time: float | None = None
        self._start_tokens = 0

    def start(self, tokens_seen: int):
        self._start_tokens = int(tokens_seen)
        if self._use_cuda:
            torch.cuda.synchronize(self._device)
            self._timer_start = torch.cuda.Event(enable_timing=True)
            self._timer_end = torch.cuda.Event(enable_timing=True)
            self._timer_start.record()
        else:
            self._start_time = time.perf_counter()

    def tick(self, tokens_seen: int) -> tuple[float, float]:
        """Return ``(interval_seconds, tok_per_sec)`` and roll the window.

        Requires a preceding ``start()``; raises otherwise.
        """
        if self._use_cuda:
            if self._timer_start is None or self._timer_end is None:
                raise RuntimeError("ThroughputMeter.tick before start()")
            self._timer_end.record()
            torch.cuda.synchronize(self._device)
            elapsed = self._timer_start.elapsed_time(self._timer_end) / 1000
            self._timer_start.record()
        else:
            if self._start_time is None:
                raise RuntimeError("ThroughputMeter.tick before start()")
            elapsed = time.perf_counter() - self._start_time
            self._start_time = time.perf_counter()
        tokens_since = int(tokens_seen) - self._start_tokens
        self._start_tokens = int(tokens_seen)
        return elapsed, tokens_since / max(elapsed, 1e-9)


class LossAccumulator:
    """GPU-side train/loss buffer: no host sync until ``mean_and_reset``.

    ``add`` is called every optimizer step with the step's (already
    detached) loss tensor; ``mean_and_reset`` computes the interval mean —
    the single ``.item()`` — and must be called with at least one pending
    add (the loop only reads it at log steps, after full steps).
    """

    def __init__(self):
        self._values: list = []

    def add(self, loss: torch.Tensor):
        self._values.append(loss.detach())

    def mean_and_reset(self) -> float:
        if not self._values:
            raise RuntimeError(
                "LossAccumulator.mean_and_reset with no pending steps; "
                "add() runs every optimizer step, so this means the mean "
                "was read before any step completed."
            )
        mean = torch.stack(self._values).mean()
        self._values.clear()
        return float(mean.item())


def gpu_memory_metrics(device) -> dict:
    """CUDA memory gauges in GiB; empty on non-cuda devices."""
    if device.type != "cuda":
        return {}
    return {
        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated(device)
        / 1024**3,
        "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved(device)
        / 1024**3,
        "system/gpu_memory_peak_allocated_gb": torch.cuda.max_memory_allocated(
            device
        )
        / 1024**3,
        "system/gpu_memory_peak_reserved_gb": torch.cuda.max_memory_reserved(
            device
        )
        / 1024**3,
    }
