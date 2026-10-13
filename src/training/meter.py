"""Run metrics: timing, throughput, and the GPU-side loss accumulator.

Plain objects owned by the trainer (``clock``, ``meter``, ``loss_accum``) —
nothing writes timer bookkeeping onto another object. The loss accumulator
is the only place ``train/loss`` syncs to the host: ``add`` stashes a
detached tensor every optimizer step, ``mean_and_reset`` is the single
``.item()`` (the interval mean, read at log steps).
"""

import time
from collections import defaultdict
import re
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


_GROUPS = (
    ("tok_emb", "emb_head"),
    ("lm_head", "emb_head"),
    ("pos_emb", "pos_emb"),
    ("W_query", "attn_q"),
    ("W_key", "attn_k"),
    ("W_value", "attn_v"),
    (".qkv.", "attn_qkv"),
    ("out_proj", "attn_out"),
    (".ffn.", "ffn"),
)  # anything else (the norm scales) falls through to "norm"
_LAYER = re.compile(r"trf_blocks\.(\d+)\.")


def _group(name: str) -> str:
    return next((g for key, g in _GROUPS if key in name), "norm")


class GradMonitor:
    """Spike diagnostics, GPU-side like ``LossAccumulator``.

    ``add`` runs every optimizer step (no host sync); ``report`` runs at log
    steps. Call ``add`` after backward and BEFORE clipping: clip_grad_norm_
    rescales ``.grad`` in place.
    """

    def __init__(self, model, clip_norm: float):
        raw = getattr(model, "_orig_mod", model)
        named = [(n, p) for n, p in raw.named_parameters() if p.requires_grad]
        self._names = [n for n, _ in named]
        self._params = [p for _, p in named]
        self._clip = float(clip_norm)
        self._norms: list = []  # per step: [num_params] pre-clip grad norms
        self._micro: list = []  # per step: [accum_steps] micro-batch losses

    @torch.no_grad()
    def add(self, micro_losses) -> None:
        zero = self._params[0].new_zeros(())
        self._norms.append(
            torch.stack(
                [
                    torch.linalg.vector_norm(p.grad) if p.grad is not None else zero
                    for p in self._params
                ]
            )
        )
        self._micro.append(torch.stack(micro_losses))

    @torch.no_grad()
    def report(self) -> tuple[float, dict]:
        """Return (interval-mean grad norm, extra metrics) and reset."""
        if not self._norms:
            raise RuntimeError("GradMonitor.report with no recorded steps")
        pp = torch.stack(self._norms)  # [steps, params]
        micro = torch.stack(self._micro)  # [steps, accum]
        self._norms.clear()
        self._micro.clear()

        gn = pp.norm(dim=1)  # [steps] total pre-clip grad norm
        mean, peak, clipped, micro_max = torch.stack(
            [gn.mean(), gn.max(), (gn > self._clip).sum().float(), micro.max()]
        ).tolist()
        extras = {
            "train/grad_norm_max": peak,
            "train/clipped_steps": clipped,
            "train/micro_loss_max": micro_max,
        }

        # Breakdown at the worst step of the interval.
        worst = int(gn.argmax())
        groups, layers = defaultdict(float), defaultdict(float)
        for name, v in zip(self._names, pp[worst].tolist()):
            groups[_group(name)] += v * v
            if m := _LAYER.search(name):
                layers[int(m.group(1))] += v * v
        extras.update({f"grad/group/{g}": s**0.5 for g, s in groups.items()})
        extras.update(
            {f"grad/layer/{i:02d}": s**0.5 for i, s in sorted(layers.items())}
        )
        extras.update(
            {
                f"grad/worst_micro_loss/{i}": v
                for i, v in enumerate(micro[worst].tolist())
            }
        )
        return mean, extras


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
        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "system/gpu_memory_peak_allocated_gb": torch.cuda.max_memory_allocated(device)
        / 1024**3,
        "system/gpu_memory_peak_reserved_gb": torch.cuda.max_memory_reserved(device)
        / 1024**3,
    }
