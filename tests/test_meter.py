"""T2.2: meter — RunClock, ThroughputMeter, LossAccumulator, gpu metrics."""

import time
import traceback

import pytest
import torch

from conftest import TOKENS_PER_STEP
from training.meter import (
    LossAccumulator,
    RunClock,
    ThroughputMeter,
    gpu_memory_metrics,
)
from training.assembly import build_trainer


def test_run_clock_offsets_elapsed_before():
    clock = RunClock(elapsed_before=10.0)
    first = clock.elapsed()
    assert 10.0 <= first < 11.0
    assert clock.elapsed() >= first  # perf_counter based: monotonic


def test_throughput_meter_cpu():
    meter = ThroughputMeter(torch.device("cpu"))
    meter.start(tokens_seen=0)
    time.sleep(0.01)
    interval, tok_per_sec = meter.tick(tokens_seen=320)

    assert interval > 0
    assert tok_per_sec == pytest.approx(320 / max(interval, 1e-9), rel=1e-9)
    # The window rolled: the next tick measures from 320.
    interval2, tok_per_sec2 = meter.tick(tokens_seen=320)
    assert tok_per_sec2 == 0.0


def test_loss_accumulator_mean_and_reset():
    acc = LossAccumulator()
    x = torch.tensor(5.7827)
    acc.add(x)
    # A single-element mean is the element itself (the golden literals
    # depend on this for log_interval=1).
    assert acc.mean_and_reset() == x.item()

    acc.add(torch.tensor(1.0))
    acc.add(torch.tensor(3.0))
    assert acc.mean_and_reset() == pytest.approx(2.0)


def test_loss_accumulator_mean_and_reset_empty_buffer_raises():
    """F6: reading the mean with no pending step is a loop bug; fail
    with a clear error instead of a nan/empty-stack surprise."""
    acc = LossAccumulator()
    with pytest.raises(RuntimeError, match="no pending"):
        acc.mean_and_reset()


def test_loss_accumulator_holds_detached_tensors():
    """add() stashes a detached view: no grad graph is retained."""
    acc = LossAccumulator()
    p = torch.tensor(2.0, requires_grad=True)
    acc.add(p * p)
    held = acc._values[0]
    assert not held.requires_grad
    assert torch.equal(held, torch.tensor(4.0))


def test_gpu_memory_metrics_empty_on_cpu():
    assert gpu_memory_metrics(torch.device("cpu")) == {}


def test_no_extra_syncs(tiny_cfg):
    """P1: non-log steps perform zero project-originated Tensor.item()
    calls — the loss stays on-device in the accumulator until a log step
    reads the mean. (torch's own AdamW calls .item() on per-parameter
    bookkeeping tensors inside site-packages; those are not counted —
    the gate is on the project's loss path.)"""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    tiny_cfg.training.log_interval = 10**9  # no log steps during the run

    calls = []
    real_item = torch.Tensor.item

    def counting_item(self, *a, **k):
        # The caller frame: count only calls made by project code, not by
        # torch internals (the optimizer's own bookkeeping syncs).
        caller = traceback.extract_stack()[-2]
        if "/site-packages/" not in caller.filename:
            calls.append(caller.filename)
        return real_item(self, *a, **k)

    torch.Tensor.item = counting_item
    try:
        build_trainer(tiny_cfg).train()
    finally:
        torch.Tensor.item = real_item

    assert calls == []
