"""T5.2: RunTracker — no trainer reference; all inputs explicit values."""

import dataclasses
import json
import math

import pytest

from training.tracking import ProgressSnapshot, RunTracker


class _RecordingLogger:
    def __init__(self):
        self.init_calls = []
        self.metrics = []
        self.samples = []
        self.checkpoints = []
        self.finished = False

    def init(self, info):
        self.init_calls.append(info)

    def log_metrics(self, metrics, step):
        self.metrics.append((metrics, step))

    def log_sample(self, text, step, loss=None):
        self.samples.append((text, step, loss))

    def log_checkpoint(self, kind, name, path, metadata):
        self.checkpoints.append((kind, name, path, metadata))

    @property
    def run_url(self):
        return None

    def finish(self):
        self.finished = True


def _snap(**overrides):
    fields = dict(
        step=10, total_steps=100, steps_per_pass=50,
        tokens_seen=640, tokens_per_step=64,
        loss=1.23456, grad_norm=0.5, lr=5e-4,
        tok_per_sec=11464.3, interval_seconds=0.05,
        elapsed_seconds=20.0, gpu_metrics={"gpu/memory_used_gib": 1.5},
    )
    fields.update(overrides)
    return ProgressSnapshot(**fields)


def _tracker(tmp_path, terminal=True):
    logger = _RecordingLogger()
    tracker = RunTracker(
        logger, save_dir=str(tmp_path), run_name="run",
        git_commit="deadbeef", config_digest="cfg123", terminal=terminal,
    )
    return tracker, logger


def test_progress_snapshot_has_the_twelve_spec_fields():
    snap = _snap()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.step = 1
    assert {f.name for f in dataclasses.fields(snap)} == {
        "step", "total_steps", "steps_per_pass", "tokens_seen",
        "tokens_per_step", "loss", "grad_norm", "lr", "tok_per_sec",
        "interval_seconds", "elapsed_seconds", "gpu_metrics",
    }


def test_log_progress_derived_metrics_and_console_line(capsys):
    tracker, logger = _tracker(None)
    tracker.log_progress(_snap())

    metrics, step = logger.metrics[-1]
    assert step == 10
    # Derived inside the tracker from the snapshot.
    assert metrics["run/progress"] == pytest.approx(0.1)
    assert metrics["run/steps_remaining"] == 90
    assert metrics["run/tokens_remaining"] == 90 * 64
    # 10 steps in 20 s -> 0.5 steps/s -> 90 / 0.5
    assert metrics["run/estimated_remaining_seconds"] == pytest.approx(180.0)
    # Explicit values pass through.
    assert metrics["train/loss"] == pytest.approx(1.23456)
    assert metrics["train/ppl"] == pytest.approx(math.exp(1.23456))
    assert metrics["train/lr"] == 5e-4
    assert metrics["train/tokens_seen"] == 640
    assert metrics["train/tok_per_sec"] == pytest.approx(11464.3)
    assert metrics["train/interval_seconds"] == pytest.approx(0.05)
    assert metrics["train/grad_norm"] == 0.5
    assert metrics["run/steps_per_epoch"] == 50
    assert metrics["run/total_steps"] == 100
    assert metrics["gpu/memory_used_gib"] == 1.5

    # The terminal console line is unchanged in shape and formatting.
    line = capsys.readouterr().out.strip()
    assert line == (
        "step 10 | tokens 640 | loss 1.2346 | lr 5.00e-04 | tok/s 11,464"
    )


def test_log_evaluation_logs_the_full_result(tmp_path, capsys):
    tracker, logger = _tracker(tmp_path)
    from training.evaluator import EvalResult

    tracker.log_evaluation(
        12, EvalResult(loss=2.5, ppl=math.exp(2.5), stderr=0.1, tokens=128)
    )
    metrics, step = logger.metrics[-1]
    assert step == 12
    assert metrics["val/loss"] == pytest.approx(2.5)
    assert metrics["val/ppl"] == pytest.approx(math.exp(2.5))
    assert metrics["val/stderr"] == pytest.approx(0.1)
    assert metrics["val/tokens"] == 128
    assert capsys.readouterr().out.strip() == (
        "eval | step 12 | val loss 2.5000"
    )


def test_log_evaluation_terminal_line_only_in_terminal_mode(tmp_path, capsys):
    tracker, logger = _tracker(tmp_path, terminal=False)
    from training.evaluator import EvalResult

    tracker.log_evaluation(
        12, EvalResult(loss=2.5, ppl=math.exp(2.5), stderr=0.1, tokens=128)
    )
    assert "val/loss" in logger.metrics[-1][0]
    assert capsys.readouterr().out == ""


def test_log_interrupted_and_finished_metrics(tmp_path):
    tracker, logger = _tracker(tmp_path)
    tracker.log_interrupted(10, 100, 64)
    metrics, step = logger.metrics[-1]
    assert (metrics, step) == (
        {
            "run/interrupted": 1,
            "run/steps_remaining": 90,
            "run/tokens_remaining": 90 * 64,
        },
        10,
    )

    tracker.log_finished(_snap())
    metrics, step = logger.metrics[-1]
    assert step == 10
    assert metrics["run/progress"] == 1.0
    assert metrics["run/steps_remaining"] == 0
    assert metrics["run/tokens_remaining"] == 0
    assert metrics["run/estimated_remaining_seconds"] == 0.0
    assert metrics["run/elapsed_seconds"] == pytest.approx(20.0)
    assert metrics["run/steps_per_epoch"] == 50


def test_log_sample_and_run_url_and_finish(tmp_path):
    tracker, logger = _tracker(tmp_path)
    tracker.log_sample(10, "some text", loss=1.0)
    assert logger.samples == [("some text", 10, 1.0)]
    assert tracker.run_url is None
    tracker.finish()
    assert logger.finished is True


def test_start_forwards_run_info(tmp_path):
    tracker, logger = _tracker(tmp_path)
    from training.log_backend import RunInfo

    info = RunInfo(
        name="exp", config={}, saved_config=None, resumed_run_id=None,
        computed={}, stage_marker={}, parameter_metrics={}, step=0,
    )
    tracker.start(info)
    assert logger.init_calls == [info]


def test_record_checkpoint_updates_manifest(tmp_path):
    tracker, _ = _tracker(tmp_path)
    tracker.record_checkpoint(
        "periodic", "step_7.pt", step=7, tokens_seen=448,
        stage="stage-1-stable", best_val_loss=1.5, wandb_run_id=None,
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    entry = manifest["checkpoints"]["step_7.pt"]
    assert entry["kind"] == "periodic"
    assert entry["step"] == 7
    assert entry["tokens_seen"] == 448
    assert entry["stage"] == "stage-1-stable"
    assert manifest["git_commit"] == "deadbeef"
    assert manifest["config_digest"] == "cfg123"


def test_record_checkpoint_failure_never_raises(tmp_path, monkeypatch, capsys):
    import training.tracking as tracking_module

    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(tracking_module, "upsert_manifest", _boom)
    tracker, _ = _tracker(tmp_path)
    tracker.record_checkpoint(
        "latest", "latest.pt", step=1, tokens_seen=64,
        stage="stage-1-stable", best_val_loss=float("inf"),
        wandb_run_id=None,
    )  # must not raise
    assert "Warning: failed to update the run manifest" in capsys.readouterr().out


def test_log_checkpoint_artifact_forwards_metadata(tmp_path):
    tracker, logger = _tracker(tmp_path)
    metadata = {"kind": "best", "step": 42}
    tracker.log_checkpoint_artifact("best", "best-checkpoint", "/x/best.pt", metadata)
    assert logger.checkpoints == [
        ("best", "best-checkpoint", "/x/best.pt", metadata)
    ]
