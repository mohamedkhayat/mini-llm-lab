from pathlib import Path

import torch
import wandb

from conftest import TOKENS_PER_STEP
from training.assembly import build_trainer
from training.log_backend import (
    RunInfo,
    TerminalLogger,
    WandbLogger,
    build_artifact_metadata,
    log_artifact,
    should_log_artifacts,
)


class FakeRun:
    """The wandb run object returned by init (carries id + url)."""

    def __init__(self, run_id):
        self.id = run_id
        self.url = f"https://wandb.test/{run_id}"


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []
        self.metadata = {}

    def add_file(self, path):
        self.files.append(path)


class FakeWandb:
    """Minimal stand-in for the wandb module: records run and artifact activity."""

    def __init__(self, fail_at="none", run_id="abc123"):
        self.fail_at = fail_at  # "none" | "create" | "log"
        self.run_id = run_id
        self.run = None
        self.logged = []
        self.init_calls = []
        self.config = {}

    def Html(self, data):
        return f"<html>{data}</html>"

    def Artifact(self, name, type):
        if self.fail_at == "create":
            raise RuntimeError("wandb server unreachable")
        return FakeArtifact(name, type)

    def log_artifact(self, artifact):
        if self.fail_at == "log":
            raise RuntimeError("artifact upload failed")
        self.logged.append(artifact)

    def init(self, *args, **kwargs):
        self.init_calls.append(kwargs)
        # A resumed init keeps the requested id; a fresh one gets the
        # fixed (or configured) id.
        self.run = FakeRun(kwargs.get("id", self.run_id))
        return self.run

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class AlwaysFailingWandb:
    """Every artifact interaction raises (worst-case W&B outage)."""

    def Artifact(self, name, type):
        raise RuntimeError("wandb server unreachable")

    def log_artifact(self, artifact):
        raise RuntimeError("artifact upload failed")


def test_should_log_artifacts_follows_the_backend():
    assert should_log_artifacts("wandb", upload_artifacts=True) is True
    assert should_log_artifacts("wandb", upload_artifacts=False) is False
    assert should_log_artifacts("terminal", upload_artifacts=True) is False


def test_build_artifact_metadata_records_the_identifying_fields():
    metadata = build_artifact_metadata(
        kind="best",
        stage="stage-2-decay",
        step=123,
        tokens_seen=31456,
        best_val_loss=1.75,
        git_commit="deadbeef",
        local_path="runs/2026-08-23/12-00-00/best.pt",
        wandb_run_id="abc123",
    )

    assert metadata == {
        "kind": "best",
        "stage": "stage-2-decay",
        "step": 123,
        "tokens_seen": 31456,
        "best_val_loss": 1.75,
        "git_commit": "deadbeef",
        "local_path": "runs/2026-08-23/12-00-00/best.pt",
        "wandb_run_id": "abc123",
    }


def test_log_artifact_uploads_the_file_with_metadata(tmp_path):
    file = tmp_path / "best.pt"
    file.write_bytes(b"weights")
    wandb = FakeWandb()

    ok = log_artifact(
        wandb,
        file,
        "best-checkpoint",
        build_artifact_metadata(
            kind="best",
            stage="stage-2-decay",
            step=123,
            tokens_seen=31456,
            best_val_loss=1.75,
            git_commit="deadbeef",
            local_path=str(file),
            wandb_run_id="abc123",
        ),
    )

    assert ok is True
    assert len(wandb.logged) == 1
    artifact = wandb.logged[0]
    assert artifact.name == "best-checkpoint"
    assert artifact.type == "models"
    assert artifact.files == [str(file)]
    assert artifact.metadata["kind"] == "best"
    assert artifact.metadata["wandb_run_id"] == "abc123"
    assert artifact.metadata["step"] == 123


def test_artifact_failure_never_propagates():
    for fail_at in ("create", "log"):
        wandb = FakeWandb(fail_at=fail_at)
        ok = log_artifact(wandb, "runs/x/best.pt", "best-checkpoint", {"kind": "best"})
        assert ok is False


def test_artifact_failure_never_aborts_training_even_worst_case():
    # Must return (not raise) so a W&B outage cannot kill a long training run.
    assert log_artifact(AlwaysFailingWandb(), "p", "n", {}) is False


def test_terminal_adapter_performs_no_wandb_interaction(monkeypatch):
    """The terminal adapter holds no W&B reference and performs no W&B
    interaction: every W&B entry point on the real module is patched to
    record, and a full adapter session must record nothing."""
    recorded = []

    def _record(*args, **kwargs):
        recorded.append((args, kwargs))
        return None

    for name in ("init", "log", "finish", "Artifact", "log_artifact"):
        monkeypatch.setattr(wandb, name, _record)

    logger = TerminalLogger()
    assert not hasattr(logger, "_wandb")
    logger.init(
        RunInfo(
            name="exp",
            config={},
            saved_config=None,
            resumed_run_id=None,
            computed={},
            stage_marker={"training_stage": "stage-1-stable"},
            parameter_metrics={},
            step=0,
        )
    )
    logger.log_metrics({"train/loss": 1.0, "step": 5}, step=5)
    logger.log_sample("hello world", step=5, loss=0.5)
    logger.log_checkpoint("best", "best-checkpoint", "p.pt", {"kind": "best"})
    assert logger.run_url is None
    logger.finish()

    assert recorded == []


def test_wandb_adapter_logs_the_artifact_through_the_interface():
    fake = FakeWandb()
    logger = WandbLogger(wandb_module=fake, upload_artifacts=True)
    logger.log_checkpoint(
        "best",
        "best-checkpoint",
        "best.pt",
        build_artifact_metadata(
            kind="best",
            stage="stage-2-decay",
            step=38,
            tokens_seen=9728,
            best_val_loss=1.5,
            git_commit="deadbeef",
            local_path="best.pt",
            wandb_run_id="abc123",
        ),
    )

    assert len(fake.logged) == 1
    assert fake.logged[0].metadata["stage"] == "stage-2-decay"
    assert fake.logged[0].metadata["step"] == 38


def test_wandb_adapter_skips_artifact_uploads_by_default():
    fake = FakeWandb()
    logger = WandbLogger(wandb_module=fake)

    logger.log_checkpoint("best", "best-checkpoint", "best.pt", {"kind": "best"})

    assert fake.logged == []


def test_wandb_adapter_resumed_run_keeps_the_original_config_and_name():
    fake = FakeWandb()
    logger = WandbLogger(wandb_module=fake)
    saved_config = {"training": {"exp_name": "original"}}

    logger.init(
        RunInfo(
            name="renamed-on-relaunch",
            config={"training": {"exp_name": "renamed-on-relaunch"}},
            saved_config=saved_config,
            resumed_run_id="abc123",
            computed={"computed_total_steps": 38},
            stage_marker={"training_stage": "stage-2-decay"},
            parameter_metrics={
                "model/parameters_total": 1000,
                "model/parameters_trainable": 900,
                "model/parameters_non_trainable": 100,
                "model/parameter_memory_bytes": 4000,
                "model/parameter_memory_mb": 0.0038,
            },
            step=38,
        )
    )

    assert fake.init_calls == [
        {
            "project": "mini-llm-lab",
            "name": "original",
            "config": saved_config,
            "id": "abc123",
            "resume": "must",
        }
    ]
    # The run id surfaces through the adapter (F3: state persistence reads
    # this after tracker.start).
    assert logger.run_id == "abc123"


def test_first_checkpoint_payload_carries_the_wandb_run_id(tiny_cfg):
    """F3: after tracker.start, state.wandb_run_id is set from the
    logger's run, so the first checkpoint payload persists the id."""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    tiny_cfg.training.save_interval = 2
    fake = FakeWandb(run_id="abc123")
    trainer = build_trainer(tiny_cfg, logger=WandbLogger(wandb_module=fake))
    trainer.train()

    assert trainer.state.wandb_run_id == "abc123"
    ckpt = torch.load(
        Path(trainer.ckpt.save_dir) / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ckpt["wandb_run_id"] == "abc123"


def test_resume_reuses_the_saved_wandb_run_id(tiny_cfg):
    """F3: a checkpoint written with a run id resumes into the SAME W&B
    run: wandb.init is called with id=<saved id> and resume='must'."""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    tiny_cfg.training.save_interval = 2

    first = FakeWandb(run_id="abc123")
    trainer = build_trainer(tiny_cfg, logger=WandbLogger(wandb_module=first))
    trainer.train()
    ckpt = torch.load(
        Path(trainer.ckpt.save_dir) / "final_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ckpt["wandb_run_id"] == "abc123"

    second = FakeWandb(run_id="fresh-should-not-be-used")
    resumed = build_trainer(
        tiny_cfg, checkpoint=ckpt, logger=WandbLogger(wandb_module=second)
    )

    assert resumed.state.wandb_run_id == "abc123"
    assert second.init_calls == [
        {
            "project": "mini-llm-lab",
            "name": "tiny",
            "config": ckpt["cfg"],
            "id": "abc123",
            "resume": "must",
        }
    ]


def test_terminal_backend_keeps_wandb_run_id_none(tiny_cfg):
    """F3: terminal runs (no injected logger, no W&B at all) keep
    state.wandb_run_id = None and persist None in the payload."""
    tiny_cfg.training.train_tokens = 5 * TOKENS_PER_STEP
    trainer = build_trainer(tiny_cfg)  # tiny_cfg: log_backend=terminal
    trainer.train()

    assert trainer.state.wandb_run_id is None
    ckpt = torch.load(
        Path(trainer.ckpt.save_dir) / "final_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ckpt["wandb_run_id"] is None
