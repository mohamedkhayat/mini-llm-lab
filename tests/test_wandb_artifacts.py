class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []
        self.metadata = {}

    def add_file(self, path):
        self.files.append(path)


class FakeWandb:
    """Minimal stand-in for the wandb module: records artifact activity."""

    def __init__(self, fail_at="none"):
        self.fail_at = fail_at  # "none" | "create" | "log"
        self.logged = []

    def Artifact(self, name, type):
        if self.fail_at == "create":
            raise RuntimeError("wandb server unreachable")
        return FakeArtifact(name, type)

    def log_artifact(self, artifact):
        if self.fail_at == "log":
            raise RuntimeError("artifact upload failed")
        self.logged.append(artifact)


class AlwaysFailingWandb:
    """Every artifact interaction raises (worst-case W&B outage)."""

    def Artifact(self, name, type):
        raise RuntimeError("wandb server unreachable")

    def log_artifact(self, artifact):
        raise RuntimeError("artifact upload failed")


def test_should_log_artifacts_follows_the_backend():
    from training.trainer import should_log_artifacts

    assert should_log_artifacts("wandb") is True
    assert should_log_artifacts("terminal") is False


def test_build_artifact_metadata_records_the_identifying_fields():
    from training.trainer import build_artifact_metadata

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
    from training.trainer import build_artifact_metadata, log_artifact

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
    from training.trainer import log_artifact

    for fail_at in ("create", "log"):
        wandb = FakeWandb(fail_at=fail_at)
        ok = log_artifact(wandb, "runs/x/best.pt", "best-checkpoint", {"kind": "best"})
        assert ok is False


def test_artifact_failure_never_aborts_training_even_worst_case():
    from training.trainer import log_artifact

    # Must return (not raise) so a W&B outage cannot kill a long training run.
    assert log_artifact(AlwaysFailingWandb(), "p", "n", {}) is False


def _stub_trainer(log_backend):
    """Bare trainer stand-in (no __init__) with just the attributes the
    artifact path touches, so the backend gate can be exercised without a
    real run."""
    from types import SimpleNamespace

    return SimpleNamespace(
        log_backend=log_backend,
        stage_label="stage-2-decay",
        step=38,
        tokens_seen=9728,
        best_val_loss=1.5,
        _manifest_git_commit="deadbeef",
        wandb_run_id="abc123",
    )


def test_terminal_mode_performs_no_wandb_interaction(monkeypatch):
    from training import trainer as trainer_module

    recorded = []
    monkeypatch.setattr(
        trainer_module.wandb,
        "Artifact",
        lambda *a, **k: recorded.append(("Artifact", a, k)) or FakeArtifact("x", "y"),
    )
    monkeypatch.setattr(trainer_module.wandb, "log_artifact", recorded.append)

    trainer_module.Trainer._log_checkpoint_artifact(
        _stub_trainer("terminal"), "best", "best-checkpoint", "p.pt"
    )

    assert recorded == []  # terminal mode must not touch wandb at all


def test_wandb_mode_logs_the_artifact_through_the_trainer(monkeypatch):
    from training import trainer as trainer_module

    fake = FakeWandb()
    monkeypatch.setattr(trainer_module, "wandb", fake)

    trainer_module.Trainer._log_checkpoint_artifact(
        _stub_trainer("wandb"), "best", "best-checkpoint", "best.pt"
    )

    assert len(fake.logged) == 1
    assert fake.logged[0].metadata["stage"] == "stage-2-decay"
    assert fake.logged[0].metadata["step"] == 38