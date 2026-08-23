import json
from pathlib import Path

from training.run_manifest import (
    MANIFEST_NAME,
    config_digest,
    make_entry,
    upsert_manifest,
)


def _upsert(**overrides):
    kwargs = dict(
        run_name="exp",
        wandb_run_id="abc123",
        wandb_url="https://wandb.local/r/abc123",
        git_commit="deadbeef",
        config_digest="cfg123",
        entry=make_entry(
            file="latest.pt",
            kind="latest",
            step=30,
            tokens_seen=7680,
            stage="stage-1-stable",
            best_val_loss=2.5,
            timestamp="2026-08-23T12:00:00Z",
            wandb_run_id="abc123",
            wandb_url="https://wandb.local/r/abc123",
        ),
    )
    kwargs.update(overrides)
    return kwargs


def test_config_digest_is_deterministic_and_sensitive():
    cfg_a = {"lr": 5e-4, "seed": 42}
    cfg_b = {"seed": 42, "lr": 5e-4}
    cfg_c = {"lr": 1e-4, "seed": 42}

    digest = config_digest(cfg_a)
    assert digest == config_digest(cfg_b)  # key order must not matter
    assert digest != config_digest(cfg_c)
    assert len(digest) == 16


def test_make_entry_records_the_spec_fields():
    entry = make_entry(
        file="best.pt",
        kind="best",
        step=40,
        tokens_seen=10240,
        stage="stage-2-decay",
        best_val_loss=2.25,
        timestamp="2026-08-23T12:05:00Z",
        wandb_run_id="abc123",
        wandb_url="https://wandb.local/r/abc123",
    )

    assert entry == {
        "file": "best.pt",
        "kind": "best",
        "step": 40,
        "tokens_seen": 10240,
        "stage": "stage-2-decay",
        "best_val_loss": 2.25,
        "timestamp": "2026-08-23T12:05:00Z",
        "wandb_run_id": "abc123",
        "wandb_url": "https://wandb.local/r/abc123",
    }


def test_make_entry_normalizes_an_infinite_val_loss():
    # No evaluation has improved yet: best_val_loss is +inf, which is not
    # representable in strict JSON.
    entry = make_entry(
        file="latest.pt",
        kind="latest",
        step=5,
        tokens_seen=1280,
        stage="stage-1-stable",
        best_val_loss=float("inf"),
        timestamp="2026-08-23T12:00:05Z",
    )

    assert entry["best_val_loss"] is None


def test_upsert_creates_a_manifest_with_top_level_run_fields(tmp_path):
    manifest = upsert_manifest(str(tmp_path), **_upsert())

    path = tmp_path / MANIFEST_NAME
    assert path.is_file()
    on_disk = json.loads(path.read_text())
    assert on_disk == manifest
    assert on_disk["run_name"] == "exp"
    assert on_disk["wandb_run_id"] == "abc123"
    assert on_disk["wandb_url"] == "https://wandb.local/r/abc123"
    assert on_disk["git_commit"] == "deadbeef"
    assert on_disk["config_digest"] == "cfg123"
    assert "latest.pt" in on_disk["checkpoints"]
    assert on_disk["checkpoints"]["latest.pt"]["step"] == 30


def test_upsert_replaces_the_same_file_and_keeps_others(tmp_path):
    upsert_manifest(str(tmp_path), **_upsert())
    upsert_manifest(
        str(tmp_path),
        **_upsert(
            entry=make_entry(
                file="best.pt",
                kind="best",
                step=40,
                tokens_seen=10240,
                stage="stage-2-decay",
                best_val_loss=2.0,
                timestamp="2026-08-23T12:05:00Z",
            )
        ),
    )
    # The latest.pt entry is refreshed with new stats, best.pt is added.
    upsert_manifest(
        str(tmp_path),
        **_upsert(
            entry=make_entry(
                file="latest.pt",
                kind="latest",
                step=50,
                tokens_seen=12800,
                stage="stage-2-decay",
                best_val_loss=2.0,
                timestamp="2026-08-23T12:10:00Z",
            )
        ),
    )

    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert on_disk["checkpoints"]["latest.pt"]["step"] == 50
    assert on_disk["checkpoints"]["best.pt"]["step"] == 40
    assert set(on_disk["checkpoints"]) == {"latest.pt", "best.pt"}


def test_upsert_recovers_from_a_corrupt_manifest(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text("{not json")

    manifest = upsert_manifest(str(tmp_path), **_upsert())

    assert manifest["checkpoints"]["latest.pt"]["step"] == 30


def test_upsert_is_atomic(tmp_path):
    upsert_manifest(str(tmp_path), **_upsert())

    leftovers = [p for p in tmp_path.iterdir() if p.name != MANIFEST_NAME]
    assert leftovers == []