"""T4.1: CheckpointManager — the single home of the save policy table."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from training.ckpt_manager import CheckpointManager, RunIdentity


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _RecordingTracker:
    def __init__(self):
        self.records = []      # (kind, filename, kwargs)
        self.artifacts = []    # (kind, name, metadata)

    def record_checkpoint(self, kind, filename, *, step, tokens_seen, stage,
                          best_val_loss, wandb_run_id):
        self.records.append((kind, filename, dict(
            step=step, tokens_seen=tokens_seen, stage=stage,
            best_val_loss=best_val_loss, wandb_run_id=wandb_run_id,
        )))

    def log_checkpoint_artifact(self, kind, name, path, metadata):
        self.artifacts.append((kind, name, metadata))


def _make_manager(tmp_path, tracker, state, before_save=None):
    identity = RunIdentity(run_name="run", git_commit="deadbeef",
                           config_digest="cfg123")
    snapshot = lambda: {"step": state.step, "blob": b"x" * 4096}  # noqa: E731
    return CheckpointManager(
        str(tmp_path), snapshot, identity, tracker, before_save=before_save
    )


def _state(step=0):
    from training.state import TrainState

    return TrainState(step=step)


def _files(tmp_path):
    return {p.name for p in tmp_path.iterdir() if p.suffix == ".pt"}


def test_ckpt_policy_table(tmp_path):
    """The four policy rows: files written, manifest entries, artifact calls."""
    tracker = _RecordingTracker()
    state = _state(step=7)
    mgr = _make_manager(tmp_path, tracker, state)

    mgr.save("latest", state)
    assert _files(tmp_path) == {"latest.pt"}
    assert [r[0:2] for r in tracker.records] == [("latest", "latest.pt")]
    assert tracker.artifacts == []

    mgr.save("best", state)
    assert _files(tmp_path) == {"latest.pt", "best.pt"}
    assert tracker.records[-1][0:2] == ("best", "best.pt")
    assert [a[0:2] for a in tracker.artifacts] == [("best", "best-checkpoint")]

    mgr.save("periodic", state)
    assert _files(tmp_path) == {"latest.pt", "best.pt", "step_7.pt"}
    assert [r[0:2] for r in tracker.records[-2:]] == [
        ("periodic", "step_7.pt"),
        ("latest", "latest.pt"),
    ]
    assert [a[0:2] for a in tracker.artifacts] == [("best", "best-checkpoint")]  # no new artifact

    mgr.save("final", state)
    assert _files(tmp_path) == {
        "latest.pt", "best.pt", "step_7.pt", "final_model.pt",
    }
    assert [r[0:2] for r in tracker.records[-2:]] == [
        ("final", "final_model.pt"), ("latest", "latest.pt"),
    ]
    assert tracker.artifacts[-1][0:2] == ("final", "final-model")

    # The payloads round-trip through torch.load.
    for name in ("latest.pt", "best.pt", "step_7.pt", "final_model.pt"):
        payload = torch.load(tmp_path / name, weights_only=True)
        assert payload == {"step": 7, "blob": b"x" * 4096}, name


def test_latest_publish_shares_inode_with_primary(tmp_path):
    """periodic/final publish latest.pt by hardlink (single write per
    event): same inode where the FS supports hardlinks, else equal bytes."""
    tracker = _RecordingTracker()
    state = _state(step=3)
    mgr = _make_manager(tmp_path, tracker, state)

    mgr.save("final", state)
    final_stat = os.stat(tmp_path / "final_model.pt")
    latest_stat = os.stat(tmp_path / "latest.pt")
    if final_stat.st_nlink >= 2 and final_stat.st_ino == latest_stat.st_ino:
        assert final_stat.st_dev == latest_stat.st_dev  # hardlink
    else:  # fallback (no hardlinks): identical content
        assert (tmp_path / "final_model.pt").read_bytes() == (
            tmp_path / "latest.pt"
        ).read_bytes()


def test_final_writes_weights_once(tmp_path, monkeypatch):
    """Exactly one torch.save per final event (P4): latest.pt must not be a
    second serialization."""
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return _real_save(*args, **kwargs)

    _real_save = torch.save
    monkeypatch.setattr(torch, "save", spy)

    tracker = _RecordingTracker()
    state = _state(step=1)
    mgr = _make_manager(tmp_path, tracker, state)
    mgr.save("final", state)

    assert len(calls) == 1
    assert (tmp_path / "final_model.pt").is_file()
    assert (tmp_path / "latest.pt").is_file()


def test_before_save_runs_before_the_snapshot(tmp_path):
    """The before_save hook (run-elapsed stamp) fires before the payload is
    captured, on every save."""
    order = []
    state = _state()
    tracker = _RecordingTracker()

    def before_save():
        state.run_elapsed_seconds = 123.0
        order.append("before_save")

    identity = RunIdentity(run_name="r", git_commit="g", config_digest="c")
    mgr = CheckpointManager(
        str(tmp_path),
        lambda: {"t": state.run_elapsed_seconds},
        identity,
        tracker,
        before_save=before_save,
    )
    mgr.save("latest", state)

    order.append("snapshot")
    assert order.index("before_save") < order.index("snapshot")
    payload = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert payload == {"t": 123.0}


def test_save_latest_does_not_mutate_a_previous_primary(tmp_path):
    """A published file is never modified in place: republishing latest.pt
    (which after a periodic save may share an inode with the primary)
    must leave the primary's bytes untouched."""
    tracker = _RecordingTracker()
    state = _state(step=7)
    mgr = _make_manager(tmp_path, tracker, state)
    mgr.save("periodic", state)
    before = _sha(tmp_path / "step_7.pt")

    state.step = 9
    mgr.save("latest", state)

    assert _sha(tmp_path / "step_7.pt") == before
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 9


def test_failed_save_leaves_previous_latest_intact(tmp_path, monkeypatch):
    """A crash mid-write must not truncate latest.pt: the previous file
    stays readable and no temp files remain."""
    tracker = _RecordingTracker()
    state = _state(step=1)
    mgr = _make_manager(tmp_path, tracker, state)
    mgr.save("latest", state)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid-save")

    monkeypatch.setattr(torch, "save", boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        mgr.save("latest", state)

    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 1
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_stale_publish_temp_is_removed_and_primaries_untouched(tmp_path):
    """A stale ``.latest.pt.tmp.<pid>`` hardlinked to a previous primary
    must be unlinked before the publish, or the copy fallback would open
    it with 'wb' and truncate the primary."""
    tracker = _RecordingTracker()
    state = _state(step=3)
    mgr = _make_manager(tmp_path, tracker, state)
    mgr.save("periodic", state)
    primary_bytes = (tmp_path / "step_3.pt").read_bytes()

    stale = tmp_path / f".latest.pt.tmp.{os.getpid()}"
    os.link(tmp_path / "step_3.pt", stale)

    state.step = 6
    mgr.save("periodic", state)

    assert not stale.exists()
    assert (tmp_path / "step_3.pt").read_bytes() == primary_bytes
    assert (tmp_path / "step_6.pt").read_bytes() == (
        tmp_path / "latest.pt"
    ).read_bytes()


def test_copy_fallback_publishes_equal_content_with_a_different_inode(
    tmp_path, monkeypatch
):
    """With hardlinks unavailable the copy fallback still publishes
    equal-content latest.pt — on a fresh inode, never on the primary's."""

    def boom(*args, **kwargs):
        raise OSError("no hardlinks on this filesystem")

    monkeypatch.setattr(os, "link", boom)

    tracker = _RecordingTracker()
    state = _state(step=4)
    mgr = _make_manager(tmp_path, tracker, state)
    mgr.save("periodic", state)

    latest = tmp_path / "latest.pt"
    primary = tmp_path / "step_4.pt"
    assert latest.read_bytes() == primary.read_bytes()
    assert os.stat(latest).st_ino != os.stat(primary).st_ino
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_manifest_written_through_upsert(tmp_path):
    """Manifest failures never raise and the entry lands in
    run_manifest.json (behavior preserved from the tracker path is covered
    end-to-end by the seam tests; here: the manager's own manifest write is
    delegated to the tracker — assert the delegation only)."""
    tracker = _RecordingTracker()
    state = _state(step=2)
    mgr = _make_manager(tmp_path, tracker, state)
    mgr.save("periodic", state)
    assert ("periodic", "step_2.pt") in [r[0:2] for r in tracker.records]
    assert ("latest", "latest.pt") in [r[0:2] for r in tracker.records]
