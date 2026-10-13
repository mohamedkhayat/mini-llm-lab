"""CheckpointManager: the single home of the save policy table (P4).

| kind     | files written               | manifest entries        | W&B artifact       |
|----------|-----------------------------|-------------------------|--------------------|
| latest   | ``latest.pt``               | latest                  | no                 |
| best     | ``best.pt``                 | best                    | yes (best-checkpoint) |
| periodic | ``step_N.pt``, ``latest.pt`` | periodic, latest        | no                 |
| final    | ``final_model.pt``, ``latest.pt`` | final, latest     | yes (final-model)  |

Single write per event: for ``periodic`` and ``final`` the payload is
serialized once to the primary file, then ``latest.pt`` is published by
hardlinking the primary's inode to a temp name and ``os.replace``-ing it
over ``latest.pt`` (falling back to a streamed copy when ``os.link``
fails — cross-device or a filesystem without hardlinks).

Every published file is written atomically (temp inode + ``os.replace``)
and is only ever replaced, never modified in place — so a crash mid-save
cannot truncate a resumable file, and a later ``latest.pt`` replace can
never corrupt a shared-inode primary.

``snapshot_fn() -> dict`` is supplied by the trainer (until T6.2 moves it
to assembly) and returns the full payload via
``build_checkpoint_payload`` using the RAW model's state_dict (spec 5.8
compile note) — the manager knows nothing about trainer, model or
optimizer. Manifest and artifact failures never raise.
"""

import os
import shutil
from pathlib import Path
from typing import Callable, NamedTuple

import torch

from training.checkpointing import atomic_torch_save
from training.run_manifest import fsync_dir


class RunIdentity(NamedTuple):
    """The run's identity, computed once in assembly (spec 5.4)."""

    run_name: str
    git_commit: str
    config_digest: str


class CheckpointManager:
    def __init__(
        self,
        save_dir,
        snapshot_fn,
        identity: RunIdentity,
        tracker,
        before_save: Callable[[], None] | None = None,
        stage_label: str | None = None,
        artifact_metadata: Callable | None = None,
    ):
        self._save_dir = Path(save_dir)
        self._snapshot_fn = snapshot_fn
        self._identity = identity
        self._tracker = tracker
        self._before_save = before_save
        self._stage_label = stage_label
        self._artifact_metadata = artifact_metadata

    @property
    def save_dir(self):
        """Where checkpoints are written (inspection / test seam)."""
        return self._save_dir

    def _record(self, state, kind: str, filename: str, final_eval=None) -> None:
        self._tracker.record_checkpoint(
            kind,
            filename,
            step=state.step,
            tokens_seen=state.tokens_seen,
            stage=self._stage_label,
            best_val_loss=state.best_val_loss,
            wandb_run_id=state.wandb_run_id,
            final_eval=final_eval,
        )

    def _artifact(self, kind: str, name: str, path: str) -> None:
        metadata = (
            self._artifact_metadata(kind, path)
            if self._artifact_metadata is not None
            else {"kind": kind}
        )
        self._tracker.log_checkpoint_artifact(kind, name, path, metadata)

    def save(self, kind: str, state, final_eval=None) -> None:
        """Persist one checkpoint event per the policy table row for
        ``kind``. ``state`` is the run's ``TrainState`` (its ``step`` names
        the periodic file; ``run_elapsed_seconds`` is stamped by the
        ``before_save`` hook first). ``final_eval`` (F7) is the end-of-run
        ``EvalResult``; for ``kind='final'`` it is recorded on the
        final_model.pt manifest entry."""
        if self._before_save is not None:
            self._before_save()
        payload = self._snapshot_fn()

        if kind == "latest":
            self._write_primary("latest.pt", payload)
            self._record(state, "latest", "latest.pt")
        elif kind == "best":
            self._write_primary("best.pt", payload)
            self._record(state, "best", "best.pt")
            self._artifact("best", "best-checkpoint", str(self._save_dir / "best.pt"))
        elif kind == "periodic":
            filename = f"step_{state.step}.pt"
            self._write_primary(filename, payload)
            self._publish_latest(filename)
            self._record(state, "periodic", filename)
            self._record(state, "latest", "latest.pt")
        elif kind == "final":
            self._write_primary("final_model.pt", payload)
            self._publish_latest("final_model.pt")
            self._record(state, "final", "final_model.pt", final_eval=final_eval)
            self._record(state, "latest", "latest.pt")
            self._artifact(
                "final", "final-model", str(self._save_dir / "final_model.pt")
            )
        else:
            raise ValueError(f"unknown checkpoint kind: {kind!r}")

    def _write_primary(self, filename: str, payload) -> None:
        """Serialize the payload exactly once to the primary file,
        atomically: temp inode + ``os.replace``, so a crash mid-write can
        never truncate a published file."""
        atomic_torch_save(payload, self._save_dir / filename)

    def _publish_latest(self, primary_name: str) -> None:
        """Publish ``latest.pt`` without re-serializing: hardlink the
        primary's inode to a temp name, then atomically rename over
        ``latest.pt``; fall back to a streamed copy if ``os.link`` fails.

        The temp is unlinked (missing_ok) BEFORE the link so a stale
        temp — possibly hardlinked to a primary — can never be opened
        with 'wb' by the copy fallback and truncate that primary."""
        primary = self._save_dir / primary_name
        latest = self._save_dir / "latest.pt"
        tmp = self._save_dir / f".latest.pt.tmp.{os.getpid()}"
        tmp.unlink(missing_ok=True)
        try:
            try:
                os.link(primary, tmp)
            except OSError:
                with open(primary, "rb") as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            tmp.replace(latest)
            fsync_dir(self._save_dir)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
