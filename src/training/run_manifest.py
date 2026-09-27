"""Per-run-dir manifest tying local checkpoint files to their W&B run.

Each run directory holds a ``run_manifest.json`` that is atomically updated on
every checkpoint save. The top level records the run identity (name, W&B run
id/URL, git commit, config digest) and ``checkpoints`` maps each checkpoint
file to its per-save stats (step, tokens seen, stage, best val loss,
timestamp) plus the W&B run it belongs to. Written in terminal mode as well —
it is a local-only lookup utility, not a tracking backend.
"""

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

MANIFEST_NAME = "run_manifest.json"


def config_digest(cfg) -> str:
    """Return a stable 16-char digest of a (resolved) config mapping.

    Key order does not matter, so two runs composed from the same settings
    digest identically even if Hydra materialized the keys differently.
    """
    encoded = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def git_commit() -> str | None:
    """Best-effort ``git rev-parse HEAD``; ``None`` when not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def utc_timestamp() -> str:
    """Current UTC time as an ISO-8601 string for manifest entries."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_entry(
    file,
    kind,
    step,
    tokens_seen,
    stage,
    best_val_loss,
    timestamp,
    wandb_run_id=None,
    wandb_url=None,
) -> dict:
    """Build one manifest entry for a checkpoint file.

    ``kind`` is one of ``latest`` / ``best`` / ``final`` / ``periodic``. A
    non-finite ``best_val_loss`` (no evaluation has improved yet) is recorded
    as ``None`` because it is not representable in strict JSON.
    """
    if best_val_loss is None or not math.isfinite(best_val_loss):
        best_val_loss = None
    return {
        "file": file,
        "kind": kind,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "stage": stage,
        "best_val_loss": best_val_loss,
        "timestamp": timestamp,
        "wandb_run_id": wandb_run_id,
        "wandb_url": wandb_url,
    }


def _load_existing_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(manifest, dict):
        return {}
    return manifest


def atomic_write_bytes(destination, data) -> None:
    """Write ``data`` to ``destination`` via a temp file + atomic rename.

    The file and its parent directory are fsynced, so a power loss cannot
    lose a completed write. Shared by the manifest writer and the torch
    checkpoint writer (``training.trainer.atomic_torch_save``).
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)

        # fsync the parent directory so a sudden power loss cannot lose the
        # rename even after the file contents have reached disk.
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so a crash never truncates the manifest."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, encoded)


def upsert_manifest(
    run_dir,
    *,
    run_name,
    wandb_run_id,
    wandb_url,
    git_commit,
    config_digest,
    entry,
) -> dict:
    """Record ``entry`` in the run manifest and return the updated manifest.

    Entries are keyed by checkpoint file name, so re-saved files (``latest.pt``
    on every periodic/stop save) refresh their entry instead of duplicating
    it. A missing or corrupt manifest is reinitialized rather than raising:
    the manifest must never take a training run down with it.
    """
    path = Path(run_dir) / MANIFEST_NAME
    manifest = _load_existing_manifest(path)
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict):
        checkpoints = {}
    checkpoints[entry["file"]] = entry

    manifest["run_name"] = run_name
    manifest["wandb_run_id"] = wandb_run_id
    manifest["wandb_url"] = wandb_url
    manifest["git_commit"] = git_commit
    manifest["config_digest"] = config_digest
    manifest["checkpoints"] = checkpoints
    manifest["updated_at"] = utc_timestamp()

    _atomic_write_json(path, manifest)
    return manifest
