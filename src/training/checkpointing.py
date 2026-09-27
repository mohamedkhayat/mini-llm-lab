"""Checkpointing: everything that saves, loads, resolves, or validates a restart.

One seam for restarts. The trainer gathers its live run state and hands it
across the functions here; nothing else knows how a checkpoint is built or
restored.

Checkpoint payload format (version 3; older versions remain readable) — the key set,
assembled by ``build_checkpoint_payload``:

    checkpoint_version   int, currently 3 (older versions are readable)
    model                model state dict
    model_cfg            resolved model config (plain dict)
    optimizer            optimizer state dict
    scheduler            LR scheduler state dict
    step                 completed optimizer steps
    best_val_loss        best validation loss seen (inf until the first eval)
    cursor               {"epoch", "batch_in_epoch"} — the next batch to consume
    tokens_seen          training tokens consumed so far
    run_elapsed_seconds  wall-clock seconds accumulated before this save
    steps_per_epoch      batches per data pass for the training loader
    tokens_per_step      tokens per optimizer batch (batch_size * seq_len)
    total_steps          the run's total-step budget at save time
    wandb_run_id         W&B run id (None in terminal mode)
    rng                  all RNG streams: python, numpy, torch, cuda, dataloader
    cfg                  the full resolved config (plain dict)
    wsd                  the WSD state dict (format owned by ``training.schedule``)
    data_state            optional native streaming/DataLoader state

Legacy checkpoints may omit ``cursor``, ``tokens_seen``, ``rng``,
``scheduler``, ``wsd``, or ``total_steps`` / ``steps_per_epoch``; resume
tolerates the missing keys (with warnings where an exact continuation is no
longer possible).

``restore_training_state`` expects a :class:`RunState` bundle assembled by
the trainer (its mutable counters plus the live loaders, so the RNG restore
can also reset the dataloader generators and the sampler's epoch).
"""

import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.run_manifest import atomic_write_bytes


@dataclass
class RunState:
    """The trainer's mutable run state, handed across the restore seam.

    Scalar counters are read back from the bundle after
    ``restore_training_state``; the loader references are used to reset the
    dataloader RNG generators and the sampler's epoch.
    """

    step: int
    best_val_loss: float
    wandb_run_id: str | None
    run_elapsed_seconds: float
    epoch: int
    batch_in_epoch: int
    tokens_seen: int
    steps_per_epoch: int
    tokens_per_step: int
    train_loader: object = None
    val_loader: object = None


def _launch_directory() -> Path:
    """Return the directory from which Hydra launched the job."""
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.cwd)
    except Exception:
        return Path.cwd()


def resolve_resume_path(training_cfg) -> Path | None:
    """Resolve an explicit checkpoint path or the newest ``latest.pt``."""
    configured_path = getattr(training_cfg, "resume_from", None)
    if configured_path in (None, "", False):
        return None

    configured_path = str(configured_path)
    root = _launch_directory()
    if configured_path.lower() in {"latest", "auto"}:
        candidates = list(root.rglob("latest.pt"))
        if not candidates:
            raise FileNotFoundError(f"No latest.pt checkpoint found below {root}")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    path = Path(configured_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint on CPU so it can be restored before device setup."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the ``weights_only`` keyword.
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Write a checkpoint atomically and make the directory entry durable."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def build_checkpoint_payload(
    model_state,
    model_cfg,
    optimizer_state,
    scheduler_state,
    step,
    best_val_loss,
    cursor,
    tokens_seen,
    run_elapsed_seconds,
    steps_per_epoch,
    tokens_per_step,
    total_steps,
    wandb_run_id,
    rng_state,
    cfg_container,
    wsd_state,
    data_state=None,
):
    """Assemble the restart-complete checkpoint payload (plain dict).

    The live WSD state (``wsd``) is persisted alongside the scheduler so a
    crash mid-decay resumes from the original trigger step; legacy
    checkpoints without the key are tolerated on resume.
    """
    return {
        "checkpoint_version": 3,
        "model": model_state,
        "model_cfg": model_cfg,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "step": step,
        "best_val_loss": best_val_loss,
        "cursor": cursor,
        "tokens_seen": tokens_seen,
        "run_elapsed_seconds": run_elapsed_seconds,
        "steps_per_epoch": steps_per_epoch,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "wandb_run_id": wandb_run_id,
        "rng": rng_state,
        "cfg": cfg_container,
        "wsd": wsd_state,
        "data_state": data_state,
    }


def capture_data_state(loader):
    """Capture only the state owned by a stateful streaming training loader.

    Ordinary map-style loaders already resume from the deterministic
    ``epoch``/``batch_in_epoch`` cursor. The streaming path is single-
    process, so the loader's state is the native dataset state. Keeping this
    seam here avoids teaching the trainer about the implementation.
    """
    if not getattr(loader, "stream_stateful", False):
        return None

    dataset = getattr(loader, "checkpoint_dataset", None)
    if dataset is None or not callable(getattr(dataset, "state_dict", None)):
        raise RuntimeError("Streaming loader has no checkpointable dataset state")
    return {"kind": "dataset", "state": dataset.state_dict()}


def restore_data_state(loader, data_state) -> None:
    """Restore a streaming loader's native state before iteration starts."""
    if not data_state:
        if getattr(loader, "stream_stateful", False):
            print(
                "Warning: checkpoint has no streaming data state; the model "
                "and optimizer resume at the saved step, but the HF stream "
                "restarts from the top of the corpus. This is a legacy "
                "checkpoint without native stream state and is not a "
                "bit-for-bit resume."
            )
        return
    kind = data_state.get("kind")
    if kind == "stateful_dataloader":
        raise ValueError(
            "This checkpoint saved forked-worker stream state "
            "(StatefulDataLoader); the streaming path is now single-process "
            "and cannot resume it. Start a fresh run or resume a "
            "single-process checkpoint."
        )
    if kind == "dataset":
        dataset = getattr(loader, "checkpoint_dataset", None)
        if dataset is None or not callable(getattr(dataset, "load_state_dict", None)):
            raise RuntimeError("Streaming loader cannot restore dataset state")
        dataset.load_state_dict(data_state["state"])
        return
    raise ValueError(f"Unsupported streaming data-state kind: {kind!r}")


def check_resume_consistency(
    saved_total_steps,
    total_steps,
    saved_steps_per_epoch,
    steps_per_epoch,
    decay_run,
    continuation_run=False,
):
    """Validate resume compatibility between a checkpoint and the new config.

    Decay runs are exempt from the total-steps check: stage 2 re-derives its
    budget from the trigger step, so the configured budget may differ from
    stage 1's. Stable continuation runs are also exempt because their new
    total is ``saved_step + additional_steps``. The steps-per-epoch check
    always applies because the data settings must remain compatible.
    """
    if (
        saved_total_steps is not None
        and int(saved_total_steps) != int(total_steps)
        and not decay_run
        and not continuation_run
    ):
        raise ValueError(
            "The resume config produces a different total step budget "
            f"({int(total_steps)}) than the checkpoint ({int(saved_total_steps)}). "
            "Resume with the original training budget for an exact continuation."
        )
    if saved_steps_per_epoch is not None and int(saved_steps_per_epoch) != int(
        steps_per_epoch
    ):
        raise ValueError(
            "The resumed dataloader has a different number of batches per "
            "data pass. Keep the original data, seq_len, stride, batch_size, "
            "and drop_last settings for an exact continuation."
        )


def restore_model(model, checkpoint) -> None:
    """Load plain or compiled-module state into the uncompiled model."""
    state = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint does not contain a model state dictionary")

    normalized_state = {
        key.removeprefix("_orig_mod."): value for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(normalized_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint model does not match the configured architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )


def restore_training_state(state: RunState, checkpoint: dict) -> None:
    """Restore counters, cursor, W&B identity, and all RNG streams.

    Mutates ``state`` in place: the trainer reads the bundle back after the
    call. Legacy checkpoints without an ``rng`` entry still restore the model
    and optimizer but are not bit-for-bit (warning printed).
    """
    state.step = int(checkpoint.get("step", 0))
    state.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    state.wandb_run_id = checkpoint.get("wandb_run_id")
    state.run_elapsed_seconds = float(checkpoint.get("run_elapsed_seconds", 0.0))

    cursor = checkpoint.get("cursor")
    if cursor is None:
        # Old checkpoints can still be loaded, but they do not contain
        # enough information for an exact continuation.
        state.epoch, state.batch_in_epoch = divmod(state.step, state.steps_per_epoch)
    else:
        state.epoch = int(cursor.get("epoch", 0))
        state.batch_in_epoch = int(cursor.get("batch_in_epoch", 0))

    saved_tokens = checkpoint.get("tokens_seen")
    state.tokens_seen = (
        int(saved_tokens)
        if saved_tokens is not None
        else state.step * state.tokens_per_step
    )

    restore_data_state(state.train_loader, checkpoint.get("data_state"))

    rng_state = checkpoint.get("rng")
    if rng_state is None:
        print(
            "Warning: checkpoint has no RNG/data-cursor state; the model and "
            "optimizer will resume, but this legacy checkpoint is not bit-for-bit."
        )
    else:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        dataloader_rng = rng_state.get("dataloader", {})
        train_generator = getattr(state.train_loader, "generator", None)
        val_generator = getattr(state.val_loader, "generator", None)
        if train_generator is not None and dataloader_rng.get("train") is not None:
            train_generator.set_state(dataloader_rng["train"])
        if val_generator is not None and dataloader_rng.get("val") is not None:
            val_generator.set_state(dataloader_rng["val"])

    sampler = getattr(state.train_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(state.epoch)
