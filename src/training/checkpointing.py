"""Checkpointing: everything that saves, loads, resolves, or validates a restart.

One seam for restarts. The trainer gathers its live run state and hands it
across the functions here; nothing else knows how a checkpoint is built or
restored.

Checkpoint payload format (version 4) — the key set, assembled by
``build_checkpoint_payload``:

    checkpoint_version   int, currently 4
    model                model state dict
    model_cfg            resolved model config (plain dict)
    optimizer            optimizer state dict
    scheduler            LR scheduler state dict
    step                 completed optimizer steps (the data position within
                         the current pass is carried by ``data_state``)
    best_val_loss        best validation loss seen (inf until the first eval)
    tokens_seen          training tokens consumed so far
    run_elapsed_seconds  wall-clock seconds accumulated before this save
    steps_per_pass       optimizer steps that fit in the phase's train-loader
                         cap (the physical cache capacity is rechecked on
                         additive phase resumes)
    tokens_per_step      tokens per optimizer batch
                         (batch_size * seq_len * accum_steps)
    total_steps          the run's total-step budget at save time
    wandb_run_id         W&B run id (None in terminal mode)
    rng                  all RNG streams: python, numpy, torch, cuda
    cfg                  the full resolved config (plain dict)
    wsd                  the WSD state dict (format owned by ``training.schedule``)
    data_state            the memmap data cursor {"kind": "memmap",
                          "state": {"current_step": <batch index in the
                          current pass>}} (None in terminal-mode stubs)

Checkpoints from older trainer versions are not resumable: a missing
``steps_per_pass``, missing ``rng``, or a non-memmap ``data_state`` raises on
resume. Start a fresh run instead.

``restore_training_state`` expects a :class:`RunState` bundle assembled by
the trainer (its mutable counters plus the live training loader, so the data
cursor can be restored into it).
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
    ``restore_training_state``; the loader reference is used to restore the
    memmap data cursor.
    """

    step: int
    best_val_loss: float
    wandb_run_id: str | None
    run_elapsed_seconds: float
    tokens_seen: int
    tokens_per_step: int
    train_loader: object = None


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
    tokens_seen,
    run_elapsed_seconds,
    steps_per_pass,
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
        "checkpoint_version": 4,
        "model": model_state,
        "model_cfg": model_cfg,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "step": step,
        "best_val_loss": best_val_loss,
        "tokens_seen": tokens_seen,
        "run_elapsed_seconds": run_elapsed_seconds,
        "steps_per_pass": steps_per_pass,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "wandb_run_id": wandb_run_id,
        "rng": rng_state,
        "cfg": cfg_container,
        "wsd": wsd_state,
        "data_state": data_state,
    }


def capture_data_state(loader):
    """Capture the training loader's data cursor (the memmap scan position).

    The memmap dataloader owns a single resume cursor: the next batch index
    in the file (one run is at most one pass over it). Keeping this seam here
    avoids teaching the trainer about the loader's state format.
    """
    state_fn = getattr(loader, "state_dict", None)
    if not callable(state_fn):
        raise RuntimeError("Training loader has no checkpointable data state")
    return {"kind": "memmap", "state": state_fn()}


def restore_data_state(loader, data_state) -> None:
    """Restore the training loader's data cursor before iteration starts.

    The checkpoint must carry the memmap cursor captured by
    ``capture_data_state``; checkpoints from the old trainers are not
    resumable.
    """
    if (
        not isinstance(data_state, dict)
        or data_state.get("kind") != "memmap"
        or not isinstance(data_state.get("state"), dict)
    ):
        raise ValueError(
            "The checkpoint does not carry a memmap data cursor; it is not "
            "resumable with this trainer. Start a fresh run instead."
        )
    load_fn = getattr(loader, "load_state_dict", None)
    if not callable(load_fn):
        raise RuntimeError("Training loader cannot restore data state")
    load_fn(data_state["state"])


def check_resume_consistency(
    saved_total_steps,
    total_steps,
    saved_steps_per_pass,
    steps_per_pass,
    saved_tokens_per_step,
    tokens_per_step,
    decay_run,
    continuation_run=False,
    allow_capacity_growth=False,
    available_steps_per_pass=None,
):
    """Validate resume compatibility between a checkpoint and the new config.

    Decay runs and stable continuations are exempt from the total-steps check
    because their endpoint includes an explicit additional budget. Their data
    capacity may grow after deterministic cache reprocessing, but it may not
    shrink below the checkpoint's capacity. A phase-specific loader cap may be
    smaller than the saved capacity when the underlying cache is still large
    enough; ``available_steps_per_pass`` carries that physical capacity. The
    tokens-per-step check always applies.
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
    if saved_steps_per_pass is None:
        raise ValueError(
            "The checkpoint does not carry a steps_per_pass value; it is "
            "not resumable with this trainer. Start a fresh run instead."
        )
    if int(saved_steps_per_pass) != int(steps_per_pass):
        phase_cap_is_intentional = (
            available_steps_per_pass is not None
            and int(available_steps_per_pass) >= int(saved_steps_per_pass)
        )
        if not (
            allow_capacity_growth
            and (
                int(steps_per_pass) > int(saved_steps_per_pass)
                or phase_cap_is_intentional
            )
        ):
            raise ValueError(
                "The resumed dataloader has a different number of batches per "
                "data pass. Keep the original data size, seq_len, and batch_size "
                "settings for an exact continuation."
            )
    if saved_tokens_per_step is None:
        raise ValueError(
            "The checkpoint does not carry a tokens_per_step value; it is "
            "not resumable with this trainer. Start a fresh run instead."
        )
    if int(saved_tokens_per_step) != int(tokens_per_step):
        raise ValueError(
            "The resume config produces a different tokens-per-step "
            "(batch_size x seq_len x accum_steps) than the checkpoint. Keep "
            "the original batch, sequence, and accumulation settings for an "
            "exact continuation."
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
    """Restore counters, W&B identity, all RNG streams, and the data cursor.

    Mutates ``state`` in place: the trainer reads the bundle back after the
    call. The memmap data cursor is restored into the training loader; a
    checkpoint without a memmap cursor or RNG state is not resumable and
    raises.
    """
    state.step = int(checkpoint.get("step", 0))
    state.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    state.wandb_run_id = checkpoint.get("wandb_run_id")
    state.run_elapsed_seconds = float(checkpoint.get("run_elapsed_seconds", 0.0))

    saved_tokens = checkpoint.get("tokens_seen")
    state.tokens_seen = (
        int(saved_tokens)
        if saved_tokens is not None
        else state.step * state.tokens_per_step
    )

    restore_data_state(state.train_loader, checkpoint.get("data_state"))

    rng_state = checkpoint.get("rng")
    if rng_state is None:
        raise ValueError(
            "The checkpoint does not carry RNG state; it is not resumable "
            "with this trainer. Start a fresh run instead."
        )
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
