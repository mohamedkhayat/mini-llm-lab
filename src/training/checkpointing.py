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
``steps_per_pass``, missing ``rng``, a non-memmap ``data_state``, or a
``wsd`` entry without a ``target_step`` key all raise on resume. Start a
fresh run instead (the single resumability policy lives in
``training.schedule``).

``restore_training_state`` expects the trainer's
``training.state.TrainState`` and restores the scalar counters and RNG
streams only; the memmap data cursor is restored separately via
``restore_data_state(loader, checkpoint['data_state'])``.
"""

import random
from pathlib import Path

import numpy as np
import torch

from training.run_manifest import atomic_write


def _launch_directory() -> Path:
    """Return the directory from which Hydra launched the job."""
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.cwd)
    except Exception:
        return Path.cwd()


def resume_requested(training_cfg) -> bool:
    """Whether the run was asked to resume. One shared predicate (F9):
    ``None``, ``''`` and ``False`` all mean not requested. Used by both
    ``build_trainer`` (refuse a fresh run when a resume was asked) and
    ``resolve_resume_path``."""
    configured = getattr(training_cfg, "resume_from", None)
    return configured not in (None, "", False)


def resolve_resume_path(training_cfg) -> Path | None:
    """Resolve an explicit checkpoint path or the newest ``latest.pt``."""
    if not resume_requested(training_cfg):
        return None

    configured_path = str(getattr(training_cfg, "resume_from", None))
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
    """Write a checkpoint atomically, streaming ``torch.save`` directly to
    the temp file (no in-memory copy of the full payload)."""
    atomic_write(path, lambda f: torch.save(payload, f))


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


def check_cursor_consistency(state, loader, accum_steps, phase) -> None:
    """Test-only invariant (never called from production code): at every
    checkpoint save the data cursor must equal the phase's start cursor
    plus ``(state.step - phase.start_step) * accum_steps`` — a stop or
    crash may never leave the cursor ahead of (or behind) the completed
    steps, or a resume would skip or duplicate data."""
    accum_steps = int(accum_steps)
    start_step = int(phase.start_step)
    start_cursor = start_step * accum_steps
    expected = start_cursor + (int(state.step) - start_step) * accum_steps
    actual = int(loader.state_dict()["current_step"])
    if actual != expected:
        raise AssertionError(
            f"Data cursor {actual} is inconsistent with step {state.step}: "
            f"expected {expected} (start cursor {start_cursor}, "
            f"{accum_steps} micro-batches per step). A discarded pull must "
            f"not advance the cursor."
        )


def restore_model(model, checkpoint) -> None:
    """Load plain or compiled-module state into the uncompiled model.

    strict=True: a missing or unexpected key surfaces as torch's native
    error instead of being silently tolerated.
    """
    state = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint does not contain a model state dictionary")

    normalized_state = {
        key.removeprefix("_orig_mod."): value for key, value in state.items()
    }
    model.load_state_dict(normalized_state, strict=True)


def restore_training_state(state, checkpoint: dict) -> None:
    """Restore the scalar counters and all RNG streams into ``state``.

    ``state`` is the trainer's ``training.state.TrainState``; the memmap
    data cursor is NOT restored here (call ``restore_data_state`` on the
    loader separately). A checkpoint without ``tokens_seen`` or RNG state
    is not resumable and raises.
    """
    state.step = int(checkpoint.get("step", 0))
    state.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    state.wandb_run_id = checkpoint.get("wandb_run_id")
    state.run_elapsed_seconds = float(checkpoint.get("run_elapsed_seconds", 0.0))

    saved_tokens = checkpoint.get("tokens_seen")
    if saved_tokens is None:
        raise ValueError(
            "The checkpoint does not carry a tokens_seen value; it is not "
            "resumable with this trainer. Start a fresh run instead."
        )
    state.tokens_seen = int(saved_tokens)

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
