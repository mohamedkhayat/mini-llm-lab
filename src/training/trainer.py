import io
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_

from data.dataloader import create_dataloaders
from data.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text
from models.gpt import GptModel
from training.run_manifest import (
    atomic_write_bytes,
    config_digest,
    git_commit,
    make_entry,
    upsert_manifest,
    utc_timestamp,
)


def format_count(value):
    """Format a large count using K/M/B suffixes for console output."""
    value = float(value)
    absolute_value = abs(value)
    if absolute_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def get_model_parameter_metrics(model):
    """Return unique parameter counts and storage size for a model.

    ``named_parameters`` removes duplicate references, so tied embeddings are
    counted once rather than inflating the reported model size.
    """
    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(
        parameter.numel() for _, parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for _, parameter in parameters
    )
    return {
        "model/parameters_total": total,
        "model/parameters_trainable": trainable,
        "model/parameters_non_trainable": total - trainable,
        "model/parameter_memory_bytes": parameter_bytes,
        "model/parameter_memory_mb": parameter_bytes / 1024**2,
    }


def resolve_log_backend(training_cfg, wandb_mode=None):
    """Resolve whether metrics should be sent to W&B or the terminal."""
    configured_backend = getattr(training_cfg, "log_backend", "wandb")
    if configured_backend is None:
        configured_backend = "wandb"

    aliases = {"term": "terminal", "console": "terminal"}
    normalized_backend = str(configured_backend).strip().lower()
    log_backend = aliases.get(normalized_backend, normalized_backend)
    if log_backend not in {"wandb", "terminal"}:
        raise ValueError(
            "training.log_backend must be either 'wandb' or 'terminal'; "
            f"got {configured_backend!r}"
        )

    # Keep the existing WANDB_MODE=disabled escape hatch useful even when the
    # config still has its default backend.
    selected_wandb_mode = (
        os.environ.get("WANDB_MODE", "") if wandb_mode is None else wandb_mode
    )
    if str(selected_wandb_mode).strip().lower() == "disabled":
        return "terminal"
    return log_backend


def resolve_total_steps(training_cfg, steps_per_epoch, tokens_per_step):
    """Resolve the training budget, preferring steps/tokens over epochs.

    ``epochs`` remains as a backwards-compatible fallback. Token budgets are
    rounded down to complete optimizer batches so training never exceeds the
    requested budget.
    """
    max_steps = getattr(training_cfg, "max_steps", None)
    max_tokens = getattr(training_cfg, "max_tokens", None)

    if max_steps is not None and max_tokens is not None:
        raise ValueError("Set only one of training.max_steps and training.max_tokens.")

    if max_steps is not None:
        total_steps = int(max_steps)
        budget_name = "max_steps"
    elif max_tokens is not None:
        requested_tokens = int(max_tokens)
        total_steps = requested_tokens // int(tokens_per_step)
        budget_name = "max_tokens"
    else:
        epochs = int(getattr(training_cfg, "epochs", 1))
        total_steps = epochs * int(steps_per_epoch)
        budget_name = "epochs"

    if total_steps <= 0:
        raise ValueError(
            "Training budget must produce at least one optimizer step; "
            "increase training.max_tokens/max_steps or training.epochs."
        )

    return total_steps, budget_name


def resolve_decay_budget(steps_done, decay_fraction):
    """Derive the WSD decay budget from where stage 1 stopped.

    The decay occupies the final ``decay_fraction`` of the whole run, so for a
    trigger at step ``S`` the decay length is ``D = round(S * f / (1 - f))``
    and the total budget is ``S + D`` (rule of three). ``f`` must be in
    ``(0, 1)``; a degenerate ``S = 0`` still runs at least one decay step.
    """
    steps_done = int(steps_done)
    decay_fraction = float(decay_fraction)

    if steps_done < 0:
        raise ValueError(
            "steps_done must be >= 0 (the trigger step of the decay run); "
            f"got {steps_done}"
        )
    if not 0.0 < decay_fraction < 1.0:
        raise ValueError(
            "training.lr_decay_fraction must be in (0, 1): the decay is the "
            f"final fraction of the run; got {decay_fraction}"
        )

    decay_steps = max(1, round(steps_done * decay_fraction / (1.0 - decay_fraction)))
    return steps_done + decay_steps, decay_steps


def resolve_wsd_state(saved_wsd, start_decay, resume_step):
    """Resolve the live WSD state for a (possibly resumed) run.

    A valid triggered state saved in a checkpoint takes precedence over a
    fresh ``start_decay`` flag, so a crash mid-decay resumes from the
    original trigger step instead of restarting the decay. Legacy
    checkpoints without a ``wsd`` entry (or with an untriggered one) fall
    back to the flag: ``start_decay`` then triggers the decay at the
    resume step.
    """
    saved = saved_wsd if isinstance(saved_wsd, dict) else {}
    saved_step = saved.get("step")
    if saved.get("triggered") is True and isinstance(saved_step, int) and saved_step >= 0:
        return {"triggered": True, "step": saved_step}
    if start_decay:
        return {"triggered": True, "step": int(resume_step)}
    return {"triggered": False, "step": None}


def check_resume_consistency(
    saved_total_steps, total_steps, saved_steps_per_epoch, steps_per_epoch, decay_run
):
    """Validate resume compatibility between a checkpoint and the new config.

    Decay runs are exempt from the total-steps check: stage 2 re-derives its
    budget from the trigger step, so the configured budget may differ from
    stage 1's. The steps-per-epoch check always applies because stage 2 must
    keep the original data settings.
    """
    if (
        saved_total_steps is not None
        and int(saved_total_steps) != int(total_steps)
        and not decay_run
    ):
        raise ValueError(
            "The resume config produces a different total step budget "
            f"({int(total_steps)}) than the checkpoint ({int(saved_total_steps)}). "
            "Resume with the original training budget for an exact continuation."
        )
    if saved_steps_per_epoch is not None and int(saved_steps_per_epoch) != int(steps_per_epoch):
        raise ValueError(
            "The resumed dataloader has a different number of batches per "
            "data pass. Keep the original data, seq_len, stride, batch_size, "
            "and drop_last settings for an exact continuation."
        )


def stage_label(resumed, decay_triggered):
    """Return the W&B stage marker for the run (e.g. ``stage-2-decay``)."""
    stage = "stage-2" if resumed else "stage-1"
    return f"{stage}-{'decay' if decay_triggered else 'stable'}"


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
):
    """Assemble the restart-complete checkpoint payload (plain dict).

    The live WSD state (``wsd``) is persisted alongside the scheduler so a
    crash mid-decay resumes from the original trigger step; legacy
    checkpoints without the key are tolerated on resume.
    """
    return {
        "checkpoint_version": 2,
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
    }


def should_log_artifacts(log_backend) -> bool:
    """W&B artifacts are only meaningful when W&B is the logging backend."""
    return log_backend == "wandb"


def build_artifact_metadata(
    kind, stage, step, tokens_seen, best_val_loss, git_commit, local_path, wandb_run_id
):
    """Identifying metadata attached to a checkpoint W&B artifact."""
    return {
        "kind": kind,
        "stage": stage,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "git_commit": git_commit,
        "local_path": local_path,
        "wandb_run_id": wandb_run_id,
    }


def log_artifact(wandb, path, name, metadata) -> bool:
    """Upload ``path`` to W&B as artifact ``name`` with ``metadata``.

    Never raises: an artifact failure (connection drop, server error) only
    prints a warning so it cannot abort a long training run. Returns True
    when the artifact was logged.
    """
    try:
        artifact = wandb.Artifact(name, type="models")
        artifact.add_file(str(path))
        artifact.metadata.update(metadata)
        wandb.log_artifact(artifact)
        return True
    except Exception as error:
        print(f"Warning: failed to log W&B artifact {name!r}: {error}")
        return False


def build_lr_lambda(warmup_steps, decay_steps, peak_lr, min_lr, wsd_state):
    warmup_steps = int(warmup_steps)
    decay_steps = max(1, int(decay_steps))
    peak_lr = float(peak_lr)
    min_lr = float(min_lr)

    if warmup_steps < 0:
        raise ValueError(f"warmup steps must be >= 0; got {warmup_steps}")
    if not 0.0 < min_lr <= peak_lr:
        raise ValueError(
            f"training.min_lr must be greater than 0 and at most training.lr; "
            f"got min_lr={min_lr}, lr={peak_lr}"
        )

    min_scale = min_lr / peak_lr

    def lr_lambda(step):
        # Warm up linearly, then decay to min_lr by decay_end_step and hold.
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        if wsd_state["triggered"] is True and wsd_state["step"] is not None:
            progress = min(max(step - wsd_state["step"], 0) / decay_steps, 1.0)
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        else:
            return 1.0
        return min_scale + (1.0 - min_scale) * cosine_scale

    return lr_lambda


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


def atomic_torch_save(payload: dict, path: str | os.PathLike) -> None:
    """Write a checkpoint atomically and make the directory entry durable."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.resume_path = resolve_resume_path(cfg.training)
        self.resume_checkpoint = None
        if self.resume_path is not None:
            self.resume_checkpoint = load_checkpoint(self.resume_path)
            print(f"Resuming from checkpoint: {self.resume_path}")

        self.set_seed(cfg.training.seed)

        self.device = torch.device(
            cfg.training.device if torch.cuda.is_available() else "cpu"
        )

        self.use_bf16 = bool(getattr(cfg.training, "use_bf16", False))
        self.use_tensor_cores = bool(getattr(cfg.training, "use_tensor_cores", False))
        self.use_compile = bool(getattr(cfg.training, "compile", False))
        self.compile_mode = str(getattr(cfg.training, "compile_mode", "default"))
        self._configure_cuda_features()

        # Reconstruct the architecture from the checkpoint when possible. In
        # particular, this preserves old checkpoints whose model config did
        # not contain ``tie_embeddings`` (missing means untied).
        self.model_cfg = cfg.model
        if self.resume_checkpoint is not None:
            saved_model_cfg = self.resume_checkpoint.get("model_cfg")
            if saved_model_cfg is None:
                saved_cfg = self.resume_checkpoint.get("cfg", {})
                if isinstance(saved_cfg, dict):
                    saved_model_cfg = saved_cfg.get("model")
            if saved_model_cfg is not None:
                self.model_cfg = OmegaConf.create(saved_model_cfg)

        # TODO: replace with a factory that dispatches on cfg.model.name (gpt2/moe/qwen)
        self.model = GptModel(self.model_cfg)

        if self.resume_checkpoint is not None:
            self._restore_model(self.resume_checkpoint)
        self.model.to(self.device)

        if self.use_compile:
            if self.device.type == "cuda":
                print(f"Compiling model with mode={self.compile_mode}")
                # Load the plain module first; compiled wrappers can add
                # ``_orig_mod`` prefixes to state-dict keys.
                self.model.compile(mode=self.compile_mode)
            else:
                print(
                    "Compilation requested, but CUDA is unavailable; skipping compile."
                )

        self.model_parameter_metrics = get_model_parameter_metrics(self.model)
        self._print_model_summary()

        optimizer_kwargs = {
            "lr": float(cfg.training.lr),
            "weight_decay": float(cfg.training.weight_decay),
        }

        if self.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)
        self.train_loader, self.val_loader = create_dataloaders(cfg.data)

        self.log_backend = resolve_log_backend(cfg.training)
        self.use_wandb = self.log_backend == "wandb"
        self.step = 0
        self.best_val_loss = float("inf")
        self.epoch = 0
        self.batch_in_epoch = 0
        self.tokens_seen = 0
        self.run_elapsed_seconds = 0.0
        self.stop_requested = False
        self.wandb_run_id = None
        self.steps_per_epoch = len(self.train_loader)
        self.tokens_per_step = int(cfg.data.batch_size) * int(cfg.data.seq_len)
        self.total_steps, self.budget_name = resolve_total_steps(
            cfg.training, self.steps_per_epoch, self.tokens_per_step
        )
        self.total_train_tokens = self.total_steps * self.tokens_per_step

        # Will this resume trigger or continue a WSD decay? Decidable before
        # the step counter is restored: the start_decay flag or a triggered
        # state saved in the checkpoint.
        saved_wsd = (
            self.resume_checkpoint.get("wsd")
            if self.resume_checkpoint is not None
            else None
        )
        self.start_decay_requested = bool(getattr(cfg.training, "start_decay", False))
        # Decidable before the step counter is restored: only whether a
        # decay triggers matters here (the true trigger step is filled in
        # after the restore), so resume_step=0 is a placeholder.
        self.decay_run = bool(
            self.resume_checkpoint is not None
            and resolve_wsd_state(saved_wsd, self.start_decay_requested, 0)["triggered"]
        )

        if self.resume_checkpoint is not None:
            check_resume_consistency(
                saved_total_steps=self.resume_checkpoint.get("total_steps"),
                total_steps=self.total_steps,
                saved_steps_per_epoch=self.resume_checkpoint.get("steps_per_epoch"),
                steps_per_epoch=self.steps_per_epoch,
                decay_run=self.decay_run,
            )

        # Decay runs re-print their derived budget after the override below.
        if not self.decay_run:
            print(
                f"Optimizer steps: {self.total_steps:,} "
                f"({self.steps_per_epoch:,}/data pass; budget={self.budget_name})"
            )
            print(f"Training tokens: {format_count(self.total_train_tokens)}")
        print(f"Logging backend: {self.log_backend}")

        # --- Checkpoint dir (hydra run dir, falls back to ./runs) ---
        if self.resume_path is not None:
            # Keep subsequent latest/best/final checkpoints beside the source
            # checkpoint, so restarting a job does not split one experiment
            # across a new Hydra directory.
            self.save_dir = str(self.resume_path.parent)
        else:
            try:
                from hydra.core.hydra_config import HydraConfig

                self.save_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                self.save_dir = "runs"
        os.makedirs(self.save_dir, exist_ok=True)

        # Run-manifest identity fields, computed once per run.
        self.run_name = cfg.training.exp_name
        self._manifest_git_commit = git_commit()
        self._manifest_config_digest = config_digest(
            OmegaConf.to_container(cfg, resolve=True)
        )

        if self.resume_checkpoint is not None:
            self._restore_training_state(self.resume_checkpoint)

        # --- LR scheduler: WSD ---
        # Seed the live WSD state after the restore so a triggered state saved
        # in the checkpoint wins over a fresh start_decay flag (a crash
        # mid-decay resumes from the original trigger step). Fresh runs never
        # trigger a decay.
        if self.resume_checkpoint is not None:
            self.wsd_decay = resolve_wsd_state(
                saved_wsd, self.start_decay_requested, self.step
            )
        else:
            self.wsd_decay = {"triggered": False, "step": None}
        decay_triggered = self.wsd_decay["triggered"]
        self.stage_label = stage_label(
            resumed=self.resume_checkpoint is not None,
            decay_triggered=decay_triggered,
        )

        peak_lr = float(cfg.training.lr)
        min_lr = float(getattr(cfg.training, "min_lr", peak_lr * 0.1))
        decay_fraction = float(cfg.training.lr_decay_fraction)
        if decay_triggered:
            # Decay runs skip warmup (stage 1 already ran it) and derive
            # their budget from the trigger step: D = round(S*f/(1-f)). The
            # run stops exactly when the decay finishes.
            warmup = 0
            self.total_steps, decay_steps = resolve_decay_budget(
                self.wsd_decay["step"], decay_fraction
            )
            self.budget_name = "wsd_decay"
            self.total_train_tokens = self.total_steps * self.tokens_per_step
            print(
                f"WSD decay: trigger step S={self.wsd_decay['step']}, "
                f"decay steps D={decay_steps} "
                f"(lr_decay_fraction={decay_fraction})"
            )
            print(
                f"Optimizer steps: {self.total_steps:,} "
                f"({self.steps_per_epoch:,}/data pass; budget={self.budget_name})"
            )
            print(f"Training tokens: {format_count(self.total_train_tokens)}")
        else:
            warmup = int(cfg.training.warmup_fraction * self.total_steps)
            decay_steps = int(decay_fraction * self.total_steps)

        lr_lambda = build_lr_lambda(
            warmup, decay_steps, peak_lr, min_lr, self.wsd_decay
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        if self.resume_checkpoint is not None:
            scheduler_state = self.resume_checkpoint.get("scheduler")
            if scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            else:
                print(
                    "Warning: checkpoint has no scheduler state; this is a legacy "
                    "checkpoint and cannot be resumed bit-for-bit."
                )
            optimizer_state = self.resume_checkpoint.get("optimizer")
            if optimizer_state is not None:
                # LambdaLR construction initializes the optimizer's current
                # learning rate. Load the optimizer after the scheduler so
                # the checkpoint's exact current LR wins.
                self.optimizer.load_state_dict(optimizer_state)

        if self.use_wandb:
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
            wandb_name = cfg.training.exp_name
            if self.resume_checkpoint is not None:
                saved_cfg = self.resume_checkpoint.get("cfg")
                if isinstance(saved_cfg, dict):
                    # Keep the resumed run's original configuration in W&B;
                    # the resume path itself is an invocation detail, not a
                    # new experiment configuration.
                    wandb_config = saved_cfg
                    saved_training_cfg = saved_cfg.get("training", {})
                    wandb_name = saved_training_cfg.get(
                        "exp_name", wandb_name
                    )
            wandb_kwargs = {
                "project": "mini-llm-lab",
                "name": wandb_name,
                "config": wandb_config,
            }
            if self.wandb_run_id is not None:
                wandb_kwargs.update(id=self.wandb_run_id, resume="must")
                print(f"Resuming W&B run: {self.wandb_run_id}")
            wandb.init(**wandb_kwargs)
            if wandb.run is not None:
                self.wandb_run_id = wandb.run.id
            stage_marker = {"training_stage": self.stage_label}
            if self.wsd_decay["triggered"]:
                stage_marker["wsd_trigger_step"] = self.wsd_decay["step"]
            wandb.config.update(
                {
                    "computed_steps_per_epoch": self.steps_per_epoch,
                    "computed_total_steps": self.total_steps,
                    "computed_tokens_per_step": self.tokens_per_step,
                    "computed_total_train_tokens": self.total_train_tokens,
                    "computed_budget_name": self.budget_name,
                    **stage_marker,
                    "model_parameters_total": self.model_parameter_metrics[
                        "model/parameters_total"
                    ],
                    "model_parameters_trainable": self.model_parameter_metrics[
                        "model/parameters_trainable"
                    ],
                    "model_parameters_non_trainable": self.model_parameter_metrics[
                        "model/parameters_non_trainable"
                    ],
                    "model_parameter_memory_mb": self.model_parameter_metrics[
                        "model/parameter_memory_mb"
                    ],
                }
            )
            wandb.log(
                {**self.model_parameter_metrics, "step": self.step},
                step=self.step,
            )

    def _print_model_summary(self):
        """Print model size and architecture metadata for terminal runs."""
        metrics = self.model_parameter_metrics
        attention = getattr(self.model_cfg, "attention", "unknown")
        tied_embeddings = bool(getattr(self.model_cfg, "tie_embeddings", False))
        print(
            "Model parameters: "
            f"{format_count(metrics['model/parameters_total'])} total | "
            f"{format_count(metrics['model/parameters_trainable'])} trainable | "
            f"{format_count(metrics['model/parameters_non_trainable'])} frozen"
        )
        print(
            "Model memory: "
            f"{metrics['model/parameter_memory_mb']:.2f} MiB | "
            f"attention: {attention} | tied embeddings: {tied_embeddings}"
        )

    def _restore_model(self, checkpoint):
        """Load plain or compiled-module state into the uncompiled model."""
        state = checkpoint.get("model") or checkpoint.get("model_state_dict")
        if state is None:
            raise KeyError("Checkpoint does not contain a model state dictionary")

        normalized_state = {
            key.removeprefix("_orig_mod."): value for key, value in state.items()
        }
        missing, unexpected = self.model.load_state_dict(
            normalized_state, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint model does not match the configured architecture: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def _restore_training_state(self, checkpoint):
        """Restore counters, cursor, W&B identity, and all RNG streams."""
        self.step = int(checkpoint.get("step", 0))
        self.best_val_loss = float(
            checkpoint.get("best_val_loss", float("inf"))
        )
        self.wandb_run_id = checkpoint.get("wandb_run_id")
        self.run_elapsed_seconds = float(checkpoint.get("run_elapsed_seconds", 0.0))

        cursor = checkpoint.get("cursor")
        if cursor is None:
            # Old checkpoints can still be loaded, but they do not contain
            # enough information for an exact continuation.
            self.epoch, self.batch_in_epoch = divmod(
                self.step, self.steps_per_epoch
            )
        else:
            self.epoch = int(cursor.get("epoch", 0))
            self.batch_in_epoch = int(cursor.get("batch_in_epoch", 0))

        saved_tokens = checkpoint.get("tokens_seen")
        self.tokens_seen = (
            int(saved_tokens)
            if saved_tokens is not None
            else self.step * self.tokens_per_step
        )

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
            train_generator = getattr(self.train_loader, "generator", None)
            val_generator = getattr(self.val_loader, "generator", None)
            if train_generator is not None and dataloader_rng.get("train") is not None:
                train_generator.set_state(dataloader_rng["train"])
            if val_generator is not None and dataloader_rng.get("val") is not None:
                val_generator.set_state(dataloader_rng["val"])

        self._set_train_epoch(self.epoch)

    def _set_train_epoch(self, epoch):
        """Set the deterministic sampler to the logical data-pass number."""
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _capture_rng_state(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "dataloader": {
                "train": getattr(self.train_loader, "generator", None).get_state()
                if getattr(self.train_loader, "generator", None) is not None
                else None,
                "val": getattr(self.val_loader, "generator", None).get_state()
                if getattr(self.val_loader, "generator", None) is not None
                else None,
            },
        }

    def _configure_cuda_features(self):
        """Configure optional CUDA precision and Tensor Core features."""
        if self.device.type != "cuda":
            if self.use_bf16:
                print("BF16 requested, but CUDA is unavailable; disabling BF16.")
                self.use_bf16 = False
            if self.use_tensor_cores:
                print(
                    "Tensor Cores requested, but CUDA is unavailable; "
                    "disabling Tensor Core settings."
                )
                self.use_tensor_cores = False
            return

        capability = torch.cuda.get_device_capability()
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU compute capability: {capability[0]}.{capability[1]}")

        if self.use_tensor_cores and capability[0] >= 8:
            # Enables TF32 Tensor Core matmuls for operations that remain FP32.
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            print("TF32 Tensor Core matmuls: enabled")
        elif self.use_tensor_cores:
            print(
                "Tensor Core setting requested, but TF32 requires compute capability "
                "8.0+; leaving FP32 matmuls at default precision."
            )
        else:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            print("TF32 Tensor Core matmuls: disabled")

        if self.use_bf16:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "BF16 was requested, but this CUDA device does not support BF16."
                )
            print("BF16 mixed precision: enabled")

    def _autocast_context(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_lr(self):
        """Current learning rate, as set by the scheduler."""
        return self.optimizer.param_groups[0]["lr"]

    def calc_loss_batch(self, x, y):
        with self._autocast_context():
            logits = self.model(x)
            loss = self.criterion(logits.flatten(0, 1), y.flatten())
        return loss

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (x, y) in enumerate(data_loader):
            if i < num_batches:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.calc_loss_batch(x, y)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    

    def generate_and_log_sample(self, loss=None):
        self.model.eval()
        encoded = text_to_token_ids(self.cfg.training.start_context, self.tokenizer, device=self.device)
        
        with torch.no_grad():
            token_ids = self.model.generate(
                encoded,
                context_length=self.cfg.model.context_length,
                max_new_tokens=self.cfg.training.max_new_tokens,
                autocast_context=self._autocast_context(),
                temperature=self.cfg.model.temperature,
                top_k=self.cfg.model.top_k,
            )
        text = token_ids_to_text(token_ids, self.tokenizer).replace("\n", " ")
        self.model.train()
        if self.use_wandb:
            wandb.log(
                {"sample_text": wandb.Html(f"<pre>{text}</pre>"), "step": self.step},
                step=self.step,
            )
        else:
            loss_text = "loss n/a" if loss is None else f"val loss {loss:.4f}"
            print(f"sample | step {self.step} | {loss_text} | text: {text}")

    def _log_metrics(self, metrics):
        """Send metrics to W&B when it is the selected logging backend."""
        if self.use_wandb:
            wandb.log(metrics, step=self.step)

    def train_step(self, x, y):
        """Run one optimization step and return the scalar loss."""
        self.optimizer.zero_grad()
        # x: [B, S] token ids -> model -> logits [B, S, V]; y: [B, S] target ids
        loss = self.calc_loss_batch(x, y)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def evaluate(self):
        """Run validation and return (avg_loss, perplexity)."""
        self.model.eval()
        eval_batches = int(
            getattr(self.cfg.training, "eval_batches", self.cfg.training.eval_interval)
        )
        with torch.no_grad():
            train_loss = self.calc_loss_loader(
                self.train_loader, num_batches=eval_batches
            )
            val_loss = self.calc_loss_loader(self.val_loader, num_batches=eval_batches)
            train_perplexity, val_perplexity = math.exp(train_loss), math.exp(val_loss)
        self.model.train()
        return train_loss, train_perplexity, val_loss, val_perplexity

    def save_checkpoint(self, path, kind=None):
        """Persist a restart-complete checkpoint to ``path`` atomically.

        ``kind`` (``latest`` / ``best`` / ``final`` / ``periodic``) also
        records the save in the run manifest.
        """
        atomic_torch_save(
            build_checkpoint_payload(
                model_state=self.model.state_dict(),
                model_cfg=OmegaConf.to_container(self.model_cfg, resolve=True),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                step=self.step,
                best_val_loss=self.best_val_loss,
                cursor={
                    "epoch": self.epoch,
                    "batch_in_epoch": self.batch_in_epoch,
                },
                tokens_seen=self.tokens_seen,
                run_elapsed_seconds=getattr(self, "run_elapsed_seconds", 0.0),
                steps_per_epoch=self.steps_per_epoch,
                tokens_per_step=self.tokens_per_step,
                total_steps=self.total_steps,
                wandb_run_id=self.wandb_run_id,
                rng_state=self._capture_rng_state(),
                cfg_container=OmegaConf.to_container(self.cfg, resolve=True),
                wsd_state=self.wsd_decay,
            ),
            path,
        )
        if kind is not None:
            self._record_manifest(kind, Path(path).name)

    def _record_manifest(self, kind, filename):
        """Record a checkpoint save in the run manifest (never fatal)."""
        try:
            wandb_url = None
            if self.use_wandb and wandb.run is not None:
                wandb_url = getattr(wandb.run, "url", None)
            entry = make_entry(
                file=filename,
                kind=kind,
                step=self.step,
                tokens_seen=self.tokens_seen,
                stage=self.stage_label,
                best_val_loss=self.best_val_loss,
                timestamp=utc_timestamp(),
                wandb_run_id=self.wandb_run_id,
                wandb_url=wandb_url,
            )
            upsert_manifest(
                self.save_dir,
                run_name=self.run_name,
                wandb_run_id=self.wandb_run_id,
                wandb_url=wandb_url,
                git_commit=self._manifest_git_commit,
                config_digest=self._manifest_config_digest,
                entry=entry,
            )
        except Exception as error:
            print(f"Warning: failed to update the run manifest: {error}")

    def _log_checkpoint_artifact(self, kind, name, path):
        """Upload a saved checkpoint as a W&B artifact (W&B backend only).

        Failures inside ``log_artifact`` are non-fatal by design; terminal
        mode performs no W&B interaction at all.
        """
        if not should_log_artifacts(self.log_backend):
            return
        log_artifact(
            wandb,
            path,
            name,
            build_artifact_metadata(
                kind=kind,
                stage=self.stage_label,
                step=self.step,
                tokens_seen=self.tokens_seen,
                best_val_loss=self.best_val_loss,
                git_commit=self._manifest_git_commit,
                local_path=path,
                wandb_run_id=self.wandb_run_id,
            ),
        )

    def _memory_metrics(self):
        """Return current and peak CUDA memory usage in GiB for W&B."""
        if self.device.type != "cuda":
            return {}

        return {
            "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated(self.device)
            / 1024**3,
            "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved(self.device)
            / 1024**3,
            "system/gpu_memory_peak_allocated_gb": torch.cuda.max_memory_allocated(
                self.device
            )
            / 1024**3,
            "system/gpu_memory_peak_reserved_gb": torch.cuda.max_memory_reserved(
                self.device
            )
            / 1024**3,
        }

    def _request_shutdown(self, signum, _frame):
        """Ask the loop to checkpoint after the current optimizer step."""
        if not self.stop_requested:
            print(
                f"Received {signal.Signals(signum).name}; "
                "checkpointing after the current optimizer step."
            )
        self.stop_requested = True

    def _install_shutdown_handlers(self):
        previous_handlers = {}
        for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
            signal_number = getattr(signal, signal_name, None)
            if signal_number is not None:
                previous_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, self._request_shutdown)
        return previous_handlers

    @staticmethod
    def _restore_shutdown_handlers(previous_handlers):
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)

    def _update_run_elapsed(self):
        if hasattr(self, "_run_start_time"):
            self.run_elapsed_seconds = self._elapsed_before_run + (
                time.perf_counter() - self._run_start_time
            )

    def _save_latest_checkpoint(self):
        self._update_run_elapsed()
        self.save_checkpoint(os.path.join(self.save_dir, "latest.pt"), kind="latest")

    def train(self):
        """Outer training loop."""
        self.model.train()
        self._run_start_time = time.perf_counter()
        self._elapsed_before_run = self.run_elapsed_seconds
        previous_handlers = self._install_shutdown_handlers()

        # Match the reference implementation: CUDA events measure GPU work,
        # while wall-clock timing is used for CPU fallback.
        use_cuda_timer = self.device.type == "cuda"
        if use_cuda_timer:
            timer_start = torch.cuda.Event(enable_timing=True)
            timer_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            timer_start.record()
        else:
            timer_start_time = time.time()
        log_start_tokens = self.tokens_seen

        try:
            while self.step < self.total_steps and not self.stop_requested:
                current_epoch = self.epoch
                self._set_train_epoch(current_epoch)
                for batch_idx, (x, y) in enumerate(self.train_loader):
                    if batch_idx < self.batch_in_epoch:
                        continue
                    if self.step >= self.total_steps or self.stop_requested:
                        break

                    x, y = x.to(self.device), y.to(self.device)
                    loss = self.train_step(
                        x, y
                    )  # train loss for this batch with dropout and stuff
                    self.step += 1
                    self.tokens_seen += x.numel()

                    # The cursor always points to the next batch to consume.
                    if batch_idx + 1 >= len(self.train_loader):
                        self.epoch = current_epoch + 1
                        self.batch_in_epoch = 0
                    else:
                        self.epoch = current_epoch
                        self.batch_in_epoch = batch_idx + 1

                    if self.step % self.cfg.training.log_interval == 0:
                        if use_cuda_timer:
                            timer_end.record()
                            torch.cuda.synchronize(self.device)
                            elapsed = timer_start.elapsed_time(timer_end) / 1000
                            timer_start.record()
                        else:
                            elapsed = time.time() - timer_start_time
                            timer_start_time = time.time()

                        tokens_since_log = self.tokens_seen - log_start_tokens
                        tok_per_sec = tokens_since_log / max(elapsed, 1e-9)
                        self._update_run_elapsed()
                        progress = min(self.step / max(self.total_steps, 1), 1.0)
                        steps_remaining = max(self.total_steps - self.step, 0)
                        steps_per_second = self.step / max(
                            self.run_elapsed_seconds, 1e-9
                        )
                        estimated_remaining_seconds = steps_remaining / max(
                            steps_per_second, 1e-9
                        )
                        memory_metrics = self._memory_metrics()
                        self._log_metrics(
                            {
                                "train/loss": loss,
                                "train/ppl": math.exp(loss),
                                "train/lr": self.get_lr(),
                                "train/tokens_seen": self.tokens_seen,
                                "train/tok_per_sec": tok_per_sec,
                                "train/interval_seconds": elapsed,
                                "run/elapsed_seconds": self.run_elapsed_seconds,
                                "run/elapsed_minutes": self.run_elapsed_seconds / 60,
                                "run/elapsed_hours": self.run_elapsed_seconds / 3600,
                                "run/steps_per_epoch": self.steps_per_epoch,
                                "run/total_steps": self.total_steps,
                                "run/steps_remaining": steps_remaining,
                                "run/progress": progress,
                                "run/estimated_remaining_seconds": estimated_remaining_seconds,
                                "step": self.step,
                                **memory_metrics,
                            }
                        )
                        print(
                            f"data pass {current_epoch + 1} | step {self.step} | "
                            f"tokens {format_count(self.tokens_seen)} | loss {loss:.4f} | "
                            f"lr {self.get_lr():.2e} | tok/s {tok_per_sec:,.0f}"
                        )
                        log_start_tokens = self.tokens_seen
                    if self.step % self.cfg.training.eval_interval == 0:
                        train_loss, train_ppl, val_loss, val_ppl = self.evaluate()
                        # Train loss here is over the first N batches in eval
                        # mode, making it comparable to validation loss.
                        self._log_metrics(
                            {
                                "val/loss": val_loss,
                                "val/ppl": val_ppl,
                                "train/loss_full": train_loss,
                                "train/ppl_full": train_ppl,
                                "step": self.step,
                            }
                        )
                        if not self.use_wandb:
                            print(
                                f"eval | step {self.step} | train loss {train_loss:.4f} | "
                                f"val loss {val_loss:.4f}"
                            )
                        self.generate_and_log_sample(loss=val_loss)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._update_run_elapsed()
                            best_path = os.path.join(self.save_dir, "best.pt")
                            self.save_checkpoint(best_path, kind="best")
                            self._log_checkpoint_artifact("best", "best-checkpoint", best_path)
                    if (
                        self.cfg.training.save_interval
                        and self.step % self.cfg.training.save_interval == 0
                    ):
                        self._update_run_elapsed()
                        self.save_checkpoint(
                            os.path.join(self.save_dir, f"step_{self.step}.pt"),
                            kind="periodic",
                        )
                        self.save_checkpoint(
                            os.path.join(self.save_dir, "latest.pt"), kind="latest"
                        )

                    if self.stop_requested:
                        print(f"Saving restart checkpoint at step {self.step}...")
                        self._save_latest_checkpoint()
                        self._log_metrics(
                            {"run/interrupted": 1, "run/steps_remaining": self.total_steps - self.step}
                        )
                        return

                # A signal can arrive while the DataLoader is preparing the
                # next batch, before the inner loop reaches the post-step
                # check above.
                if self.stop_requested:
                    print(f"Saving restart checkpoint at step {self.step}...")
                    self._save_latest_checkpoint()
                    self._log_metrics(
                        {"run/interrupted": 1, "run/steps_remaining": self.total_steps - self.step}
                    )
                    return

            self._update_run_elapsed()
            self._log_metrics(
                {
                    "run/elapsed_seconds": self.run_elapsed_seconds,
                    "run/elapsed_minutes": self.run_elapsed_seconds / 60,
                    "run/elapsed_hours": self.run_elapsed_seconds / 3600,
                    "run/steps_per_epoch": self.steps_per_epoch,
                    "run/total_steps": self.total_steps,
                    "run/steps_remaining": 0,
                    "run/progress": 1.0,
                    "run/estimated_remaining_seconds": 0.0,
                    "step": self.step,
                }
            )
            self._save_latest_checkpoint()
            final_path = os.path.join(self.save_dir, "final_model.pt")
            self.save_checkpoint(final_path, kind="final")
            self._log_checkpoint_artifact("final", "final-model", final_path)
        except KeyboardInterrupt:
            # Covers an external KeyboardInterrupt that bypasses our signal
            # handler, while still preserving the last completed optimizer step.
            self.stop_requested = True
            print(f"Interrupted; saving restart checkpoint at step {self.step}...")
            self._save_latest_checkpoint()
        finally:
            self._restore_shutdown_handlers(previous_handlers)
            self.close()

    def close(self):
        """Flush and close the W&B run when W&B logging is enabled."""
        if self.use_wandb and wandb.run is not None:
            wandb.finish()
