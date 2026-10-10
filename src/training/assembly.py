"""Trainer assembly (spec 5.11): build_trainer.

The single entry point that turns a composed config and an optional
already-loaded checkpoint into a ready-to-``train()`` loop-only
``Trainer``.  Everything config-derived (runtime, model, data, budget,
state, optimizer, scheduler, loss, evaluators, tracker, checkpoint
manager) is built here, in dependency order, and the loop-only ``Trainer``
receives ready-made components with no cfg (spec 5.12).

Validation that must run BEFORE any logging/W&B side effect (the
resume-argument rule, the budget vs. pass-length rule, resume
consistency) is enforced ahead of ``tracker.start`` — so a bad config
produces zero W&B calls (P3).  ``train.py`` remains the only caller of
``load_checkpoint`` / ``resolve_resume_path`` (P9): it loads the
checkpoint once and hands the dict here.
"""

import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from data.budget import parse_token_budget
from data.dataloader import MemmapDataLoader
from data.prepare import get_data_dir
from data.tokenizer import get_tokenizer
from models.gpt import GptModel
from training.ckpt_manager import CheckpointManager, RunIdentity
from training.checkpointing import (
    build_checkpoint_payload,
    capture_data_state,
    resume_requested,
    resolve_resume_path,
    restore_data_state,
    restore_model,
    restore_training_state,
)
from training.evaluator import Evaluator, Sampler
from training.loss import build_loss
from training.log_backend import (
    RunInfo,
    build_artifact_metadata,
    create_logger,
    resolve_log_backend,
)
from training.meter import RunClock, ThroughputMeter
from training.run_manifest import config_digest, git_commit
from training.schedule import (
    build_lr_lambda,
    phase_state,
    resolve_budget,
    resolve_phase_budget,
    stage_label,
)
from training.state import TrainState
from training.tracking import RunTracker
from training.trainer import Intervals, Trainer
from training.utils import format_count, get_model_parameter_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    # The memmap dataloaders have no RNG of their own: the data stream is a
    # deterministic scan and its position is checkpointed as data_state.
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


@dataclass(frozen=True)
class _Runtime:
    """The resolved runtime facts of a run (device, precision, paths)."""

    device: torch.device
    save_dir: str
    run_name: str
    use_bf16: bool
    use_compile: bool
    compile_mode: str


def resolve_runtime(cfg, checkpoint) -> _Runtime:
    """Seed, device, CUDA precision/Tensor Core configuration, the save
    directory, and the run name (spec 5.11, step 1).

    The checkpoint is loaded exactly once, by train.py, and passed in —
    assembly only reads its existence to pick the save dir (P9).
    """
    set_seed(cfg.training.seed)
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    use_bf16 = bool(getattr(cfg.training, "use_bf16", False))
    use_tensor_cores = bool(getattr(cfg.training, "use_tensor_cores", False))
    use_compile = bool(getattr(cfg.training, "compile", False))
    compile_mode = str(getattr(cfg.training, "compile_mode", "default"))
    _configure_cuda_features(device, use_bf16, use_tensor_cores)
    # F8: pass the resolved truth downstream — on a non-CUDA device the
    # loss path would otherwise enter a CUDA autocast (with warnings) on
    # every call.
    use_bf16 = use_bf16 and device.type == "cuda"

    if checkpoint is not None:
        resume_path = resolve_resume_path(cfg.training)
        if resume_path is None:
            # A checkpoint was loaded but resume_from is unset: keep the
            # Hydra run dir (fresh output layout, resumed weights).
            save_dir = _hydra_save_dir()
        else:
            # Keep subsequent latest/best/final checkpoints beside the
            # source checkpoint, so restarting a job does not split one
            # experiment across a new Hydra directory.
            save_dir = str(resume_path.parent)
    else:
        save_dir = _hydra_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    return _Runtime(
        device=device,
        save_dir=save_dir,
        run_name=cfg.training.exp_name,
        use_bf16=use_bf16,
        use_compile=use_compile,
        compile_mode=compile_mode,
    )


def _hydra_save_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        return HydraConfig.get().runtime.output_dir
    except Exception:
        return "runs"


def _configure_cuda_features(device, use_bf16, use_tensor_cores) -> None:
    """Configure optional CUDA precision and Tensor Core features."""
    if device.type != "cuda":
        if use_bf16:
            print("BF16 requested, but CUDA is unavailable; disabling BF16.")
        if use_tensor_cores:
            print(
                "Tensor Cores requested, but CUDA is unavailable; "
                "disabling Tensor Core settings."
            )
        return

    capability = torch.cuda.get_device_capability()
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU compute capability: {capability[0]}.{capability[1]}")

    if use_tensor_cores and capability[0] >= 8:
        # Enables TF32 Tensor Core matmuls for operations that remain FP32.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        print("TF32 Tensor Core matmuls: enabled")
    elif use_tensor_cores:
        print(
            "Tensor Core setting requested, but TF32 requires compute capability "
            "8.0+; leaving FP32 matmuls at default precision."
        )
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        print("TF32 Tensor Core matmuls: disabled")

    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 was requested, but this CUDA device does not support BF16."
        )
    if use_bf16:
        print("BF16 mixed precision: enabled")


def build_model(cfg, checkpoint, runtime: _Runtime):
    """Model config override, model build, state restore, device move,
    compile, parameter metrics + summary print (spec 5.11, step 2)."""
    # Reconstruct the architecture from the checkpoint when possible. In
    # particular, this preserves old checkpoints whose model config did
    # not contain ``tie_embeddings`` (missing means untied).
    model_cfg = cfg.model
    if checkpoint is not None:
        saved_model_cfg = checkpoint.get("model_cfg")
        if saved_model_cfg is None:
            saved_cfg = checkpoint.get("cfg", {})
            if isinstance(saved_cfg, dict):
                saved_model_cfg = saved_cfg.get("model")
        if saved_model_cfg is not None:
            model_cfg = OmegaConf.create(saved_model_cfg)
        # F4: a saved model_cfg without the lm_head_bias key cannot decide
        # the architecture — checkpoints written after the bias removal
        # but before the key existed have no key AND no bias tensor. The
        # checkpoint's tensors are the source of truth.
        if "lm_head_bias" not in model_cfg:
            state = checkpoint.get("model") or {}
            has_bias = any(
                key.removeprefix("_orig_mod.") == "lm_head.bias" for key in state
            )
            model_cfg = OmegaConf.merge(model_cfg, {"lm_head_bias": has_bias})

    # TODO: replace with a factory that dispatches on cfg.model.name
    # (gpt2/moe/qwen)
    model = GptModel(model_cfg)
    if checkpoint is not None:
        restore_model(model, checkpoint)
    model.to(runtime.device)

    if runtime.use_compile:
        if runtime.device.type == "cuda":
            print(f"Compiling model with mode={runtime.compile_mode}")
            # Load the plain module first; compiled wrappers can add
            # ``_orig_mod`` prefixes to state-dict keys.
            model = cast(
                GptModel, torch.compile(model, mode=runtime.compile_mode)
            )
        else:
            print(
                "Compilation requested, but CUDA is unavailable; skipping compile."
            )

    metrics = get_model_parameter_metrics(model)
    _print_model_summary(model_cfg, metrics)
    return model, model_cfg, metrics


def _print_model_summary(model_cfg, metrics) -> None:
    """Print model size and architecture metadata for terminal runs."""
    attention = getattr(model_cfg, "attention", "unknown")
    tied_embeddings = bool(getattr(model_cfg, "tie_embeddings", False))
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


def build_data(cfg, model, checkpoint, runtime: _Runtime):
    """Tokenizer, memmap dataloaders, micro-batch / per-step token counts
    and the optimizer (spec 5.11, step 3). The phase endpoint is resolved
    FIRST — the loader needs it to bound the scan before the loaders
    exist. A resumed run resolves against the checkpoint, so the saved
    phase endpoint wins."""
    tokenizer = get_tokenizer(cfg.data.tokenizer_name)
    data_dir = get_data_dir(cfg)
    batch_size = int(cfg.training.batch_size)
    accum_steps = int(cfg.training.accum_steps)
    seq_len = int(cfg.data.seq_len)
    tokens_per_step = batch_size * seq_len * accum_steps

    # The extra lookahead token is required because each training window
    # reads one shifted target token beyond its counted input tokens.
    phase = resolve_phase_budget(
        cfg.training, checkpoint, tokens_per_step=tokens_per_step
    )
    train_limit_tokens = None
    if phase.target_step is not None:
        train_limit_tokens = phase.target_step * tokens_per_step + 1
    train_loader = MemmapDataLoader(
        data_dir,
        "train",
        batch_size,
        seq_len,
        runtime.device,
        max_tokens=train_limit_tokens,
    )
    physical_batches = (
        len(train_loader.data) - 1
    ) // train_loader.batch_tokens
    available_steps_per_pass = max(1, physical_batches // accum_steps)
    val_loader = MemmapDataLoader(data_dir, "eval", batch_size, seq_len, runtime.device)
    if len(train_loader) == 0:
        raise ValueError(
            "The training data holds fewer tokens than one micro-batch "
            f"({batch_size * seq_len:,}); increase data.max_tokens or "
            "reduce training.batch_size / data.seq_len."
        )
    # One data pass = one scan of the train file (single pass over the
    # data). The loader yields micro-batches; an optimizer step consumes
    # accum_steps of them, so the file capacity in optimizer steps is
    # len // accum_steps and one step moves batch_size * seq_len *
    # accum_steps tokens.
    steps_per_pass = max(1, len(train_loader) // accum_steps)

    decay_params = [
        p for p in model.parameters() if p.requires_grad and p.ndim >= 2
    ]
    nodecay_params = [
        p for p in model.parameters() if p.requires_grad and p.ndim < 2
    ]
    optim_groups = [
        {"params": decay_params, "weight_decay": float(cfg.training.weight_decay)},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer_kwargs = {
        "lr": float(cfg.training.lr),
        "betas": [float(cfg.training.beta1), float(cfg.training.beta2)],
    }
    if runtime.device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(optim_groups, **optimizer_kwargs)

    return SimpleNamespace(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        phase=phase,
        steps_per_pass=steps_per_pass,
        available_steps_per_pass=available_steps_per_pass,
        tokens_per_step=tokens_per_step,
        optimizer=optimizer,
        accum_steps=accum_steps,
    )


def _print_budget_summary(budget, train_loader, val_loader) -> None:
    """Console budget report (moved out of the trainer by T6.3)."""
    print(f"Optimizer steps: {budget.total_steps:,} (budget={budget.budget_name})")
    print(f"Training tokens: {format_count(budget.total_train_tokens)}")
    if budget.phase.requested_tokens is not None:
        print(
            "Phase tokens: "
            f"requested={format_count(budget.phase.requested_tokens)} | "
            f"effective={format_count(budget.phase.effective_tokens)}"
        )
    print(
        "Phase: "
        f"{budget.phase.stage} | start={budget.phase.start_step:,} | "
        f"target={budget.phase.target_step if budget.phase.target_step is not None else 'cache end'}"
    )
    print(
        "Cache: "
        f"{len(train_loader.data) + len(val_loader.data):,} total | "
        f"{len(train_loader.data):,} train | "
        f"{len(val_loader.data):,} val | "
        f"phase loader={train_loader.total_tokens:,} train tokens"
    )


def _print_phase_marker(budget) -> None:
    """The decay / stable-continuation marker lines (unchanged)."""
    if budget.phase.stage == "decay":
        print(
            f"WSD decay: start step S={budget.phase.decay_start_step}, "
            f"additional tokens={format_count(budget.phase.effective_tokens)}, "
            f"decay steps D={budget.phase.decay_steps}, "
            f"end step={budget.phase.target_step}"
        )
    elif budget.phase.mode == "stable_continue":
        print(
            f"Stable continuation: additional tokens="
            f"{format_count(budget.phase.effective_tokens)}, "
            f"end step={budget.phase.target_step}"
        )


def build_state(checkpoint, train_loader) -> TrainState:
    """Fresh or restored run state (spec 5.11, step 4): counters, W&B
    identity, all RNG streams, and the memmap data cursor (restored
    separately into the training loader)."""
    state = TrainState()
    if checkpoint is not None:
        restore_training_state(state, checkpoint)
        restore_data_state(train_loader, checkpoint.get("data_state"))
    return state


def build_scheduler(cfg, optimizer, budget, checkpoint):
    """Warmup / decay math and the LR lambda, then the scheduler — and
    the load-bearing restore order: the scheduler state is restored
    BEFORE the optimizer state, so the checkpoint's exact current
    learning rate wins (LambdaLR construction reinitializes the
    optimizer's current LR)."""
    peak_lr = float(cfg.training.lr)
    min_lr = float(getattr(cfg.training, "min_lr", peak_lr * 0.1))

    # Warmup was computed once at budget resolution and persisted on the
    # phase, so the curve is independent of later warmup_fraction edits.
    lr_lambda = build_lr_lambda(
        budget.phase.warmup_steps, peak_lr, min_lr, budget.phase
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if checkpoint is not None:
        if checkpoint.get("scheduler") is None:
            raise ValueError(
                "The checkpoint does not carry scheduler state; it is "
                "not resumable with this trainer. Start a fresh run "
                "instead."
            )
        if checkpoint.get("optimizer") is None:
            raise ValueError(
                "The checkpoint does not carry optimizer state; it is "
                "not resumable with this trainer. Start a fresh run "
                "instead."
            )
        # Load the optimizer after the scheduler so the checkpoint's
        # exact current LR wins (LambdaLR construction reinitializes the
        # optimizer's current LR).
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    return scheduler


def build_loss_fn(model, cfg, runtime: _Runtime):
    """One shared loss path for training and evaluation; FLCE is
    resolved once here (and Liger imported lazily only when active)."""
    use_flce = bool(getattr(cfg.training, "use_flce", False))
    raw_model = getattr(model, "_orig_mod", model)
    return build_loss(
        model, raw_model, use_flce, runtime.device, use_bf16=runtime.use_bf16
    )


def build_evaluators(model, loss_fn, tokenizer, cfg, model_cfg, val_loader,
                     device, use_bf16):
    """The periodic (batch-capped) evaluator, the once-at-completion full
    scan, and the sample generator. Generation settings come from a plain
    namespace (the generation config group proper arrives in T7.2).
    Returns (evaluator, final_evaluator, sampler, autocast_ctx)."""
    eval_batches = int(
        getattr(cfg.training, "eval_batches", cfg.training.eval_interval)
    )
    evaluator = Evaluator(
        model,
        loss_fn,
        val_loader,
        num_batches=eval_batches,
        device=device,
    )
    # F7: optional cap on the final scan — tokens -> micro-batches
    # (null = full scan; decimal suffixes like 8m are accepted).
    final_cap = getattr(cfg.training, "final_eval_max_tokens", None)
    final_num_batches = None
    if final_cap is not None:
        cap_tokens = parse_token_budget(
            final_cap, "training.final_eval_max_tokens"
        )
        if cap_tokens is None:
            raise ValueError(
                "training.final_eval_max_tokens must be a token count "
                f"(e.g. 512 or 8m); got {final_cap!r}"
            )
        final_num_batches = cap_tokens // (
            int(cfg.training.batch_size) * int(cfg.data.seq_len)
        )
    final_evaluator = Evaluator(
        model, loss_fn, val_loader, num_batches=final_num_batches, device=device
    )
    gen_cfg = SimpleNamespace(
        start_context=str(cfg.training.start_context),
        max_new_tokens=int(cfg.training.max_new_tokens),
        temperature=float(model_cfg.temperature),
        top_k=model_cfg.top_k,
    )
    from contextlib import nullcontext

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )
    sampler = Sampler(
        model,
        tokenizer=tokenizer,
        gen_cfg=gen_cfg,
        context_length=int(model_cfg.context_length),
        autocast_ctx=autocast_ctx,
    )
    return evaluator, final_evaluator, sampler, autocast_ctx


def build_tracker(cfg, checkpoint, budget, state, stage_label_str, runtime,
                  model_metrics, train_loader, val_loader, logger=None):
    """Initialize run tracking AFTER model, budget, and resume state
    (spec 5.11 — the first W&B-capable step; everything before it is
    side-effect free, P3). An injected ``logger`` (test seam) replaces
    ``create_logger``."""
    backend = "terminal" if logger is not None else resolve_log_backend(cfg.training)
    upload_artifacts = bool(getattr(cfg.training, "upload_artifacts", False))
    print(f"Logging backend: {backend}")
    print(
        "W&B checkpoint artifact uploads: "
        f"{'enabled' if upload_artifacts and backend == 'wandb' else 'disabled'}"
    )
    # No trainer reference inside the tracker: everything it needs is
    # explicit (spec 5.7).
    tracker = RunTracker(
        logger if logger is not None else create_logger(
            backend, upload_artifacts=upload_artifacts
        ),
        save_dir=runtime.save_dir,
        run_name=runtime.run_name,
        git_commit=git_commit(),
        config_digest=config_digest(
            OmegaConf.to_container(cfg, resolve=True)
        ),
        terminal=backend == "terminal",
    )
    tracker.start(_build_run_info(
        cfg=cfg,
        checkpoint=checkpoint,
        budget=budget,
        state=state,
        stage_label=stage_label_str,
        run_name=runtime.run_name,
        model_metrics=model_metrics,
        train_loader=train_loader,
        val_loader=val_loader,
    ))
    # F3: persist the backend's run id into state BEFORE any checkpoint
    # save, so payloads carry it and a resume reuses the same W&B run
    # (resume='must'). Terminal backends report None and keep id-less.
    if tracker.run_id is not None:
        state.wandb_run_id = tracker.run_id
    return tracker


def _build_run_info(*, cfg, checkpoint, budget, state, stage_label, run_name,
                    model_metrics, train_loader, val_loader) -> RunInfo:
    """The run's identity for the tracker (explicit values; spec 5.7)."""
    stage_marker: dict[str, Any] = {"training_stage": stage_label}
    if budget.phase.stage == "decay":
        stage_marker["wsd_trigger_step"] = budget.phase.decay_start_step
    saved_cfg = checkpoint.get("cfg") if checkpoint is not None else None
    return RunInfo(
        name=run_name,
        config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
        saved_config=saved_cfg if isinstance(saved_cfg, dict) else None,
        resumed_run_id=state.wandb_run_id,
        computed={
            "computed_steps_per_pass": budget.steps_per_pass,
            "computed_total_steps": budget.total_steps,
            "computed_tokens_per_step": budget.tokens_per_step,
            "computed_total_train_tokens": budget.total_train_tokens,
            "computed_budget_name": budget.budget_name,
            "computed_phase_mode": budget.phase.mode,
            "computed_phase_stage": budget.phase.stage,
            "computed_requested_phase_tokens": budget.phase.requested_tokens,
            "computed_effective_phase_tokens": budget.phase.effective_tokens,
            "computed_phase_target_step": budget.phase.target_step,
            "computed_decay_steps": budget.phase.decay_steps,
            "computed_train_loader_tokens": train_loader.total_tokens,
            "computed_available_train_tokens": len(train_loader.data),
            "computed_available_val_tokens": len(val_loader.data),
            "computed_available_cache_tokens": len(train_loader.data)
            + len(val_loader.data),
            "computed_available_steps_per_pass": budget.available_steps_per_pass,
            "computed_phase_start_step": budget.phase.start_step,
        },
        stage_marker=stage_marker,
        parameter_metrics=model_metrics,
        step=state.step,
    )


def build_ckpt_manager(
    save_dir,
    run_name,
    model,
    model_cfg,
    cfg,
    optimizer,
    scheduler,
    budget,
    state,
    clock,
    train_loader,
    tracker,
    stage_label_str,
):
    """Wire the CheckpointManager (P4): the snapshot closure captures the
    cfg-derived payload values here (the loop-only trainer holds no cfg),
    and the before_save hook stamps run-elapsed so the loop no longer
    does it at every save call site."""
    raw_model = getattr(model, "_orig_mod", model)
    model_cfg_container = OmegaConf.to_container(model_cfg, resolve=True)
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    git = tracker.git_commit

    def snapshot() -> dict:
        return build_checkpoint_payload(
            model_state=raw_model.state_dict(),
            model_cfg=model_cfg_container,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            step=state.step,
            best_val_loss=state.best_val_loss,
            tokens_seen=state.tokens_seen,
            run_elapsed_seconds=state.run_elapsed_seconds,
            steps_per_pass=budget.steps_per_pass,
            tokens_per_step=budget.tokens_per_step,
            total_steps=budget.total_steps,
            wandb_run_id=state.wandb_run_id,
            rng_state=capture_rng_state(),
            cfg_container=cfg_container,
            wsd_state=phase_state(budget.phase),
            data_state=capture_data_state(train_loader),
        )

    def before_save() -> None:
        state.run_elapsed_seconds = clock.elapsed()

    def artifact_metadata(kind, path) -> dict:
        return build_artifact_metadata(
            kind=kind,
            stage=stage_label_str,
            step=state.step,
            tokens_seen=state.tokens_seen,
            best_val_loss=state.best_val_loss,
            git_commit=git,
            local_path=path,
            wandb_run_id=state.wandb_run_id,
        )

    identity = RunIdentity(
        run_name=run_name,
        git_commit=git,
        config_digest=tracker.config_digest,
    )
    return CheckpointManager(
        save_dir,
        snapshot,
        identity,
        tracker,
        before_save=before_save,
        stage_label=stage_label_str,
        artifact_metadata=artifact_metadata,
    )


def build_trainer(cfg, checkpoint=None, *, logger=None) -> Trainer:
    """Assemble and configure a loop-only Trainer from a composed config
    (spec 5.11 pipeline).

    ``checkpoint`` is the payload dict already loaded by train.py (or
    ``None`` for a fresh run); the trainer never loads checkpoints itself.
    ``logger`` is a test seam: an injected RunLogger replaces
    ``create_logger`` (terminal backend, no W&B).

    Raises if ``training.resume_from`` is set without a loaded checkpoint
    instead of silently starting a fresh run, and (before any W&B call,
    P3) if the budget cannot fit one pass over the data or the resume
    geometry is inconsistent.
    """
    # F9: None / '' / False all mean not requested (shared predicate).
    if resume_requested(cfg.training) and checkpoint is None:
        raise ValueError(
            f"training.resume_from is set "
            f"({getattr(cfg.training, 'resume_from', None)!r}) but no "
            "checkpoint was loaded; refusing to silently start a fresh run. "
            "Load the checkpoint (train.py does this) and pass it to "
            "build_trainer."
        )

    # 1. runtime (seed/device/cuda/save_dir) — no W&B side effects yet.
    runtime = resolve_runtime(cfg, checkpoint)
    # 2. model.
    model, model_cfg, model_metrics = build_model(cfg, checkpoint, runtime)
    # 3. data + optimizer (the phase endpoint is resolved inside, before
    #    the loaders, so the loader can bound its scan; a resumed run
    #    resolves against the checkpoint).
    data = build_data(cfg, model, checkpoint, runtime)
    # 4. budget + validation — BEFORE the tracker starts (P3).
    budget = resolve_budget(
        cfg.training,
        checkpoint,
        steps_per_pass=data.steps_per_pass,
        available_steps_per_pass=data.available_steps_per_pass,
        tokens_per_step=data.tokens_per_step,
        phase=data.phase,
    )
    _print_budget_summary(budget, data.train_loader, data.val_loader)
    # 5. state (+ data cursor restore).
    state = build_state(checkpoint, data.train_loader)
    stage_label_str = stage_label(
        resumed=checkpoint is not None,
        decay_triggered=budget.phase.stage == "decay",
    )
    _print_phase_marker(budget)
    # 6. scheduler (+ the load-bearing restore order).
    scheduler = build_scheduler(cfg, data.optimizer, budget, checkpoint)
    # 7. loss + evaluators + sampler.
    loss_fn = build_loss_fn(model, cfg, runtime)
    evaluator, final_evaluator, sampler, autocast_ctx = build_evaluators(
        model, loss_fn, data.tokenizer, cfg, model_cfg,
        data.val_loader, runtime.device, runtime.use_bf16,
    )
    # 8. tracker (the first W&B-capable step) + checkpoint manager.
    #    One clock is shared between the loop's progress metering and the
    #    checkpoint before_save stamp.
    clock = RunClock(state.run_elapsed_seconds)
    tracker = build_tracker(
        cfg, checkpoint, budget, state, stage_label_str, runtime,
        model_metrics, data.train_loader, data.val_loader, logger=logger,
    )
    ckpt = build_ckpt_manager(
        runtime.save_dir,
        runtime.run_name,
        model, model_cfg, cfg,
        data.optimizer, scheduler, budget, state,
        clock, data.train_loader, tracker,
        stage_label_str,
    )
    # 9. the loop-only trainer (spec 5.12).
    intervals = Intervals(
        log_interval=int(cfg.training.log_interval),
        eval_interval=int(cfg.training.eval_interval),
        save_interval=cfg.training.save_interval,
    )
    max_grad_norm = float(cfg.training.max_grad_norm)
    return Trainer(
        model=model,
        optimizer=data.optimizer,
        scheduler=scheduler,
        train_loader=data.train_loader,
        loss_fn=loss_fn,
        evaluator=evaluator,
        final_evaluator=final_evaluator,
        sampler=sampler,
        ckpt=ckpt,
        tracker=tracker,
        state=state,
        budget=budget,
        clock=clock,
        meter=ThroughputMeter(runtime.device),
        intervals=intervals,
        accum_steps=data.accum_steps,
        max_grad_norm=max_grad_norm,
        autocast_ctx=autocast_ctx,
    )
