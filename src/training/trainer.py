import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_

from data.dataloader import create_dataloaders
from data.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text
from models.gpt import GptModel
from training.checkpointing import (
    RunState,
    atomic_torch_save,
    build_checkpoint_payload,
    check_resume_consistency,
    load_checkpoint,
    resolve_resume_path,
    restore_model,
    restore_training_state,
)
from training.log_backend import (
    build_artifact_metadata,
    create_logger,
    resolve_log_backend,
)
from training.run_manifest import (
    config_digest,
    git_commit,
    make_entry,
    upsert_manifest,
    utc_timestamp,
)
from training.schedule import (
    build_lr_lambda,
    resolve_decay_budget,
    resolve_total_steps,
    resolve_wsd_state,
    stage_label,
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

        # --- Log backend: one adapter per backend, resolved above ---
        self.logger = create_logger(self.log_backend)
        stage_marker = {"training_stage": self.stage_label}
        if self.wsd_decay["triggered"]:
            stage_marker["wsd_trigger_step"] = self.wsd_decay["step"]
        saved_cfg = (
            self.resume_checkpoint.get("cfg") if self.resume_checkpoint is not None else None
        )
        self.logger.init(
            name=self.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            saved_config=saved_cfg if isinstance(saved_cfg, dict) else None,
            computed={
                "computed_steps_per_epoch": self.steps_per_epoch,
                "computed_total_steps": self.total_steps,
                "computed_tokens_per_step": self.tokens_per_step,
                "computed_total_train_tokens": self.total_train_tokens,
                "computed_budget_name": self.budget_name,
            },
            stage_marker=stage_marker,
            parameter_metrics=self.model_parameter_metrics,
            step=self.step,
            resumed_run_id=self.wandb_run_id,
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
        """Load the checkpoint's weights into the freshly built model."""
        restore_model(self.model, checkpoint)

    def _restore_training_state(self, checkpoint):
        """Restore counters, cursor, W&B identity, and all RNG streams.

        The live state is handed to ``restore_training_state`` as a
        :class:`RunState` bundle and read back afterwards.
        """
        bundle = RunState(
            step=self.step,
            best_val_loss=self.best_val_loss,
            wandb_run_id=self.wandb_run_id,
            run_elapsed_seconds=self.run_elapsed_seconds,
            epoch=self.epoch,
            batch_in_epoch=self.batch_in_epoch,
            tokens_seen=self.tokens_seen,
            steps_per_epoch=self.steps_per_epoch,
            tokens_per_step=self.tokens_per_step,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
        )
        restore_training_state(bundle, checkpoint)
        self.step = bundle.step
        self.best_val_loss = bundle.best_val_loss
        self.wandb_run_id = bundle.wandb_run_id
        self.run_elapsed_seconds = bundle.run_elapsed_seconds
        self.epoch = bundle.epoch
        self.batch_in_epoch = bundle.batch_in_epoch
        self.tokens_seen = bundle.tokens_seen

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
        self.logger.log_sample(text, step=self.step, loss=loss)

    def _log_metrics(self, metrics):
        """Log metrics through the run's log backend adapter."""
        self.logger.log_metrics(metrics, self.step)

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
            wandb_url = self.logger.run_url
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
        """Record a saved checkpoint in the log backend (the W&B adapter
        uploads it as an artifact; terminal mode performs no W&B
        interaction). Failures are non-fatal by design."""
        self.logger.log_checkpoint(
            kind,
            name,
            path,
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
                        if self.log_backend != "wandb":
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
        """Flush and close the run's log backend."""
        self.logger.finish()
