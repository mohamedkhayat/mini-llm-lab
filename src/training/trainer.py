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

from data.dataloader import MemmapDataLoader
from data.prepare import get_data_dir
from data.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text
from models.gpt import GptModel
from training.checkpointing import (
    RunState,
    atomic_torch_save,
    build_checkpoint_payload,
    capture_data_state,
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
    phase_state,
    resolve_phase_budget,
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
    """Run assembly (the constructor) and the training loop (:meth:`train`)."""

    def __init__(self, cfg):
        """Assemble a run in seven named setup steps, in dependency order."""
        self.cfg = cfg
        self._setup_runtime()
        self._build_model()
        self._setup_data()
        self._resolve_budget()
        self._apply_resume()
        self._build_scheduler()
        self._setup_logging()

    def _setup_runtime(self):
        """Resume checkpoint load, seed, device, CUDA precision / Tensor
        Core / compile configuration, the save directory, and run identity."""
        self.resume_path = resolve_resume_path(self.cfg.training)
        self.resume_checkpoint = None
        if self.resume_path is not None:
            self.resume_checkpoint = load_checkpoint(self.resume_path)
            print(f"Resuming from checkpoint: {self.resume_path}")

        self.set_seed(self.cfg.training.seed)

        self.device = torch.device(
            self.cfg.training.device if torch.cuda.is_available() else "cpu"
        )

        self.use_bf16 = bool(getattr(self.cfg.training, "use_bf16", False))
        self.use_tensor_cores = bool(
            getattr(self.cfg.training, "use_tensor_cores", False)
        )
        self.use_compile = bool(getattr(self.cfg.training, "compile", False))
        self.compile_mode = str(getattr(self.cfg.training, "compile_mode", "default"))
        self._configure_cuda_features()

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
        self.run_name = self.cfg.training.exp_name
        self._manifest_git_commit = git_commit()
        self._manifest_config_digest = config_digest(
            OmegaConf.to_container(self.cfg, resolve=True)
        )

    def _build_model(self):
        """Checkpoint model-config override, model build, state restore,
        device move, compile, parameter metrics + summary print."""
        # Reconstruct the architecture from the checkpoint when possible. In
        # particular, this preserves old checkpoints whose model config did
        # not contain ``tie_embeddings`` (missing means untied).
        self.model_cfg = self.cfg.model
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

    def _setup_data(self):
        """Tokenizer, memmap dataloaders, micro-batch / per-step token
        counts, optimizer, and loss."""
        self.tokenizer = get_tokenizer(self.cfg.data.tokenizer_name)
        data_dir = get_data_dir(self.cfg)
        self.data_dir = data_dir
        batch_size = int(self.cfg.training.batch_size)
        self.accum_steps = int(self.cfg.training.accum_steps)
        seq_len = int(self.cfg.data.seq_len)
        self.tokens_per_step = batch_size * seq_len * self.accum_steps

        # Resolve the requested endpoint before creating the loaders.  The
        # extra lookahead token is required because each training window reads
        # one shifted target token beyond its counted input tokens.
        self.phase_budget = resolve_phase_budget(
            self.cfg.training,
            self.resume_checkpoint,
            self.tokens_per_step,
        )
        train_limit_tokens = None
        if self.phase_budget.target_step is not None:
            train_limit_tokens = (
                self.phase_budget.target_step * self.tokens_per_step + 1
            )
        self.train_loader = MemmapDataLoader(
            data_dir,
            "train",
            batch_size,
            seq_len,
            self.device,
            max_tokens=train_limit_tokens,
        )
        physical_batches = (len(self.train_loader.data) - 1) // self.train_loader.batch_tokens
        self.available_steps_per_pass = max(1, physical_batches // self.accum_steps)
        self.val_loader = MemmapDataLoader(
            data_dir, "eval", batch_size, seq_len, self.device
        )
        if len(self.train_loader) == 0:
            raise ValueError(
                "The training data holds fewer tokens than one micro-batch "
                f"({batch_size * seq_len:,}); increase data.max_tokens or "
                "reduce training.batch_size / data.seq_len."
            )
        # Shadow loader for the train-loss eval line: reads the same train
        # file without touching the main stream's position.
        self.train_eval_loader = MemmapDataLoader(
            data_dir,
            "train",
            batch_size,
            seq_len,
            self.device,
            max_tokens=train_limit_tokens,
        )

        # One data pass = one scan of the train file (single pass over the
        # data). The loader yields micro-batches; an optimizer step consumes
        # accum_steps of them, so the file capacity in optimizer steps is
        # len // accum_steps and one step moves batch_size * seq_len *
        # accum_steps tokens.
        self.steps_per_pass = max(1, len(self.train_loader) // self.accum_steps)

        optimizer_kwargs = {
            "lr": float(self.cfg.training.lr),
            "weight_decay": float(self.cfg.training.weight_decay),
            "betas" : [float(self.cfg.training.beta1), float(self.cfg.training.beta2)]
        }
        if self.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()

    def _resolve_budget(self):
        """Resolve the explicit phase endpoint and validate resume geometry."""
        self.step = 0
        self.best_val_loss = float("inf")
        self.tokens_seen = 0
        self.run_elapsed_seconds = 0.0
        self.stop_requested = False
        self.wandb_run_id = None

        self.resume_mode = str(getattr(self.cfg.training, "resume_mode", "exact"))
        if self.resume_mode not in {"exact", "continue"}:
            raise ValueError("training.resume_mode must be 'exact' or 'continue'")

        # _setup_data resolves this before the loader is built.  The fallback
        # keeps the small Trainer.__new__ test seam and older callers working.
        self.phase_budget = getattr(
            self,
            "phase_budget",
            resolve_phase_budget(
                self.cfg.training,
                self.resume_checkpoint,
                self.tokens_per_step,
            ),
        )
        self.continuation_run = self.phase_budget.mode == "stable_continue"
        self.decay_run = self.phase_budget.stage == "decay"
        self.start_decay_requested = bool(
            getattr(self.cfg.training, "start_decay", False)
        )

        self.total_steps = (
            self.phase_budget.target_step
            if self.phase_budget.target_step is not None
            else self.steps_per_pass
        )
        budget_names = {
            "one_pass": "one_pass",
            "train_tokens": "train_tokens",
            "stable_continue": "continue_tokens",
            "decay_tokens": "wsd_decay_tokens",
            "legacy_decay": "wsd_decay_legacy",
            "exact_resume": "exact_resume",
        }
        self.budget_name = budget_names.get(
            self.phase_budget.mode, self.phase_budget.mode
        )
        self.total_train_tokens = self.total_steps * self.tokens_per_step

        if self.resume_checkpoint is not None:
            check_resume_consistency(
                saved_total_steps=self.resume_checkpoint.get("total_steps"),
                total_steps=self.total_steps,
                saved_steps_per_pass=self.resume_checkpoint.get(
                    "steps_per_pass"
                ),
                steps_per_pass=self.steps_per_pass,
                saved_tokens_per_step=self.resume_checkpoint.get(
                    "tokens_per_step"
                ),
                tokens_per_step=self.tokens_per_step,
                decay_run=self.decay_run,
                continuation_run=self.continuation_run,
                allow_capacity_growth=self.continuation_run or self.decay_run,
                available_steps_per_pass=getattr(
                    self, "available_steps_per_pass", self.steps_per_pass
                ),
            )

        self._print_budget_summary()

    def _print_budget_summary(self):
        """Console budget report."""
        print(f"Optimizer steps: {self.total_steps:,} (budget={self.budget_name})")
        print(f"Training tokens: {format_count(self.total_train_tokens)}")
        if self.phase_budget.requested_tokens is not None:
            print(
                "Phase tokens: "
                f"requested={format_count(self.phase_budget.requested_tokens)} | "
                f"effective={format_count(self.phase_budget.effective_tokens)}"
            )
        print(
            "Phase: "
            f"{self.phase_budget.stage} | start={self.phase_budget.start_step:,} | "
            f"target={self.phase_budget.target_step if self.phase_budget.target_step is not None else 'cache end'}"
        )
        if hasattr(self, "train_loader") and hasattr(self, "val_loader"):
            print(
                "Cache: "
                f"{len(self.train_loader.data) + len(self.val_loader.data):,} total | "
                f"{len(self.train_loader.data):,} train | "
                f"{len(self.val_loader.data):,} val | "
                f"phase loader={self.train_loader.total_tokens:,} train tokens"
            )

    def _apply_resume(self):
        """Restore state, seed the explicit phase state, and print its budget."""
        if self.resume_checkpoint is not None:
            self._restore_training_state(self.resume_checkpoint)

        # A saved explicit endpoint wins over command-line flags.  For a new
        # request phase_budget was resolved before loader construction; its
        # start step is the restored checkpoint step.
        self.wsd_decay = phase_state(self.phase_budget)
        decay_triggered = self.wsd_decay["triggered"]
        self.stage_label = stage_label(
            resumed=self.resume_checkpoint is not None,
            decay_triggered=decay_triggered,
        )

        if decay_triggered:
            print(
                f"WSD decay: start step S={self.wsd_decay['decay_start_step']}, "
                f"additional tokens={format_count(self.wsd_decay['effective_tokens'])}, "
                f"decay steps D={self.wsd_decay['decay_steps']}, "
                f"end step={self.wsd_decay['target_step']}"
            )
        elif self.phase_budget.mode == "stable_continue":
            print(
                f"Stable continuation: additional tokens="
                f"{format_count(self.phase_budget.effective_tokens)}, "
                f"end step={self.phase_budget.target_step}"
            )

    def _build_scheduler(self):
        """Warmup / decay math and the LR lambda, then the scheduler — and
        the load-bearing restore order: the scheduler state is restored
        before the optimizer state, so the checkpoint's exact current
        learning rate wins (LambdaLR construction reinitializes the
        optimizer's current LR)."""
        peak_lr = float(self.cfg.training.lr)
        min_lr = float(getattr(self.cfg.training, "min_lr", peak_lr * 0.1))
        if self.wsd_decay["triggered"]:
            warmup = 0
            decay_steps = int(
                self.wsd_decay.get("decay_steps")
                or self.total_steps - self.wsd_decay["step"]
            )
        elif self.continuation_run or self.resume_checkpoint is not None:
            # A resume is an extension of an already-trained stable run. It
            # must not introduce a second warmup window.
            warmup = 0
            decay_steps = 1
        else:
            warmup = int(self.cfg.training.warmup_fraction * self.total_steps)
            decay_steps = 1

        lr_lambda = build_lr_lambda(
            warmup, decay_steps, peak_lr, min_lr, self.wsd_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        if self.resume_checkpoint is not None:
            if self.resume_checkpoint.get("scheduler") is None:
                raise ValueError(
                    "The checkpoint does not carry scheduler state; it is "
                    "not resumable with this trainer. Start a fresh run "
                    "instead."
                )
            if self.resume_checkpoint.get("optimizer") is None:
                raise ValueError(
                    "The checkpoint does not carry optimizer state; it is "
                    "not resumable with this trainer. Start a fresh run "
                    "instead."
                )
            # Load the optimizer after the scheduler so the checkpoint's
            # exact current LR wins (LambdaLR construction reinitializes
            # the optimizer's current LR).
            self.scheduler.load_state_dict(self.resume_checkpoint["scheduler"])
            self.optimizer.load_state_dict(self.resume_checkpoint["optimizer"])

    def _setup_logging(self):
        """Backend resolution, the adapter init, and the initial
        parameter-metric log."""
        self.log_backend = resolve_log_backend(self.cfg.training)
        self.upload_artifacts = bool(
            getattr(self.cfg.training, "upload_artifacts", False)
        )
        print(f"Logging backend: {self.log_backend}")
        print(
            "W&B checkpoint artifact uploads: "
            f"{'enabled' if self.upload_artifacts and self.log_backend == 'wandb' else 'disabled'}"
        )

        # One adapter per backend; the trainer holds exactly one and contains
        # no W&B branches of its own.
        self.logger = create_logger(
            self.log_backend, upload_artifacts=self.upload_artifacts
        )
        stage_marker = {"training_stage": self.stage_label}
        if self.wsd_decay["triggered"]:
            stage_marker["wsd_trigger_step"] = self.wsd_decay["step"]
        saved_cfg = (
            self.resume_checkpoint.get("cfg")
            if self.resume_checkpoint is not None
            else None
        )
        self.logger.init(
            name=self.run_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            saved_config=saved_cfg if isinstance(saved_cfg, dict) else None,
            computed={
                "computed_steps_per_pass": self.steps_per_pass,
                "computed_total_steps": self.total_steps,
                "computed_tokens_per_step": self.tokens_per_step,
                "computed_total_train_tokens": self.total_train_tokens,
                "computed_budget_name": self.budget_name,
                "computed_phase_mode": self.phase_budget.mode,
                "computed_phase_stage": self.phase_budget.stage,
                "computed_requested_phase_tokens": self.phase_budget.requested_tokens,
                "computed_effective_phase_tokens": self.phase_budget.effective_tokens,
                "computed_phase_target_step": self.phase_budget.target_step,
                "computed_decay_steps": self.phase_budget.decay_steps,
                "computed_train_loader_tokens": self.train_loader.total_tokens,
                "computed_available_train_tokens": len(self.train_loader.data),
                "computed_available_val_tokens": len(self.val_loader.data),
                "computed_available_cache_tokens": len(self.train_loader.data)
                + len(self.val_loader.data),
                "computed_available_steps_per_pass": self.available_steps_per_pass,
                "computed_phase_start_step": self.phase_budget.start_step,
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
            tokens_seen=self.tokens_seen,
            tokens_per_step=self.tokens_per_step,
            train_loader=self.train_loader,
        )
        restore_training_state(bundle, checkpoint)
        self.step = bundle.step
        self.best_val_loss = bundle.best_val_loss
        self.wandb_run_id = bundle.wandb_run_id
        self.run_elapsed_seconds = bundle.run_elapsed_seconds
        self.tokens_seen = bundle.tokens_seen

    def _capture_rng_state(self):
        # The memmap dataloaders have no RNG of their own: the data stream
        # is a deterministic scan and its position is checkpointed as
        # data_state.
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
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
        encoded = text_to_token_ids(
            self.cfg.training.start_context, self.tokenizer, device=self.device
        )

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

    def train_step(self, micro_batches):
        """One optimizer step over a group of micro-batches (gradient
        accumulation): gradients are summed (each micro-batch loss is
        scaled by 1/accum_steps), clipped, then a single optimizer + LR
        step is taken."""
        self.optimizer.zero_grad()
        total = 0.0
        for x, y in micro_batches:
            loss = self.calc_loss_batch(x, y) / self.accum_steps
            loss.backward()
            total += loss.item()
        clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return total

    def evaluate(self):
        """Run validation and return (avg_loss, perplexity)."""
        self.model.eval()
        eval_batches = int(
            getattr(self.cfg.training, "eval_batches", self.cfg.training.eval_interval)
        )
        with torch.no_grad():
            # Train-loss line over the first eval_batches batches of
            # train.bin — a fixed, deterministic window: the scan is
            # sequential, so the start of every data pass is batch 0. The
            # shadow loader keeps the main train stream's position
            # untouched, so evals never consume or rewind it.
            self.train_eval_loader.load_state_dict({"current_step": 0})
            train_loss = self.calc_loss_loader(
                self.train_eval_loader, num_batches=eval_batches
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
                tokens_seen=self.tokens_seen,
                run_elapsed_seconds=getattr(self, "run_elapsed_seconds", 0.0),
                steps_per_pass=self.steps_per_pass,
                tokens_per_step=self.tokens_per_step,
                total_steps=self.total_steps,
                wandb_run_id=self.wandb_run_id,
                rng_state=self._capture_rng_state(),
                cfg_container=OmegaConf.to_container(self.cfg, resolve=True),
                wsd_state=self.wsd_decay,
                data_state=capture_data_state(self.train_loader),
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

    def _log_progress(self, loss):
        """Timer math (CUDA events on GPU, wall clock on CPU), the progress
        metrics, and the console line for one logged step."""
        if self._use_cuda_timer:
            self._timer_end.record()
            torch.cuda.synchronize(self.device)
            elapsed = self._timer_start.elapsed_time(self._timer_end) / 1000
            self._timer_start.record()
        else:
            elapsed = time.time() - self._timer_start_time
            self._timer_start_time = time.time()

        tokens_since_log = self.tokens_seen - self._log_start_tokens
        tok_per_sec = tokens_since_log / max(elapsed, 1e-9)
        self._update_run_elapsed()
        progress = min(self.step / max(self.total_steps, 1), 1.0)
        steps_remaining = max(self.total_steps - self.step, 0)
        steps_per_second = self.step / max(self.run_elapsed_seconds, 1e-9)
        estimated_remaining_seconds = steps_remaining / max(steps_per_second, 1e-9)
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
                "run/steps_per_epoch": self.steps_per_pass,
                "run/total_steps": self.total_steps,
                "run/steps_remaining": steps_remaining,
                "run/tokens_remaining": steps_remaining * self.tokens_per_step,
                "run/progress": progress,
                "run/estimated_remaining_seconds": estimated_remaining_seconds,
                "step": self.step,
                **memory_metrics,
            }
        )
        print(
            f"step {self.step} | tokens {format_count(self.tokens_seen)} | "
            f"loss {loss:.4f} | lr {self.get_lr():.2e} | tok/s {tok_per_sec:,.0f}"
        )
        self._log_start_tokens = self.tokens_seen

    def _run_evaluation(self):
        """Evaluation losses + generated sample + best-checkpoint save."""
        train_loss, train_ppl, val_loss, val_ppl = self.evaluate()
        # Train loss here is over the first N batches in eval mode, making it
        # comparable to validation loss.
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

    def _save_periodic_checkpoint(self):
        """Periodic restart checkpoint: a step-N file plus a latest refresh."""
        self._update_run_elapsed()
        self.save_checkpoint(
            os.path.join(self.save_dir, f"step_{self.step}.pt"), kind="periodic"
        )
        self.save_checkpoint(os.path.join(self.save_dir, "latest.pt"), kind="latest")

    def _handle_stop(self):
        """One unified stop path for signal stops and KeyboardInterrupt:
        checkpoint the current step, then log the interrupted-run metrics."""
        print(f"Saving restart checkpoint at step {self.step}...")
        self._save_latest_checkpoint()
        self._log_metrics(
            {
                "run/interrupted": 1,
                "run/steps_remaining": self.total_steps - self.step,
                "run/tokens_remaining": max(self.total_steps - self.step, 0)
                * self.tokens_per_step,
            }
        )

    def _finish_run(self):
        """Run completion: closing run metrics, latest + final checkpoints,
        and the final checkpoint's artifact."""
        self._update_run_elapsed()
        self._log_metrics(
            {
                "run/elapsed_seconds": self.run_elapsed_seconds,
                "run/elapsed_minutes": self.run_elapsed_seconds / 60,
                "run/elapsed_hours": self.run_elapsed_seconds / 3600,
                "run/steps_per_epoch": self.steps_per_pass,
                "run/total_steps": self.total_steps,
                "run/steps_remaining": 0,
                "run/tokens_remaining": 0,
                "run/progress": 1.0,
                "run/estimated_remaining_seconds": 0.0,
                "step": self.step,
            }
        )
        self._save_latest_checkpoint()
        final_path = os.path.join(self.save_dir, "final_model.pt")
        self.save_checkpoint(final_path, kind="final")
        self._log_checkpoint_artifact("final", "final-model", final_path)

    def train(self):
        """Outer training loop."""
        if self.total_steps > self.steps_per_pass:
            raise ValueError(
                f"The training budget ({self.total_steps:,} steps) exceeds one "
                f"pass over the data file ({self.steps_per_pass:,} steps). "
                "The launcher should have expanded the cache before loader "
                "construction; check data.max_tokens and the phase budget, "
                "then resume from the last checkpoint."
            )
        self.model.train()
        self._run_start_time = time.perf_counter()
        self._elapsed_before_run = self.run_elapsed_seconds
        previous_handlers = self._install_shutdown_handlers()

        # Match the reference implementation: CUDA events measure GPU work,
        # while wall-clock timing is used for the CPU fallback.
        self._use_cuda_timer = self.device.type == "cuda"
        if self._use_cuda_timer:
            self._timer_start = torch.cuda.Event(enable_timing=True)
            self._timer_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            self._timer_start.record()
        else:
            self._timer_start_time = time.time()
        self._log_start_tokens = self.tokens_seen

        try:
            batch_iter = iter(self.train_loader)
            while self.step < self.total_steps and not self.stop_requested:
                # Pull one group of micro-batches for the optimizer step.
                # A stop can be requested between pulls (e.g. between two
                # micro-batches); it is honored before the step, so a
                # checkpoint never lands mid-accumulation.
                micro_batches = []
                for _ in range(self.accum_steps):
                    try:
                        micro_batches.append(next(batch_iter))
                    except StopIteration:
                        raise RuntimeError(
                            "Ran out of data before the budget was reached; "
                            "the trainer does not rescan the file. This "
                            "should be impossible: train() rejects budgets "
                            "larger than one pass over the data."
                        )
                if self.stop_requested:
                    break

                loss = self.train_step(micro_batches)
                self.step += 1
                self.tokens_seen += self.tokens_per_step

                if self.step % self.cfg.training.log_interval == 0:
                    self._log_progress(loss)
                if self.step % self.cfg.training.eval_interval == 0:
                    self._run_evaluation()
                if (
                    self.cfg.training.save_interval
                    and self.step % self.cfg.training.save_interval == 0
                ):
                    self._save_periodic_checkpoint()
                if self.stop_requested:
                    break

            if self.stop_requested:
                self._handle_stop()
            else:
                self._finish_run()
        except KeyboardInterrupt:
            # Covers an external KeyboardInterrupt that bypasses our signal
            # handler, while still preserving the last completed optimizer
            # step; the unified stop path logs the interrupted-run metric.
            self.stop_requested = True
            self._handle_stop()
        finally:
            self._restore_shutdown_handlers(previous_handlers)
            self.close()

    def close(self):
        """Flush and close the run's log backend."""
        self.logger.finish()
