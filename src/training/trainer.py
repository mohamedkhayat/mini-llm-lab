import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_

from data.dataloader import create_dataloaders
from data.tokenizer import get_tokenizer
from models.gpt import GptModel


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


class Trainer:
    def __init__(self, cfg):
        self.set_seed(cfg.training.seed)

        self.device = torch.device(
            cfg.training.device if torch.cuda.is_available() else "cpu"
        )

        self.use_bf16 = bool(getattr(cfg.training, "use_bf16", False))
        self.use_tensor_cores = bool(getattr(cfg.training, "use_tensor_cores", False))
        self.use_compile = bool(getattr(cfg.training, "compile", False))
        self.compile_mode = str(getattr(cfg.training, "compile_mode", "default"))
        self._configure_cuda_features()

        # TODO: replace with a factory that dispatches on cfg.model.name (gpt2/moe/qwen)
        self.model = GptModel(cfg.model).to(self.device)

        if self.use_compile:
            if self.device.type == "cuda":
                print(f"Compiling model with mode={self.compile_mode}")
                # Compile in-place so checkpoint parameter names remain stable.
                self.model.compile(mode=self.compile_mode)
            else:
                print(
                    "Compilation requested, but CUDA is unavailable; skipping compile."
                )

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

        self.cfg = cfg
        self.step = 0
        self.best_val_loss = float("inf")
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = int(cfg.training.epochs) * self.steps_per_epoch
        self.tokens_per_step = int(cfg.data.batch_size) * int(cfg.data.seq_len)
        self.total_train_tokens = self.total_steps * self.tokens_per_step

        print(f"Optimizer steps: {self.total_steps:,} ({self.steps_per_epoch:,}/epoch)")
        print(f"Training tokens: {format_count(self.total_train_tokens)}")

        # --- LR scheduler: one-way linear warmup -> cosine decay ---
        warmup = int(cfg.training.warmup_steps)
        decay_steps = max(self.total_steps - warmup, 1)
        peak_lr = float(cfg.training.lr)
        min_lr = float(getattr(cfg.training, "min_lr", 0.0))
        if not 0.0 <= min_lr <= peak_lr:
            raise ValueError(
                f"training.min_lr must be between 0 and training.lr; "
                f"got min_lr={min_lr}, lr={peak_lr}"
            )

        def lr_lambda(step):
            # Clamp the schedule so it cannot restart or rise after total_steps.
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup

            progress = min(max(step - warmup, 0) / decay_steps, 1.0)
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_scale = min_lr / peak_lr if peak_lr > 0 else 0.0
            return min_scale + (1.0 - min_scale) * cosine_scale

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # --- Checkpoint dir (hydra run dir, falls back to ./runs) ---
        try:
            from hydra.core.hydra_config import HydraConfig

            self.save_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            self.save_dir = "runs"
        os.makedirs(self.save_dir, exist_ok=True)

        wandb.init(
            project="mini-llm-lab",
            name=cfg.training.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update(
            {
                "computed_steps_per_epoch": self.steps_per_epoch,
                "computed_total_steps": self.total_steps,
                "computed_tokens_per_step": self.tokens_per_step,
                "computed_total_train_tokens": self.total_train_tokens,
            }
        )

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

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        encoded_tensor = torch.tensor(encoded, device=self.device).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)
        decoded = self.tokenizer.decode(flat.tolist())
        return decoded

    def generate_simple_text(self, token_idx, max_new_tokens=10):
        for _ in range(max_new_tokens):
            idx_cond = token_idx[:, -self.cfg.model.context_length :]
            with torch.no_grad(), self._autocast_context():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :]  # take last position
            probas = torch.softmax(logits, dim=-1)
            next_token_idx = torch.argmax(probas, dim=-1, keepdim=True)
            token_idx = torch.cat((token_idx, next_token_idx), dim=-1)
        return token_idx

    def generate_and_log_sample(self):
        self.model.eval()
        encoded = self.text_to_token_ids(self.cfg.training.start_context)

        with torch.no_grad():
            token_ids = self.generate_simple_text(
                encoded,
                max_new_tokens=self.cfg.training.max_new_tokens,
            )
        text = self.token_ids_to_text(token_ids).replace("\n", " ")
        self.model.train()
        wandb.log({"sample_text": wandb.Html(f"<pre>{text}</pre>"), "step": self.step})

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

    def save_checkpoint(self, path):
        """Persist model, optimizer, and training state to ``path``."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "best_val_loss": self.best_val_loss,
                "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            },
            path,
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

    def train(self):
        """Outer training loop."""
        self.model.train()
        tokens_seen = 0
        run_start_time = time.perf_counter()

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
        log_start_tokens = 0

        for epoch in range(int(self.cfg.training.epochs)):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.train_step(
                    x, y
                )  # train loss for this batch with dropout and stuff
                self.step += 1
                tokens_seen += x.numel()
                if self.step % self.cfg.training.log_interval == 0:
                    if use_cuda_timer:
                        timer_end.record()
                        torch.cuda.synchronize(self.device)
                        elapsed = timer_start.elapsed_time(timer_end) / 1000
                        timer_start.record()
                    else:
                        elapsed = time.time() - timer_start_time
                        timer_start_time = time.time()

                    tokens_since_log = tokens_seen - log_start_tokens
                    tok_per_sec = tokens_since_log / max(elapsed, 1e-9)
                    run_elapsed_seconds = time.perf_counter() - run_start_time
                    progress = min(self.step / max(self.total_steps, 1), 1.0)
                    steps_remaining = max(self.total_steps - self.step, 0)
                    steps_per_second = self.step / max(run_elapsed_seconds, 1e-9)
                    estimated_remaining_seconds = steps_remaining / max(
                        steps_per_second, 1e-9
                    )
                    memory_metrics = self._memory_metrics()
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/ppl": math.exp(loss),
                            "train/lr": self.get_lr(),
                            "train/tokens_seen": tokens_seen,
                            "train/tok_per_sec": tok_per_sec,
                            "train/interval_seconds": elapsed,
                            "run/elapsed_seconds": run_elapsed_seconds,
                            "run/elapsed_minutes": run_elapsed_seconds / 60,
                            "run/elapsed_hours": run_elapsed_seconds / 3600,
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
                        f"epoch {epoch + 1} | step {self.step} | "
                        f"tokens {format_count(tokens_seen)} | loss {loss:.4f} | "
                        f"tok/s {tok_per_sec:,.0f}"
                    )
                    log_start_tokens = tokens_seen
                if self.step % self.cfg.training.eval_interval == 0:
                    train_loss, train_ppl, val_loss, val_ppl = self.evaluate()
                    # here train loss is loss over first N batches of train, in eval mode
                    # gives us a more apples to apples comparison to val loss
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/ppl": val_ppl,
                            "train/loss_full": train_loss,
                            "train/ppl_full": train_ppl,
                            "step": self.step,
                        }
                    )
                    self.generate_and_log_sample()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(os.path.join(self.save_dir, "best.pt"))
                if (
                    self.cfg.training.save_interval
                    and self.step % self.cfg.training.save_interval == 0
                ):
                    self.save_checkpoint(
                        os.path.join(self.save_dir, f"step_{self.step}.pt")
                    )

        run_elapsed_seconds = time.perf_counter() - run_start_time
        wandb.log(
            {
                "run/elapsed_seconds": run_elapsed_seconds,
                "run/elapsed_minutes": run_elapsed_seconds / 60,
                "run/elapsed_hours": run_elapsed_seconds / 3600,
                "run/steps_per_epoch": self.steps_per_epoch,
                "run/total_steps": self.total_steps,
                "run/steps_remaining": 0,
                "run/progress": 1.0,
                "run/estimated_remaining_seconds": 0.0,
                "step": self.step,
            }
        )
        self.close()

    def close(self):
        """Flush and close the wandb run."""
        wandb.finish()
