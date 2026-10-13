# mini-llm-lab

Pretrain decoder-only language models from scratch in PyTorch. Intentionally single-GPU and built for inspection and experimentation: every component is written cleanly from the ground up, ready to be modified directly in source or reconfigured via Hydra overrides. When throughput matters, drop-in toggles allow you to swap in optimized kernels and paths, including PyTorch SDPA, torch.compile, and Liger Kernel fused cross-entropy.

---

## Features

- **Model:** Decoder-only transformer supporting MHA, PyTorch SDPA, and Grouped-Query Attention (GQA). Includes RoPE, RMSNorm, QK-norm, and gated SwiGLU-style MLPs.
- **Data:** Preprocessing for Hugging Face Parquet datasets into binary memmap caches. Includes sequential cursor tracking for deterministic resume.
- **Training:** AdamW with gradient accumulation. Supports BF16, TF32, `torch.compile`, and Liger fused linear cross-entropy.
- **Schedule:** Two-stage WSD (Warmup-Stable-Decay) learning rate schedule allowing for flexible plateau extension and manual decay triggers.
- **Reliability:** Atomic checkpointing, signal handling for graceful shutdowns (SIGINT/SIGTERM), and experiment tracking via W&B or terminal.

## Model Presets

| Preset | Status | Description |
| --- | --- | --- |
| `gpt2` | Active | Default baseline architecture. |
| `qwen` | In progress | Uses RoPE, RMSNorm, and gated SiLU. Dedicated architecture pending. |
| `moe` | In progress | Configuration placeholder for Mixture-of-Experts routing. |

Architecture features can be combined via CLI overrides:
```bash
python train.py model=gpt2 model.normalization=rmsnorm model.position_embedding=rope model.attention=gqa model.n_kv_heads=4 model.gated=true
```

## Quickstart

Requires Python 3.12+ and PyTorch.

```bash
# Install
pip install -e '.[dev]'

# Start training (Default: CUDA, W&B logging)
python train.py

# Terminal logging only
python train.py training.log_backend=terminal

# Use Liger Kernel for reduced VRAM
python train.py training.use_flce=true
```

## Pretraining Workflow

The training loop uses a Warmup-Stable-Decay (WSD) schedule:

1.  **Stage 1 (Stable):** Run training until the desired loss plateau is reached.
    ```bash
    python train.py training.save_interval=500
    ```
2.  **Continue:** If more data is needed, extend the plateau at a constant learning rate.
    ```bash
    python train.py training.resume_from=latest training.resume_mode=continue training.continue_tokens=1b
    ```
3.  **Stage 2 (Decay):** Trigger the final cosine cooldown to the learning rate floor.
    ```bash
    python train.py training.resume_from=latest training.start_decay=true training.decay_tokens=1b
    ```

## Documentation

For a full breakdown of all Hydra configuration options, hardware settings, and model hyperparameters, see the **[Configuration Reference](configuration.md)**.

## Project Structure

- `configs/`: Hydra YAML configurations.
- `src/data/`: Dataset processing and binary loaders.
- `src/models/`: Transformer layers and attention backends.
- `src/training/`: Trainer, LR schedules, and checkpointing.
- `chat.py`: Chainlit interface for model inference.
- `train.py`: Main execution entry point.
