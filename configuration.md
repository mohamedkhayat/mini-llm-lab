# Configuration Reference

All parameters are managed via [Hydra](https://hydra.cc). Overrides follow the `section.key=value` syntax.

## Data Parameters (`data.*`)

| Parameter | Default | Description |
| --- | --- | --- |
| `hf_dataset` | `HuggingFaceFW/...` | HF Parquet repository. |
| `tokenizer_name` | `gpt2` | `tiktoken` encoding for the binary cache. |
| `max_tokens` | `6b` | Total tokens to cache (train + eval). |
| `seq_len` | `1024` | Context window size. |
| `val_ratio` | `0.1` | Fraction of tokens held out for validation. |

## Training Parameters (`training.*`)

### Hardware & Performance
| Parameter | Default | Description |
| --- | --- | --- |
| `device` | `cuda` | Compute device (`cuda` or `cpu`). |
| `compile` | `false` | Apply `torch.compile` to the model. |
| `use_bf16` | `false` | Enable bfloat16 mixed precision. |
| `use_tensor_cores` | `false` | Enable TF32 matmuls (Ampere+). |
| `use_flce` | `false` | Enable Liger fused linear cross-entropy. |

### Optimization & Schedule
| Parameter | Default | Description |
| --- | --- | --- |
| `lr` | `5e-4` | Peak learning rate. |
| `min_lr` | `5e-5` | Learning rate floor for decay. |
| `batch_size` | `8` | Micro-batch size. |
| `accum_steps` | `8` | Gradient accumulation steps. |
| `train_tokens` | `null` | Stage 1 budget. If null, uses entire cache. |
| `warmup_fraction` | `0.1` | Proportion of Stage 1 used for warmup. |
| `decay_tokens` | `null` | Stage 2 budget (required if `start_decay=true`). |

### Checkpointing & Resume
| Parameter | Default | Description |
| --- | --- | --- |
| `resume_from` | `null` | Path or `latest`/`auto`. |
| `resume_mode` | `exact` | `exact` (direct resume) or `continue` (extend plateau). |
| `save_interval` | `null` | Optimizer steps between periodic saves. |

## Model Parameters (`model.*`)

### Architecture
| Parameter | Default | Description |
| --- | --- | --- |
| `vocab_size` | `50257` | Must match tokenizer. |
| `emb_dim` | `768` | Model hidden dimension. |
| `n_layers` | `12` | Number of transformer blocks. |
| `n_heads` | `12` | Query attention heads. |
| `n_kv_heads` | `12` | KV heads (set lower for GQA). |
| `tie_embeddings` | `true` | Tie input/output embedding weights. |

### Features & Normalization
| Parameter | Default | Description |
| --- | --- | --- |
| `attention` | `mha` | `mha`, `fastmha`, or `gqa`. |
| `position_embedding`| `absolute` | `absolute` or `rope`. |
| `normalization` | `layernorm`| `layernorm` or `rmsnorm`. |
| `activation` | `gelu` | `gelu`, `silu`, or `sigmoid`. |
| `gated` | `false` | Enable gated MLP blocks (SwiGLU). |
| `qk_norm` | `false` | Apply RMSNorm to Q and K per head. |

## Resume and WSD Semantics

- **Exact Resume:** Restores weights, optimizer buffers, scheduler state, and the precise memmap data cursor.
- **Plateau Extension:** Using `resume_mode=continue` with `continue_tokens` allows adding budget to a stable run without restarting the warmup phase. 
- **Decay Trigger:** Setting `start_decay=true` transitions the schedule to a cosine cooldown. This state is saved in the checkpoint; if the decay run is interrupted, it will resume the decay trajectory rather than restarting it.