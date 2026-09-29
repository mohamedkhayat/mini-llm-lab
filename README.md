# mini-llm-lab

A small, from-scratch laboratory for learning how decoder-only language models
work. The project is intentionally compact: it exposes the data pipeline,
attention implementation, model configuration, and training loop
without hiding the interesting parts behind a large framework.

This is a learning and experimentation repository, not a production training
system or a package of pretrained models.

## What is implemented

- GPT-style next-token batches read from sequential memory-mapped token caches
- Hugging Face parquet dataset input with resumable local preprocessing
- GPT-2 tokenization through `tiktoken`
- Hugging Face dataset tokenization into a local, resumable token cache
  (`train.bin` / `eval.bin`), scanned sequentially by a memmap dataloader
- Causal attention backends: multi-head, fast SDPA, and grouped-query attention
- Configurable feed-forward: activation by name and optional gated
  SwiGLU-style projections
- Optional tied input/output embeddings
- A GPT-2-style transformer model and training loop (schedule, checkpointing,
  and log backend as separate modules)
- W&B or terminal metrics, text samples, artifacts, and Hydra-run checkpoints
- Hydra configuration groups for GPT-2, Qwen-style, and Mixture-of-Experts
  experiments

The GPT-2-style model is currently wired into `train.py`. The Qwen-style and
Mixture-of-Experts files describe planned variants; all three configs currently
instantiate the same `GptModel` implementation with different dimensions.

## Project layout

```text
mini-llm-lab/
├── configs/                 # Composable Hydra configuration
│   ├── data/                # HF preparation, token caches, and memmap loader
│   ├── model/               # GPT-2, Qwen-style, and MoE settings
│   └── training/            # Optimizer and run settings
├── src/
│   ├── data/                # Tokenization, datasets, dataloaders, token caches
│   ├── models/              # GPT model, attention backends, FFN, normalization
│   └── training/            # Training loop, LR schedule, checkpointing, log backend
├── tests/                   # Pytest tests
└── train.py                 # Hydra training entry point
```

## Setup

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

PyTorch is installed using its default package-index build. If you need a
specific CUDA build, install PyTorch using the command recommended for your
system, then install this project.

## Run training

Training runs from a Hugging Face dataset. Before training, `train.py`
tokenizes `data.hf_dataset` into a local cache at
`data/<dataset>[__<config>]/<tokenizer_name>/` (`train.bin`, `eval.bin`,
`meta.json`) —
see "Data preparation" below; an existing valid cache is reused.

The default dataset is set in `configs/data/default.yaml`. Point a run at any
Hugging Face dataset (parquet) with a Hydra override:

```bash
python train.py data.hf_dataset=Salesforce/wikitext data.hf_config=wikitext-2-raw-v1
```

For a normal GPU run:

```bash
python train.py
```

The trainer scans the tokenized files with the memmap dataloader and trains
the GPT-style model. W&B logs online by default;
use `WANDB_MODE=offline` for local logging or `WANDB_MODE=disabled` to skip W&B.
You can choose terminal logging directly with `training.log_backend=terminal`.
In terminal mode, training/evaluation losses and generated sample text are
printed to the terminal:

```bash
python train.py training.log_backend=terminal
```

The default is `training.log_backend=wandb`; `WANDB_MODE=disabled` switches to
terminal logging automatically.

`data.max_tokens` is the preprocessing/cache budget: total tokens written to
the train and validation files. The current phase budget is separate:
`training.train_tokens` selects a fresh stable prefix,
`training.continue_tokens` adds stable tokens on resume, and
`training.decay_tokens` adds decay tokens after the manual phase-2 trigger.
Budgets accept plain integers or decimal suffixes such as `1.2b` and `100m`.
If a phase endpoint exceeds the cache's train side, the launcher reprocesses a
larger cache while preserving the existing token prefix.

The default learning-rate schedule is manual WSD (see **Two-stage WSD
training** below): a linear warmup, a flat plateau at `training.lr`, and a
manual cosine decay to `training.min_lr` over the explicit phase-2 token
budget.

### `train.py` command reference

`train.py` is a Hydra application. Its arguments are configuration overrides in
`key=value` form, rather than argparse-style flags:

```text
python train.py [config-group=value] [section.parameter=value] ...
```

The main config groups are:

| Override | Choices | Default | Selects |
| --- | --- | --- | --- |
| `data=...` | `default` | `default` | Hugging Face dataset, tokenizer, and tokenized cache |
| `model=...` | `gpt2`, `qwen`, `moe` | `gpt2` | Model configuration file |
| `training=...` | `default` | `default` | Optimizer, schedule, logging, and run settings |

For example:

```bash
python train.py data.hf_dataset=Salesforce/wikitext \
  data.hf_config=wikitext-2-raw-v1 model=gpt2 training.device=cuda
```

Use Hydra's built-in help and config inspection when you need the composed
configuration:

```bash
python train.py --help
python train.py --cfg job --resolve
```

The tables below document every project-level parameter in the default config.
The value shown in the default column is from the `default` preset unless the
parameter is only present in another preset.

#### Data parameters (`data.*`)

| Parameter | Default | Description |
| --- | --- | --- |
| `data.hf_dataset` | `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` | Hugging Face dataset repository name (parquet). `train.py` tokenizes it into the local cache when that cache is missing or stale. |
| `data.hf_config` | `null` | Optional dataset configuration or subset. |
| `data.revision` | `null` | Optional immutable Hugging Face revision. Cache expansion reuses the revision recorded in `meta.json` when this is null. |
| `data.text_column` | `text` | Field read from each Hugging Face example during preparation. |
| `data.file_format` | `parquet` | File format of the downloaded dataset, passed to the preparation step. |
| `data.tokenizer_name` | `gpt2` | `tiktoken` encoding name used for the tokenized cache. The model's `vocab_size` must be compatible with the selected tokenizer. |
| `data.max_tokens` | `6000000000` | Total tokens the preparation step caches (train + eval, split by `data.val_ratio`). It may be written as `6b`. A fresh `training.train_tokens` budget must fit this cache; additive stable/decay phases expand it automatically when needed. |
| `data.seq_len` | `1024` | Number of input tokens in each training example. Each target is the same window shifted one token to the right. Must not exceed `model.context_length`. |
| `data.val_ratio` | `0.1` | Validation fraction of the tokenized cache. The split is positional at preparation time, so validation windows never contain training tokens. |

The data stream is a sequential scan. The cache metadata must agree with
`data.max_tokens` and `data.val_ratio`; the train split contains the
`floor((1 - val_ratio) * max_tokens)` prefix and the remainder is validation.
Phase budgets are rounded down to complete optimizer steps. Evaluation reads the first
`training.eval_batches` micro-batches of
`eval.bin` (deterministic) plus the same number of micro-batches from the
start of `train.bin` through a shadow loader, so evaluations never consume or
rewind the training stream.

#### Training parameters (`training.*`)

| Parameter | Default | Description |
| --- | --- | --- |
| `training.exp_name` | `mini_llm` | W&B run name. It does not change the Hydra output directory. |
| `training.device` | `cuda` | Requested device, normally `cuda` or `cpu`. If CUDA is unavailable, the trainer automatically falls back to CPU. |
| `training.compile` | `false` | Enable `torch.compile` on CUDA. Compilation is skipped on CPU. The first run can spend extra time compiling. |
| `training.compile_mode` | `default` | Mode passed to `torch.compile`, such as `default` or another mode supported by the installed PyTorch version. |
| `training.use_bf16` | `false` | Use CUDA BF16 autocast for forward/evaluation work. Requires a BF16-capable GPU; it is disabled on CPU and errors on unsupported CUDA hardware. |
| `training.use_tensor_cores` | `false` | Enable TF32 matmuls for eligible CUDA devices (compute capability 8.0+). This trades some numerical precision for throughput. |
| `training.log_backend` | `wandb` | `wandb` logs metrics and samples to Weights & Biases; `terminal` prints samples and evaluation summaries locally. `term` and `console` are aliases for `terminal`. |
| `training.upload_artifacts` | `false` | Upload full checkpoint files to W&B as artifacts when `training.log_backend=wandb`. Local checkpoints are always written; enable this only when remote checkpoint copies are wanted. |
| `training.lr` | `5e-4` | AdamW peak learning rate. |
| `training.weight_decay` | `0.1` | AdamW weight decay. |
| `training.batch_size` | `8` | Micro-batch: windows per data pull and per forward/backward pass. The loader geometry — the tokenized cache and resume cursor are in micro-batch units. |
| `training.accum_steps` | `8` | Micro-batches per optimizer step (gradient accumulation): gradients are summed over the group, then one optimizer + LR step is taken. The effective batch is `batch_size * accum_steps`; the token counter and the data-pass capacity count effective batches. Must be positive. |
| `training.train_tokens` | `null` | Fresh phase-1 stable budget. It may be an integer or a suffix value such as `1.2b`; null uses the available cached train capacity. |
| `training.warmup_fraction` | `0.1` | Fraction of the stage-1 budget used for the linear learning-rate warmup. Decay runs (stage 2, `start_decay=true`) skip warmup because it already happened in stage 1. |
| `training.min_lr` | `1e-5` | Learning-rate floor reached at the end of the WSD decay. It must be positive and no greater than `training.lr`. |
| `training.decay_tokens` | `null` | Additional phase-2 decay budget, required with `training.start_decay=true`. It may be an integer or a suffix value such as `1b`; the cosine decay ends after its complete optimizer-step prefix. |
| `training.max_grad_norm` | `1.0` | Maximum gradient norm for global gradient clipping. |
| `training.eval_interval` | `200` | Run evaluation and generate a sample every this many optimizer steps. The best validation checkpoint is updated when validation loss improves. |
| `training.eval_batches` | `50` | Maximum number of training and validation batches used for each evaluation. Set it lower for quick experiments; evaluation uses model-eval mode. |
| `training.log_interval` | `10` | Print and log training loss, learning rate, throughput, and progress every this many optimizer steps. |
| `training.save_interval` | `null` | Optional periodic restart interval in optimizer steps. It writes both `step_<N>.pt` and an updated `latest.pt`. |
| `training.resume_from` | `null` | Checkpoint path to resume, relative to the launch directory, or `latest`/`auto` to select the newest `latest.pt` below the project. |
| `training.resume_mode` | `exact` | `exact` resumes the saved phase endpoint; `continue` adds `training.continue_tokens` to an untriggered stable checkpoint at constant LR and expands the cache when necessary. |
| `training.continue_tokens` | `null` | Additional stable-phase budget for `resume_mode=continue`, rounded down to complete optimizer batches. An interrupted continuation resumes its saved endpoint without adding the value twice. |
| `training.start_decay` | `false` | On a stable checkpoint, manually start phase-2 decay using `training.decay_tokens`. An active decay checkpoint resumes automatically; this flag cannot restart it. |
| `training.seed` | `42` | Global Python, NumPy, PyTorch, and CUDA seed. It is also stored in checkpoints for reproducible continuation. |
| `training.start_context` | `Every effort moves you` | Prompt used when generating the periodic text sample. |
| `training.max_new_tokens` | `50` | Maximum number of tokens appended to `start_context` for each sample. |

The learning-rate schedule is WSD (warmup → stable → decay): a linear warmup,
a flat plateau at `lr`, and a manual cosine decay to `min_lr` that is
triggered by a stage-2 run (see **Two-stage WSD training** below). A single
run that never triggers the decay trains warmup plus the stable plateau for
its selected phase-1 budget.

#### Model parameters (`model.*`)

| Parameter | Default | Description |
| --- | --- | --- |
| `model.name` | `gpt2` | Preset name. It is metadata today: the trainer currently constructs `GptModel` for every preset. |
| `model.attention` | `mha` | Attention implementation: `mha`, `fastmha` (PyTorch scaled dot-product attention), or `gqa` (grouped-query attention). |
| `model.tie_embeddings` | `true` | Reuse the token-embedding weights for the output projection when `true`, reducing unique parameter storage. |
| `model.vocab_size` | `50257` | Token embedding and output vocabulary size. Keep it compatible with `data.tokenizer_name`. |
| `model.context_length` | `1024` | Maximum sequence length supported by the positional embedding and attention mask. `data.seq_len` must be no larger. |
| `model.emb_dim` | `768` | Transformer hidden width. It must be divisible by `model.n_heads`. |
| `model.n_heads` | `12` | Number of query/attention heads in each transformer block. |
| `model.n_kv_heads` | `12` | Number of key/value heads for GQA. MHA and fast MHA use the full head count for key/value projections. For GQA, it must divide `n_heads`. |
| `model.n_layers` | `12` | Number of transformer blocks. |
| `model.drop_rate` | `0.1` | Dropout probability used in embeddings, attention, and residual blocks. |
| `model.qkv_bias` | `false` | Add bias terms to query, key, and value projections. |
| `model.normalization` | `layernorm` | Normalization implementation: `layernorm` or `rmsnorm`. The lookup is case- and underscore-insensitive. |
| `model.activation` | `gelu` | Feed-forward activation, dispatched by name: `gelu`, `silu`, or `sigmoid`. |
| `model.hidden_dim` | `3072` | Feed-forward hidden width. With `gated=true`, `equalize_params=true` shrinks it to two-thirds so the three gated matrices match the parameter count of the ungated width. |
| `model.gated` | `false` | Use a gated SwiGLU-style feed-forward: a second upcast projection multiplies the activated hidden stream elementwise. |
| `model.equalize_params` | `true` | When `gated=true`, shrink the hidden width by two-thirds so the gated FFN has roughly the ungated parameter count. |
| `model.ffn_bias` | `false` | Add bias terms to the feed-forward linear layers. |
| `model.position_embedding` | `absolute` | Position-encoding label. The active GPT implementation uses learned absolute positional embeddings; RoPE is not yet dispatched. |
| `model.residual_style` | `serial` | Residual-layout label. The active transformer block uses serial pre-norm residual connections. |
| `model.logit_softcap` | `null` | Reserved config field; not currently applied by the GPT implementation. |
| `model.rope_theta` | `null` | Reserved RoPE parameter; not used while absolute positional embeddings are active. |
| `model.temperature` | `1.0` | Temperature for periodic sample generation. Positive values sample from the distribution; `0` or a negative value uses greedy argmax. |
| `model.top_k` | `25` | Restrict periodic sample generation to the top K logits. Set `null` to disable top-k filtering. |

The `moe` preset also contains `num_experts`,
`num_experts_per_token`, and `shared_expert` fields. They describe planned
Mixture-of-Experts behavior but are not consumed by the current `GptModel`.
The Qwen-style preset's RMSNorm and gated SiLU feed-forward are active, while
its `rope`/`rope_theta` settings are metadata only. GQA is available when
`model.attention=gqa`, but the current Qwen preset still selects `mha`. Use
`model=gpt2` for the supported end-to-end training path until a model factory
is added.

| Parameter | MoE default | Description |
| --- | --- | --- |
| `model.num_experts` | `8` | Number of expert feed-forward networks in the planned MoE implementation. |
| `model.num_experts_per_token` | `2` | Number of experts each token would be routed to in the planned MoE implementation. |
| `model.shared_expert` | `false` | Whether the planned MoE block should include a shared expert. |

#### Hydra options and override rules

Hydra options use dashes and are separate from project parameters. The most
useful ones are:

| Option | Effect |
| --- | --- |
| `--help` | Show Hydra's generated help for the composed application. |
| `--cfg job --resolve` | Print the fully resolved job config and exit. |
| `--multirun` or `-m` | Run a sweep over comma-separated override values, for example `training.lr=1e-4,5e-4`. |
| `--config-name NAME` | Compose a different config filename from `configs/`. |
| `--config-path PATH` | Use a different config directory. |

Quote list overrides so the shell does not reinterpret them:

```bash
python train.py -m training.lr=1e-4,5e-4
```

For an exact resume, keep the original data source, tokenizer, model
architecture, `training.batch_size`, `training.accum_steps`, `data.seq_len`,
and `data.max_tokens` unchanged.

For a quick CPU smoke test without downloading a dataset, write a tiny fake
tokenized cache (the trainer skips preparation when the cache is valid) and
reduce the model and data sizes:

```bash
python - <<'EOF'
import json
import numpy as np
from pathlib import Path

d = Path("data/smoke/gpt2")
d.mkdir(parents=True, exist_ok=True)
np.zeros(6336, dtype=np.uint16).tofile(d / "train.bin")
np.ones(704, dtype=np.uint16).tofile(d / "eval.bin")
(d / "meta.json").write_text(
    json.dumps(
        {"tokenizer_name": "gpt2", "hf_dataset": "smoke", "hf_config": None,
         "text_column": "text", "file_format": "parquet", "dtype": "uint16",
         "max_tokens": 7040, "train_tokens": 6336, "val_tokens": 704,
         "val_ratio": 0.1}
    )
)
EOF
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" python train.py \
  data.hf_dataset=smoke data.max_tokens=7040 \
  model.emb_dim=128 model.n_heads=4 model.n_layers=2 model.context_length=128 \
  data.seq_len=128 training.batch_size=2 training.accum_steps=1 \
  training.device=cpu training.eval_interval=20 \
  training.eval_batches=2 training.max_new_tokens=2
```

The run trains the whole one-pass budget (24 optimizer steps on this cache)
and writes `final_model.pt` at the end.

The fake corpus is intentionally small; keep the reduced sequence length.
Tokenized caches and generated experiment outputs are local-only and ignored
by Git.

### Data preparation

`train.py` tokenizes `data.hf_dataset` before training: it downloads the
dataset snapshot, reads the configured text column with PyArrow, encodes it
with `data.tokenizer_name`, appends an end-of-text token after every example,
and writes the first `data.max_tokens` tokens to
`data/<dataset>[__<config>]/<tokenizer_name>/` — `train.bin`, `eval.bin` (the
`data.val_ratio` split), and a `meta.json` recording the dataset identity,
source revision, tokenizer, split counts, validation ratio, and storage dtype
(`uint16` when the vocabulary fits, else `uint32`). A cache is reused only when
its metadata, file sizes, split geometry, and requested capacity are compatible
with the run; a missing, stale, or undersized cache is re-prepared.

Preparation writes sidecar files and installs them atomically. When an
additive phase needs a larger cache, the old training prefix is compared with
the regenerated prefix before replacement, so a changed dataset revision cannot
silently invalidate a saved training cursor.

The trainer scans the files sequentially through `np.memmap` (one data pass
is one scan of `train.bin`), so neither preparation nor training materializes
the corpus in memory.

## Configuration

Hydra composes the defaults in `configs/config.yaml` from three groups:

| Group | Default | Alternatives | Purpose |
| --- | --- | --- | --- |
| `data` | `default` | — | Hugging Face dataset, tokenizer, and tokenized cache |
| `model` | `gpt2` | `qwen`, `moe` | Model dimensions and planned architecture variants |
| `training` | `default` | — | Optimizer, evaluation, and run settings |

Values can be changed from the command line without editing YAML:

```bash
python train.py model=qwen training.batch_size=8 data.seq_len=512 training.device=cpu
```

Hydra writes run output beneath `runs/YYYY-MM-DD/HH-MM-SS/`. A validation run
can write `best.pt` there; periodic checkpoints are enabled with
`training.save_interval=<steps>`.

### Stop and resume training

Checkpoints written by the current trainer contain the model, optimizer,
learning-rate scheduler, Python/NumPy/PyTorch/CUDA RNG states, the global
optimizer step, the memmap data cursor (the next batch of the current data
pass), token counter, W&B run ID, and the WSD trigger state. Checkpoint
writes use a temporary file plus an atomic rename, so an interrupted write
cannot leave a partially written `latest.pt`.

Enable periodic restart checkpoints and resume with:

```bash
python train.py training.save_interval=500
python train.py training.resume_from=runs/2026-08-19/12-00-00/latest.pt
```

You can also use `training.resume_from=latest` to select the newest
`latest.pt` below the project directory. Pressing Ctrl-C, or receiving a
graceful SIGTERM/SIGHUP, finishes the current optimizer step and writes
`latest.pt`; the next command continues from the next batch and keeps the same
W&B run. Keep the original data, tokenizer, model, batch size, and sequence
length for a bit-for-bit continuation. An exact resume keeps the saved phase
endpoint; a stable continuation or manual decay may request an additional
token budget and will expand/reprocess the cache if the train prefix is too
short. A hard power loss or `kill -9` can only resume from the most recent
periodic checkpoint, so choose `save_interval` according to the amount of work
you are willing to repeat.

To add stable plateau training after a finished or interrupted run, use an
explicit additional token budget:

```bash
python train.py training.resume_from=latest \
  training.resume_mode=continue training.continue_tokens=100000000
```

This mode is only for a stable checkpoint. It does not start a new warmup or
decay; a later `start_decay=true` run uses the extended step timeline. If the
continuation is interrupted, resuming it with the same command finishes the
saved endpoint without adding `continue_tokens` a second time.

Checkpoints from the old streaming trainer are not resumable with this
trainer — they carry no memmap data cursor; resume raises with a clear error.
Start a fresh run instead.

Each run directory also holds a `run_manifest.json` (written in terminal mode
as well) mapping every checkpoint file to the W&B run id/URL with per-save
step, tokens seen, stage, best val loss, and timestamp, plus the run name,
git commit, and a config digest at the top level. Use it to go from a W&B run
page to the local files (top-level `wandb_run_id` / `wandb_url`), and from a
local checkpoint file back to its run page (the `wandb_run_id` / `wandb_url`
of its manifest entry).

### Two-stage WSD training (warmup → stable → decay)

The schedule is WSD: a linear warmup (`warmup_fraction` of the selected
phase-1 budget), a flat plateau at `lr` (the stable phase), and a final cosine
decay to `min_lr` over the explicit `training.decay_tokens` budget.

**Stage 1** — train warmup + stable over the data file. Stop early
with Ctrl-C (the trainer writes `latest.pt` after the current optimizer step)
or let it run the whole pass:

**Stage 2** — resume from `latest.pt`: either keep the stable plateau
(extending pretraining in any number of increments) or start the final decay:

```bash
python train.py training.resume_from=latest \
  training.resume_mode=continue training.continue_tokens=1b
python train.py training.resume_from=latest training.start_decay=true \
  training.decay_tokens=1b
```

With `start_decay=true`, the decay starts at the checkpoint's current step and
rounds `training.decay_tokens` down to complete optimizer steps. The run
**stops exactly when that decay budget finishes** — no steps are wasted at the
floor. For example, with 64K tokens per optimizer step,
`training.decay_tokens=1m` produces 15 complete decay steps (960K effective
tokens).

- Stage 2 continues stage 1's W&B run (the checkpoint carries the run id), so
  both phases appear as one continuous loss curve. The run page records the
  stage (`training_stage`: `stage-1-stable` / `stage-2-stable` /
  `stage-2-decay`) and, for decay runs, the trigger step; the W&B-computed
  fields include the requested/effective phase budget and cache capacity.
- The terminal output of a decay run prints the explicit budget
  (`budget=wsd_decay_tokens`), its effective complete-step token count, and the
  current learning rate on every progress line, so local output matches W&B.
- A crash mid-decay resumes correctly: the trigger state is saved inside the
  checkpoint, and a resumed decay continues from the original trigger step
  instead of restarting the decay.
- When `training.upload_artifacts=true`, `best.pt` and `final_model.pt` are
  uploaded to the W&B run as artifacts (metadata: kind, stage, step, tokens
  seen, best val loss, git commit, local path, W&B run id). Artifact upload
  failures only warn — they never abort training. The default is `false`.
  All local checkpoints, including periodic `latest.pt` / `step_<N>.pt` saves,
  are still written locally regardless of this setting.

## Data behavior

The current pipeline performs the following steps:

1. Download a Hugging Face dataset snapshot at `data.revision` (or reuse the
   cached source revision during an expansion).
2. Read the configured text column in PyArrow batches, tokenize with
   `data.tokenizer_name`, and append one end-of-text token per source row.
3. Write exactly `data.max_tokens` tokens into temporary `train.bin` and
   `eval.bin` files. The train/eval sizes are derived from `data.val_ratio`, so
   `floor((1 - val_ratio) * max_tokens)` tokens go to train and the remainder
   go to validation. A source row may cross that boundary; its tokens are not
   dropped.
4. Validate metadata and byte sizes, then atomically install the cache.
5. Read the cache sequentially with `MemmapDataLoader`. Each batch contains
   contiguous windows; targets are the same tokens shifted by one position.
   The final lookahead token needed for the shift is taken from the next cache
   position, and incomplete final batches are dropped.

The training loader has one cursor per data pass. Evaluation uses fresh shadow
loaders, so validation never consumes or rewinds the training cursor. A phase
budget caps how many complete optimizer steps may be taken from the train file;
it does not create a second copy of the data.

## Outputs

- W&B metrics include training/validation loss, perplexity, learning rate,
  throughput, generated text samples, and model parameter counts. When W&B is
  the backend and `training.upload_artifacts=true`, `best.pt` and
  `final_model.pt` are additionally uploaded to the run page as artifacts.
- Terminal logging includes model parameter counts, training/evaluation loss,
  learning rate, and generated sample text.
- Hydra stores run metadata and checkpoints under `runs/`. Each run directory
  also holds `run_manifest.json`, mapping every checkpoint file to the W&B
  run id/URL with per-save stats (see **Two-stage WSD training**).
- `runs/`, `wandb/`, and `checkpoints/` are ignored by Git.

### Chat UI

Start the Chainlit chat interface from the project root:

```bash
chainlit run chat.py
```

The UI discovers checkpoints under `runs/`, lists them newest first with model
metadata, loads the tokenizer recorded in each checkpoint, and exposes model,
temperature, top-k, and maximum-new-token settings. It also applies the saved
attention, BF16, and Tensor Core settings during inference. `torch.compile` is
not enabled automatically for chat because generation changes sequence length
at every token; it should be treated as an explicit performance experiment
after the model is loaded.

## Tests

```bash
pytest
```

The test suite covers causal attention, model-config validation, normalization
and FFN dispatch, cache-budget parsing and validation, memmap cursor behavior,
checkpoint state, the training schedule (including the WSD two-stage budget,
resume consistency, the run manifest, and W&B artifacts), and end-to-end
training-loop behaviors driven on a stub-constructed trainer without GPU, W&B,
or a downloaded corpus.

## Roadmap

- Expand the GPT-2-style transformer stack
- Add RoPE and wire the Qwen preset to its intended positional encoding
- Add sparse Mixture-of-Experts routing
- Add a model factory for the Qwen and MoE configurations
- Add richer evaluation controls
- Expand unit and integration coverage

## License

This project is licensed under the terms in [LICENSE](LICENSE).
