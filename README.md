# mini-llm-lab

A small, from-scratch laboratory for learning how decoder-only language models
work. The project is intentionally compact: it exposes the data pipeline,
attention implementation, model configuration, and training loop
without hiding the interesting parts behind a large framework.

This is a learning and experimentation repository, not a production training
system or a package of pretrained models.

## What is implemented

- GPT-style next-token datasets built from sliding windows
- Local text-file and Hugging Face dataset inputs
- GPT-2 tokenization through `tiktoken`
- Deterministic train/validation splitting
- Causal multi-head self-attention in PyTorch
- A GPT-2-style transformer model and training loop
- W&B metrics, text samples, and Hydra-run checkpoints
- Hydra configuration groups for GPT-2, Qwen-style, and Mixture-of-Experts
  experiments

The GPT-2-style model is currently wired into `train.py`. The Qwen-style and
Mixture-of-Experts files describe planned variants; all three configs currently
instantiate the same `GptModel` implementation with different dimensions.

## Project layout

```text
mini-llm-lab/
├── configs/                 # Composable Hydra configuration
│   ├── data/                # Local-file and Hugging Face inputs
│   ├── model/               # GPT-2, Qwen-style, and MoE settings
│   └── training/            # Optimizer and run settings
├── src/
│   ├── data/                # Tokenization, datasets, and dataloaders
│   ├── models/              # GPT model and transformer components
│   └── training/            # Training loop and checkpointing
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

The default data config uses the small *The Verdict* corpus at
`data/the_verdict`. If it is missing, the loader downloads it automatically.
For a larger local corpus, put UTF-8 text files under `data/` and override the
`data.files` setting.

```text
data/
├── book-one.txt
└── book-two.txt
```

Select those files with a Hydra override:

```bash
python train.py 'data.files=[data/book-one.txt,data/book-two.txt]'
```

For a normal GPU run:

```bash
python train.py
```

The trainer tokenizes the corpus, splits it 90/10, creates overlapping
next-token windows, and trains the GPT-style model. W&B logs online by default;
use `WANDB_MODE=offline` for local logging or `WANDB_MODE=disabled` to skip W&B.
You can choose terminal logging directly with `training.log_backend=terminal`.
In terminal mode, training/evaluation losses and generated sample text are
printed to the terminal:

```bash
python train.py training.log_backend=terminal
```

The default is `training.log_backend=wandb`; `WANDB_MODE=disabled` switches to
terminal logging automatically.

Training budgets can be expressed directly in optimizer steps or tokens. A
token budget is rounded down to complete batches, and the dataloader is cycled
when the budget spans multiple passes through the data. `epochs` remains as a
backwards-compatible fallback when neither budget is set:

```bash
python train.py training.max_tokens=100000000
python train.py training.max_steps=10000
```

The default learning-rate schedule warms up for `training.warmup_steps`, then
cosine-decays to `training.min_lr` by 90% of the selected budget and holds that
floor for the final 10%. Set `training.lr_decay_end_fraction` to change the
endpoint. `training.max_steps` and `training.max_tokens` are mutually
exclusive.

### `train.py` command reference

`train.py` is a Hydra application. Its arguments are configuration overrides in
`key=value` form, rather than argparse-style flags:

```text
python train.py [config-group=value] [section.parameter=value] ...
```

The main config groups are:

| Override | Choices | Default | Selects |
| --- | --- | --- | --- |
| `data=...` | `default`, `hf_dataset`, `gutenberg` | `default` | Corpus source and dataloader settings |
| `model=...` | `gpt2`, `qwen`, `moe` | `gpt2` | Model configuration file |
| `training=...` | `default` | `default` | Optimizer, schedule, logging, and run settings |

For example:

```bash
python train.py data=hf_dataset model=gpt2 training.device=cuda
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
| `data.source` | `files` | Input type: `files` for local text/glob patterns or `hf_dataset` for a Hugging Face dataset. If `data.tokenized_dir` is set, the token cache takes precedence. |
| `data.files` | `[data/the_verdict]` | Local UTF-8 files or glob patterns. Relative paths are resolved from the directory where `train.py` was launched, not Hydra's run directory. If no file matches, the small *The Verdict* sample is downloaded as a fallback. |
| `data.max_files` | `null` | Maximum number of raw files to read, after sorted glob expansion. Useful for limiting the Gutenberg preset. Ignored when using a tokenized cache. |
| `data.tokenized_dir` | `null` | Directory containing `manifest.json` and binary token shards created by `python -m data.token_shards`. The cache must use the same tokenizer as `data.tokenizer_name`. |
| `data.hf_dataset` | `null` | Hugging Face dataset repository name, for example `Salesforce/wikitext`. Used when `data.source=hf_dataset`. |
| `data.hf_config` | `null` | Optional dataset configuration or subset, for example `wikitext-2-raw-v1`. |
| `data.text_column` | `text` | Field read from each Hugging Face example. List-valued fields are joined with spaces. |
| `data.split` | `train` | Hugging Face dataset split to load. This is the source split; the trainer still creates its own train/validation split from the resulting token stream. |
| `data.streaming` | `false` | Ask `datasets` for a streaming split. The current loader still concatenates the streamed text in memory before tokenization, so this is not an end-to-end streaming trainer. |
| `data.tokenizer_name` | `gpt2` | `tiktoken` encoding name. The model's `vocab_size` must be compatible with the selected tokenizer. |
| `data.seq_len` | `256` | Number of input tokens in each training example. Each target is the same window shifted one token to the right. Must not exceed `model.context_length`. |
| `data.stride` | `128` | Distance between the starts of consecutive windows. `seq_len` gives adjacent windows; smaller values increase overlap. Must be positive. |
| `data.batch_size` | `2` | Number of windows per optimizer step. The effective tokens per step are `batch_size * seq_len`. |
| `data.val_ratio` | `0.1` | Fraction of the token stream reserved for validation. The split is deterministic and positional: the first portion is training data and the final portion is validation data. |
| `data.shuffle` | `true` | Shuffle training windows with a deterministic epoch-dependent sampler. Validation windows are never shuffled. |
| `data.drop_last` | `true` | Drop an incomplete training batch. Set `false` to retain it, but the effective token count per step then varies for the final batch of a data pass. |
| `data.num_workers` | `4` | PyTorch dataloader worker processes. Use `0` for the simplest CPU debugging and smoke-test behavior. |
| `data.pin_memory` | `true` | Pin host batches for faster CPU-to-CUDA transfers when training on a GPU. |
| `data.persistent_workers` | `true` | Keep dataloader workers alive between passes. It is automatically disabled when `data.num_workers=0`. |
| `data.seed` | `42` | Seed used for dataloader generators and deterministic training-window order. `training.seed` controls the model and global random streams. |

Preset-specific data defaults are `seq_len=1024`, `stride=512`, and
`batch_size=4` for both `data=hf_dataset` and `data=gutenberg`. The Gutenberg
preset additionally defaults to `data.max_files=200`; use a token cache for
the full corpus.

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
| `training.max_steps` | `null` | Explicit optimizer-step budget. Takes precedence over `epochs`; it cannot be set together with `training.max_tokens`. |
| `training.max_tokens` | `null` | Explicit token budget. It is converted to complete optimizer steps using `data.batch_size * data.seq_len`, rounded down, and cannot be set together with `training.max_steps`. |
| `training.epochs` | `10` | Backwards-compatible fallback when neither `max_steps` nor `max_tokens` is set. One epoch means one complete pass through the training dataloader. |
| `training.lr` | `5e-4` | AdamW peak learning rate. |
| `training.weight_decay` | `0.1` | AdamW weight decay. |
| `training.warmup_steps` | `500` | Number of initial optimizer steps in the linear learning-rate warmup. Must be non-negative and less than the total step budget. |
| `training.min_lr` | `5e-5` | Learning-rate floor after cosine decay. It must be positive and no greater than `training.lr`. |
| `training.lr_decay_end_fraction` | `0.9` | Fraction of the total budget at which cosine decay reaches `min_lr`; the floor is held for the remainder. Must be in `(0, 1]`. |
| `training.max_grad_norm` | `1.0` | Maximum gradient norm for global gradient clipping. |
| `training.eval_interval` | `200` | Run evaluation and generate a sample every this many optimizer steps. The best validation checkpoint is updated when validation loss improves. |
| `training.eval_batches` | `50` | Maximum number of training and validation batches used for each evaluation. Set it lower for quick experiments; evaluation uses model-eval mode. |
| `training.log_interval` | `10` | Print and log training loss, learning rate, throughput, and progress every this many optimizer steps. |
| `training.save_interval` | `null` | Optional periodic restart interval in optimizer steps. It writes both `step_<N>.pt` and an updated `latest.pt`. |
| `training.resume_from` | `null` | Checkpoint path to resume, relative to the launch directory, or `latest`/`auto` to select the newest `latest.pt` below the project. |
| `training.seed` | `42` | Global Python, NumPy, PyTorch, and CUDA seed. It is also stored in checkpoints for reproducible continuation. |
| `training.start_context` | `Every effort moves you` | Prompt used when generating the periodic text sample. |
| `training.max_new_tokens` | `50` | Maximum number of tokens appended to `start_context` for each sample. |

The schedule is linear warmup followed by cosine decay from `lr` to `min_lr`.
The selected budget determines the schedule length, so changing a budget also
changes the step at which the learning rate reaches its floor.

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
| `model.norm` | `layernorm` | Normalization label in the config. The current `GptModel` implementation uses its LayerNorm implementation regardless of this label. |
| `model.activation` | `gelu` | Activation label in the config. The current feed-forward block uses GELU; other labels are not yet dispatched. |
| `model.position_embedding` | `absolute` | Position-encoding label. The active GPT implementation uses learned absolute positional embeddings; RoPE is not yet dispatched. |
| `model.residual_style` | `serial` | Residual-layout label. The active transformer block uses serial pre-norm residual connections. |
| `model.ff_mult` | `4` | Feed-forward hidden width multiplier: `emb_dim * ff_mult`. |
| `model.logit_softcap` | `null` | Reserved config field; not currently applied by the GPT implementation. |
| `model.rope_theta` | `null` | Reserved RoPE parameter; not used while absolute positional embeddings are active. |
| `model.temperature` | `1.0` | Temperature for periodic sample generation. Positive values sample from the distribution; `0` or a negative value uses greedy argmax. |
| `model.top_k` | `25` | Restrict periodic sample generation to the top K logits. Set `null` to disable top-k filtering. |

The `moe` preset also contains `num_experts`,
`num_experts_per_token`, and `shared_expert` fields. They describe planned
Mixture-of-Experts behavior but are not consumed by the current `GptModel`.
Likewise, the Qwen-style labels (`rmsnorm`, `swiglu`, and `rope`) are config
metadata rather than active implementations. Use `model=gpt2` for the
supported end-to-end training path until a model factory is added.

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
python train.py 'data.files=[data/book-one.txt,data/book-two.txt]'
python train.py -m training.lr=1e-4,5e-4 training.max_steps=1000
```

Do not set both `training.max_steps` and `training.max_tokens`. For an exact
resume, keep the original data source, tokenizer, model architecture, batch
size, sequence length, stride, `drop_last` setting, and total budget unchanged.

For a quick CPU smoke test, reduce the model and data sizes:

```bash
WANDB_MODE=disabled python train.py \
  model.emb_dim=128 model.n_heads=4 model.n_layers=2 model.context_length=128 \
  data.seq_len=128 data.stride=64 data.batch_size=2 data.num_workers=0 \
  training.device=cpu training.max_steps=20 training.eval_interval=20 \
  training.eval_batches=2 training.max_new_tokens=2
```

The sample corpus is intentionally small, so use a reduced sequence length.
Training data files and generated experiment outputs are local-only and ignored
by Git.

### Use a Hugging Face dataset

The included preset loads WikiText-2:

```bash
python train.py data=hf_dataset data.num_workers=0
```

Any compatible dataset can be selected with Hydra overrides. Its examples must
contain the configured text column:

```bash
python train.py \
  data=hf_dataset \
  data.hf_dataset=Salesforce/wikitext \
  data.hf_config=wikitext-2-raw-v1 \
  data.text_column=text
```

Dataset downloads require internet access. The current streaming option avoids
the Hugging Face download materialization step, but still concatenates all text
in memory before tokenization.

For the Project Gutenberg preset, cap the number of books while experimenting:

```bash
python train.py data=gutenberg data.max_files=10 data.num_workers=0
```

### Train from the full Gutenberg corpus

The raw-file loader concatenates its inputs in memory and is intended for
small experiments. For the full Gutenberg corpus, first build a reusable,
memory-mappable token cache. The builder reads one book at a time and writes
binary token shards; it does not create a large in-memory token tensor:

```bash
python -m data.token_shards \
  --input data/gutenberg/data/text \
  --output data/gutenberg/tokenized \
  --tokenizer gpt2 \
  --shard-tokens 256000000 \
  --workers 8
```

Workers tokenize different books in parallel; the parent process alone writes
the shards, so the output remains deterministic. Choose `--workers` according
to the available CPU cores and storage bandwidth.

Then point training at that cache. The loader keeps random window shuffling,
but maps token shards from disk and only materializes each batch:

```bash
python train.py \
  data=gutenberg \
  data.tokenized_dir=data/gutenberg/tokenized \
  data.seq_len=2048 \
  data.stride=2048 \
  data.batch_size=8 \
  training.max_tokens=3200000000
```

The manifest stores the tokenizer, vocabulary, split counts, source files,
and shard sizes. Do not use `data.max_files` when training from a complete
cache; the cache already defines the corpus.

## Configuration

Hydra composes the defaults in `configs/config.yaml` from three groups:

| Group | Default | Alternatives | Purpose |
| --- | --- | --- | --- |
| `data` | `default` | `hf_dataset`, `gutenberg` | Corpus, tokenizer, windows, and loaders |
| `model` | `gpt2` | `qwen`, `moe` | Model dimensions and planned architecture variants |
| `training` | `default` | — | Optimizer, evaluation, and run settings |

Values can be changed from the command line without editing YAML:

```bash
python train.py model=qwen data.batch_size=8 data.seq_len=512 training.device=cpu
```

Hydra writes run output beneath `runs/YYYY-MM-DD/HH-MM-SS/`. A validation run
can write `best.pt` there; periodic checkpoints are enabled with
`training.save_interval=<steps>`.

### Stop and resume training

Checkpoints written by the current trainer contain the model, optimizer,
learning-rate scheduler, Python/NumPy/PyTorch/CUDA RNG states, deterministic
data cursor, token counter, and W&B run ID. Checkpoint writes use a temporary
file plus an atomic rename, so an interrupted write cannot leave a partially
written `latest.pt`.

Enable periodic restart checkpoints and resume with:

```bash
python train.py training.save_interval=500
python train.py training.resume_from=runs/2026-08-19/12-00-00/latest.pt
```

You can also use `training.resume_from=latest` to select the newest
`latest.pt` below the project directory. Pressing Ctrl-C, or receiving a
graceful SIGTERM/SIGHUP, finishes the current optimizer step and writes
`latest.pt`; the next command continues from the next batch and keeps the same
W&B run. Keep the original data, model, batch size, sequence length, stride,
and training budget for a bit-for-bit continuation. A hard power loss or
`kill -9` can only resume from the most recent periodic checkpoint, so choose
`save_interval` according to the amount of work you are willing to repeat.

Checkpoints from the older trainer can still load their model and optimizer,
but they do not contain the scheduler, RNG, data cursor, or W&B ID and are
therefore not exact-resume checkpoints.

## Data behavior

The pipeline performs the following steps:

1. Read and concatenate local files, or load a Hugging Face dataset.
2. Encode the text with the configured `tiktoken` tokenizer.
3. Split the token stream by position into training and validation portions.
4. Produce `(input, target)` pairs, where the target is the input shifted by one
   token.
5. Batch the windows with PyTorch `DataLoader`s.

`stride` controls overlap. A stride equal to `seq_len` creates adjacent windows;
a smaller stride creates overlapping examples. The corpus must be large enough
for at least one full window in both splits and, when `drop_last=true`, at least
one complete training batch.

## Outputs

- W&B metrics include training/validation loss, perplexity, learning rate,
  throughput, generated text samples, and model parameter counts.
- Terminal logging includes model parameter counts, training/evaluation loss,
  and generated sample text.
- Hydra stores run metadata and checkpoints under `runs/`.
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

The test suite covers causal attention, model-config validation, token-shard
datasets, checkpoint state, and the training schedule.

## Roadmap

- Expand the GPT-2-style transformer stack
- Add Qwen-style grouped-query attention, RoPE, RMSNorm, and SwiGLU
- Add sparse Mixture-of-Experts routing
- Add a model factory for the Qwen and MoE configurations
- Add richer evaluation controls
- Expand unit and integration coverage

## License

This project is licensed under the terms in [LICENSE](LICENSE).
