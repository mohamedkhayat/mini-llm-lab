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

For a quick CPU smoke test, reduce the model and data sizes:

```bash
WANDB_MODE=disabled python train.py \
  model.emb_dim=128 model.n_heads=4 model.n_layers=2 model.context_length=128 \
  data.seq_len=128 data.stride=64 data.batch_size=2 data.num_workers=0 \
  training.device=cpu training.epochs=1 training.eval_interval=20 \
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
  throughput, and generated text samples.
- Hydra stores run metadata and checkpoints under `runs/`.
- `runs/`, `wandb/`, and `checkpoints/` are ignored by Git.

## Tests

```bash
pytest
```

The current test checks the causal multi-head attention module's output shape,
finiteness, and batch consistency.

## Roadmap

- Expand the GPT-2-style transformer stack
- Add Qwen-style grouped-query attention, RoPE, RMSNorm, and SwiGLU
- Add sparse Mixture-of-Experts routing
- Add a model factory for the Qwen and MoE configurations
- Add checkpoint resume and richer evaluation controls
- Expand unit and integration coverage

## License

This project is licensed under the terms in [LICENSE](LICENSE).
