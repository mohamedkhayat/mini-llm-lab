# mini-llm-lab

A small, from-scratch laboratory for learning how decoder-only language models
work. The project is intentionally compact: it exposes the data pipeline,
attention implementation, model configuration, and (eventually) training loop
without hiding the interesting parts behind a large framework.

This is a learning and experimentation repository, not a production training
system or a package of pretrained models.

## What is implemented

- GPT-style next-token datasets built from sliding windows
- Local text-file and Hugging Face dataset inputs
- GPT-2 tokenization through `tiktoken`
- Deterministic train/validation splitting
- Causal multi-head self-attention in PyTorch
- Hydra configuration groups for GPT-2, Qwen-style, and Mixture-of-Experts
  experiments
- A smoke-test entry point that loads data and prints batch shapes

The complete transformer models, loss/optimizer loop, checkpointing, and text
generation are still under construction. At present, `train.py` validates the
configured data pipeline; it does not train a model yet. The files
`src/models/gpt.py`, `src/models/qwen.py`, and `src/models/moe.py` are model
placeholders.

## Project layout

```text
mini-llm-lab/
├── configs/                 # Composable Hydra configuration
│   ├── data/                # Local-file and Hugging Face inputs
│   ├── model/               # GPT-2, Qwen-style, and MoE settings
│   └── training/            # Future training-loop settings
├── src/
│   ├── data/                # Tokenization, datasets, and dataloaders
│   └── models/
│       └── attention/       # Causal multi-head attention
├── tests/                   # Pytest tests
└── train.py                 # Data-pipeline smoke test
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

## Try the data pipeline

Put one or more sufficiently large UTF-8 text files under `data/`:

```text
data/
├── book-one.txt
└── book-two.txt
```

Then run:

```bash
python train.py
```

By default, files matching `data/*.txt` are concatenated, tokenized, split 90/10,
and converted into overlapping sequences of 1,024 tokens. The program prints
the resolved configuration, corpus sizes, and the shape of one training batch.

For a small corpus, reduce the sequence length and worker count:

```bash
python train.py data.seq_len=128 data.stride=64 data.batch_size=2 data.num_workers=0
```

If `data/*.txt` matches nothing, the loader attempts to download the small
*The Verdict* sample corpus. Because that corpus is small, use the reduced
sequence-length command above so both the training and validation splits contain
at least one window.

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

## Configuration

Hydra composes the defaults in `configs/config.yaml` from three groups:

| Group | Default | Alternatives | Purpose |
| --- | --- | --- | --- |
| `data` | `default` | `hf_dataset` | Corpus, tokenizer, windows, and loaders |
| `model` | `gpt2` | `qwen`, `moe` | Architecture parameters for future models |
| `training` | `default` | — | Optimizer and run settings for the future loop |

Values can be changed from the command line without editing YAML:

```bash
python train.py model=qwen data.batch_size=8 data.seq_len=512 training.device=cpu
```

Hydra writes run output beneath `runs/YYYY-MM-DD/HH-MM-SS/`. Selecting a model
configuration currently changes only the printed configuration because model
construction has not yet been wired into `train.py`.

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
for at least one full window in both splits.

## Tests

```bash
pytest
```

The current test checks the causal multi-head attention module's output shape,
finiteness, and batch consistency.

## Roadmap

- Implement the GPT-2-style transformer stack
- Add Qwen-style grouped-query attention, RoPE, RMSNorm, and SwiGLU
- Add sparse Mixture-of-Experts routing
- Wire model construction into the Hydra configs
- Implement training, evaluation, checkpoints, and text generation
- Expand unit and integration coverage

## License

This project is licensed under the terms in [LICENSE](LICENSE).
