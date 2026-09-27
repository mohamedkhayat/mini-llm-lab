# AGENTS.md — mini-llm-lab

A small, from-scratch lab for learning how decoder-only LLMs work. It is **not** a
production training system or a pretrained-model package, and the code is small enough
to read whole. Two properties shape every session: it is a **learning project**, and the
agent has a specific dual role (below). `README.md` is the detailed reference (parameter
tables, data recipes, checkpoint/resume) — this file is the orientation.

## Your role here (read this first)

Pick the mode from the user's request; when it is ambiguous, ask one question rather
than guessing.

1. **Teach** — the user wants to _understand_ something ("teach me X", "tutor", "explain /
   walk me through", or they send past work to be corrected). Use the **`tutor` skill**
   (load it) and follow its STRICT state machine. **Never write the solution for them.**
2. **Boilerplate / utility** — the user wants _plumbing built for them_ (data loaders,
   config options, checkpointing, token sharding, chat UI, attention/FFN backends, test
   scaffolds). Implement it, keep it consistent with the codebase, don't over-engineer.
   The user owns the "interesting" ML parts.

## Tutoring mode — follow the `tutor` skill (do this extensively)

Load the `tutor` skill and follow it exactly; it is the canonical procedure, not advice.
Facts that are easy to miss:

- **State machine:** `START → TRY → TEACH → PRACTICE → QUIZ → LOG`, one state per reply,
  tag every reply `[TUTOR:<STATE>]`, move forward only. (Early exit straight to `LOG` if
  the user says "log it / that's enough".)
- **Never solve for the learner.** ≤10 lines of scaffolding, point at gaps with questions,
  reveal a line only as the last hint rung. The only exception is a learner-confirmed
  `[TUTOR:OVERRIDE]`.
- **Exactly one question per reply**, then stop.
- **Persist & resume across sessions.** Files live in the project root unless noted:
  - `TUTOR_STATE.md` — one line, the resume pointer. Re-read it at session start; if it
    exists you are **resuming** (skip START). Keep it when the lesson is intentionally
    parked; delete it only at a real `LOG` close.
  - `LEARNING_LOG.md` — append strict one-line entries
    (`topic/band/quiz/help/weak/reflect/next-review … by: slm`) plus `pref-observation`
    lines and free-form narrative. This is the durable record to **re-read before re-teaching**.
  - `~/learning-vault/inbox.md` — append-only session summaries (one line, ends `| by: slm`).
- **Validate before claiming "Logged":** from the project root run
  `python ~/.config/opencode/skills/tutor/scripts/check.py`; fix any `FAIL` lines and
  re-run until it prints `PASS`.
- **Known learner preferences** (recorded in `LEARNING_LOG.md`; honor these): wants
  **hints/guidance, not solutions**; gets lost in dense theory/jargon and anchors best on
  **concrete code + small runnable examples**; likes to **sanity-check their own
  reasoning** first; keeps the main thread resumable across side questions.

`LEARNING_LOG.md` and `TUTOR_STATE.md` are tutor working files, **not** project code —
never delete/rewrite them mid-lesson, and don't commit them as code.

## Commands

Python **3.12+**; there is a venv at `.venv/`.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'        # runtime deps + pytest

python train.py                          # GPU run (W&B online by default)
pytest                                   # full test suite
pytest tests/test_training_schedule.py -k test_name   # one test
python -m data.token_shards --input <in> --output <out> --tokenizer gpt2   # build a token cache
chainlit run chat.py                     # chat UI over saved checkpoints
```

Fast CPU smoke test before any real run (no GPU / no W&B):

```bash
WANDB_MODE=disabled python train.py \
  model.emb_dim=128 model.n_heads=4 model.n_layers=2 model.context_length=128 \
  data.seq_len=128 data.stride=64 data.batch_size=2 data.num_workers=0 \
  training.device=cpu training.max_steps=20 training.eval_interval=20 \
  training.eval_batches=2 training.max_new_tokens=2
```

## Non-obvious things that will bite you

- **Imports are rooted at `src/`, not the repo root.** `pyproject.toml` sets
  `pythonpath=["src"]`, so it is `import training.trainer`, `import data.dataloader`,
  `import models.gpt` — **never** `import src.…`. Same for `-m`: `python -m data.token_shards`
  (not `python -m src.data.token_shards`).
- **`train.py` is a Hydra app, not argparse.** Arguments are `key=value` overrides
  (`python train.py model=qwen training.max_steps=1000`); **quote list overrides**
  (`'data.files=[a.txt,b.txt]'`). Inspect the composed config with
  `python train.py --cfg job --resolve`. Config groups: `data` (default / hf_dataset /
  gutenberg), `model` (gpt2 / qwen / moe), `training` (default). Run output goes to
  `runs/<date>/<time>/`.
- **Only `GptModel` is wired.** There is no model factory — the trainer always builds
  `GptModel` (`src/training/trainer.py`). `model=qwen` / `model=moe` still construct
  `GptModel`; they change dimensions plus the normalization/activation/FFN knobs, not
  the architecture class.
- **Which `model.*` labels are real and which are metadata:**
  - **Real (dispatched in `src/models/blocks/transformerBlock.py` / `gpt.py` / `ffn.py`):**
    `model.attention` → `mha` | `fastmha` | `gqa` — all three are implemented,
    including `GroupQueryAttention` in `src/models/attention/gqa.py`.
    `model.normalization` → `layer_norm` | `rms_norm` — lookup is case- and
    underscore-insensitive (`layernorm`, `LAYER_NORM` …); a config without the key
    defaults to layer norm (older runs stay resumable); unknown names raise with the
    valid choices. `rms_norm` constructs real `RMSNorm` blocks (`src/models/normalization/`),
    which the qwen/moe presets select.
    `model.activation` → `gelu` | `silu` | `sigmoid` and `model.gated` — both are
    dispatched in `src/models/feed_forward/ffn.py`. `gated: true` adds the second GLU
    projection, so `gated: true` + `activation: silu` is SwiGLU (with `gelu` it is
    GEGLU); `equalize_params: true` scales `hidden_dim` by 2/3 when gated to keep the
    parameter count comparable.
  - **Metadata only (the label is ignored by the code):** `model.position_embedding` /
    `model.rope_theta` (always learned **absolute** embeddings, no RoPE),
    `model.residual_style`, `model.name`. So `model=qwen` gives you RMSNorm and
    SwiGLU-style gated FFN but **not** RoPE — that is roadmap, not code.
  - MoE routing is not wired; the `moe` preset's `num_experts*` fields are unused by
    `GptModel`.
- **W&B is on by default.** `WANDB_MODE=offline` for local logging,
  `WANDB_MODE=disabled` (or `training.log_backend=terminal`) to print to the terminal.
- **Budgets are mutually exclusive:** set `training.max_steps` **or** `training.max_tokens`
  (never both); `training.epochs` is the fallback when neither is set.
- **What is validated, and where** — do not assume setup-time checks for the rest:
  `training.min_lr ≤ training.lr` (and > 0) and warmup ≥ 0 are enforced in
  `training/schedule.py` on every run; decay runs run with `warmup = 0`.
  `0 < training.lr_decay_fraction < 1` is enforced on decay runs only (a fresh
  stable run does not check the range). `data.seq_len ≤ model.context_length` is not
  checked at setup — it raises on the first forward (`GptModel.forward`). The
  divisibility checks (`model.emb_dim % model.n_heads == 0`; for GQA
  `model.n_heads % model.n_kv_heads == 0`) are `assert`s in the attention
  constructors and vanish under `python -O`. `model.vocab_size` is **not** checked
  against `data.tokenizer_name` anywhere — a mismatch surfaces as a tensor-size
  error at the first forward.

## Checkpoints & resume

The trainer writes under the run dir: `latest.pt` (moving resume/recovery pointer,
atomic), `best.pt` (best validation loss), `step_<N>.pt` (periodic, when
`training.save_interval` is set), and `final_model.pt` (only when a run completes). A
checkpoint carries model + optimizer + LR scheduler + RNG + data cursor + token counter

- W&B run id + the WSD trigger state (`wsd`). Each run dir also holds
  `run_manifest.json` (written in terminal mode too), mapping every checkpoint file to
  the W&B run id/URL with step/tokens/stage/val-loss/timestamp — the local↔W&B lookup
  (see README, "Two-stage WSD training").

```bash
python train.py training.save_interval=500                 # periodic restart checkpoints
python train.py training.resume_from=latest                # or a full path
```

`Ctrl-C` / `SIGINT` / `SIGTERM` finish the current optimizer step and write `latest.pt`.
For a bit-for-bit resume, keep the data source, tokenizer, model dims, batch size,
`seq_len`, `stride`, and total budget unchanged.

## Adding an attention backend (boilerplate pattern)

Implement a module under `src/models/attention/` with the 7-arg constructor
`(d_in, d_out, context_length, dropout, num_heads, num_kv_groups, qkv_bias)` and a
`forward(x)`, then register it in the `attention_impls` dict in
`src/models/blocks/transformerBlock.py`. That is all the wiring a new attention
implementation needs.

## What's local-only (git-ignored)

`/data/` (corpora, incl. the auto-downloaded `the_verdict` sample and `gutenberg`),
`runs/`, `wandb/`, `checkpoints/`, and `.venv/` are ignored. Real **code** is **not**
ignored and should be committed — e.g. `src/models/attention/gqa.py`,
`src/data/token_shards.py`, `chat.py`, and `tests/test_*.py`. The working tree is a WIP
lab with an evolving set of uncommitted changes — run `git status` before large edits.

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature-slug>/` in this repo.
See `docs/agents/issue-tracker.md`.

### Triage labels

The default five-label vocabulary is used as-is (`needs-triage`, `needs-info`,
`ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one root `CONTEXT.md` + `docs/adr/` (neither exists yet; they are
created lazily when terms or decisions get resolved). See `docs/agents/domain.md`.
