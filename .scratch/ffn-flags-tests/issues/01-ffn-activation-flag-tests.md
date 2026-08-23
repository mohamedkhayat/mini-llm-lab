# 01 — Tests for the new FFN flags (activation / gated / ffn_bias / equalize_params)

Status: ready-for-agent

Carried over from the 2026-08-23 GLU/SwiGLU session (`LEARNING_LOG.md`, `TODO.md` §2 sub-step).
The branching `FeedForward` implementation in `src/models/feed_forward/ffn.py` (plus
`src/models/feed_forward/utils.py::equalize_params` and the preset configs) is in place, but
`tests/` has no coverage for the new `model.*` flags.

## Behavior to cover

1. `equalize_params(hidden_dim, gated, do_equalize)`:
   - returns `int(2/3 · hidden_dim)` only when `gated` is true AND `do_equalize` is true;
   - returns `hidden_dim` unchanged otherwise.
2. `FeedForward` branching under flag combinations:
   - `activation: gelu, gated: false` (gpt2 preset) — plain path, no `fc2` attribute, output shape
     `[B, S, emb_dim]`;
   - `activation: silu, gated: true` (qwen/moe presets) — `fc2` exists, Hadamard product applied,
     output shape preserved;
   - `activation: sigmoid` accepted by the dispatch table;
   - an unknown `activation` label raises.
3. `ffn_bias`: bias present on all linears when true, absent when false.
4. Preset wiring: the three preset configs (`configs/model/{gpt2,qwen,moe}.yaml`) compose into a
   constructible `TransformerBlock`/`GptModel` (guards against preset keys drifting from what
   `FeedForward` reads — the failure mode from the 2026-08-23 session).

## Notes

- Reuse the small `make_model_config` fixture pattern from `tests/test_model_config.py`.
- This ticket is tracked alongside `.scratch/wsd-two-stage/spec.md` but is independent of it and
  not part of that spec's acceptance.

## Comments