# Spec — WSD two-stage decay + checkpoint/run visibility

Status: done (2026-08-23, committed on `main` — see ## Comments for acceptance evidence and KNOWN items)
Source: /home/mohamed/.opencode/plan/wsd-two-stage-spec.md (2026-08-23, post WSD tutoring session)

## Problem Statement

Training with a learning-rate schedule that must know the total step count up front doesn't fit
the real workflow: train stable "until I say stop", then (optionally keep pretraining), then run
a final decay whose length depends on where I stopped. Today the trainer's schedule decays over a
fraction of a pre-known total, a stage-2 decay run cannot even resume (budget-mismatch guard
raises), and after a run there is no easy way to see which local checkpoint files belong to which
W&B run, with what hyperparameters, stages, and token counts.

## Solution

A two-stage WSD workflow fully supported by the trainer and configs, plus run-visibility
plumbing:

- **Stage 1**: any normal run (warmup + flat stable plateau at peak LR). Stop early with Ctrl-C
  (writes `latest.pt`) or run out the chosen budget.
- **Stage 2**: resume from `latest.pt`. With `start_decay` off → keeps stable (extends
  pretraining). With `start_decay` on → decays from the resume step over
  `D = round(S·f/(1−f))` steps (the final `f` of the whole run) and **stops exactly when the
  decay finishes**.
- W&B: one continuous run across stages, with a visible stage marker and correct (non-stale)
  computed-budget values; `best.pt` and `final_model.pt` uploaded as W&B artifacts with
  identifying metadata.
- Local: every run dir carries a manifest tying each checkpoint file to the W&B run id/URL, step,
  tokens seen, stage, and val loss; README documents the workflow and how to go from a W&B run
  page to the local files (and back).

## User Stories

1. As a trainer, I want stage 1 to run warmup + stable against a budget I choose, so that I can
   train as long as I want without the schedule knowing the total.
2. As a trainer, I can stop stage 1 early at any step (Ctrl-C) and pick it up, so that
   "when to stop" is a decision I make, not a number I commit to up front.
3. As a trainer, I want a stage-2 run to start decaying from the step I resumed at, so that decay
   never depends on knowing the total steps ahead of time.
4. As a trainer, I want the decay to be the final `lr_decay_fraction` of the whole run (decay
   length derived from where I stopped), so that the decay is proportionally right no matter
   where stage 1 stopped.
5. As a trainer, I want the decay run to stop exactly when the decay completes, so that no steps
   are wasted sitting at the floor LR.
6. As a trainer, I can resume and keep training stable (no decay) as many times as I want, so
   that pretraining can be extended in any number of increments before the final decay.
7. As a trainer, I want stage 2 to continue in the same W&B run as stage 1, so the two phases
   appear as one continuous experiment with one loss curve.
8. As a trainer, I want the W&B run page to say which stage the run is in, so I can tell
   stable-phase from decay-phase visually.
9. As a trainer, I want the W&B-computed budget values (total steps, total tokens) to reflect the
   decay run's real (derived) budget, so the UI never shows stale numbers.
10. As a trainer, I want the resume budget-mismatch guard to accept decay runs (and still reject
    everything else that disagrees), so I can resume with a different effective budget without a
    false error.
11. As a trainer, I want the decay trigger state saved inside the checkpoint, so that a crashed
    decay run resumes mid-decay from the original trigger step instead of restarting the decay.
12. As a trainer, I want the terminal output of a decay run to show the derived budget, so local
    output matches W&B.
13. As a trainer, I want `best.pt` and `final_model.pt` uploaded as W&B artifacts with metadata
    (stage, step, tokens seen, val loss, git commit), so the checkpoints I care about are visible
    on the run page in W&B.
14. As a trainer, I want a manifest in each local run dir mapping every checkpoint file to the
    W&B run id/URL with per-checkpoint stats, so I can find the files for the run I'm looking at
    in W&B without guessing.
15. As a trainer, I want the reverse lookup too (from a local file to its W&B run), so I can open
    the right run page from the files on disk.
16. As a trainer, I want the README to document the two-stage workflow and the local↔W&B lookup,
    so future-me and agents can run it correctly.
17. As a maintainer, I want the schedule tests to assert the new WSD behavior (plateau, decay
    over D, floor hold, warmup=0 legal, invalid decay fraction rejected), so regressions in the
    scheduler are caught.
18. As a trainer, I want decay runs to be impossible to misconfigure into a zero/negative decay
    (fraction must be in (0,1)), so bad configs fail fast with a clear message.

## Implementation Decisions

- The existing single `build_lr_lambda` (warmup → plateau / triggered cosine decay) is kept; the
  mutable `wsd` state dict remains the live holder, seeded after checkpoint restore (existing,
  already-correct ordering). No new scheduler class.
- Relative budget helper: `D = round(S·f/(1−f))`, `total = S + D`, computed only for decay runs;
  validated `0 < f < 1`. The existing `lr_decay_fraction` knob keeps its name; its documented
  meaning becomes "fraction of the final run occupied by the decay".
- The override happens after the `wsd` seed and **before** W&B init, so the W&B config values
  (`computed_total_steps`, `computed_total_train_tokens`) are fresh; the terminal budget printout
  and the token-budget variable are re-derived and re-printed after the override.
- Decay runs run with `warmup = 0`; the warmup validation is fixed to reject only negative values
  (currently `not warmup_steps` rejects 0 while the message says ">= 0").
- The resume total-steps mismatch check is bypassed **only** for decay runs (flag or checkpoint
  `wsd` state indicates a triggered decay); the steps-per-epoch equality check always applies
  (stage 2 must keep identical data settings).
- Checkpoint payload gains the `wsd` state dict; loaders tolerate its absence (legacy
  checkpoints); on resume the checkpoint's saved `wsd` state takes precedence over a fresh flag
  seed.
- W&B stage marker: a config update recording the stage (e.g. "stage-1-stable" /
  "stage-2-decay") and, for decay runs, the trigger step. Terminal/disabled mode is untouched.
- W&B artifacts: created only at `best.pt` save and final completion; metadata includes kind,
  stage, step, tokens seen, best val loss, git commit, local path, and the W&B run id. Artifact
  failures must never abort training (warn and continue). Fully skipped when W&B is off.
- Run manifest: a JSON file in the run dir, atomically updated on every checkpoint save; per
  checkpoint entry records file, kind (latest/best/final/periodic), step, tokens seen, stage,
  best val loss, timestamp; top level records run name, W&B run id/URL, git commit, config
  digest. Written even in terminal mode (local-only utility).
- `final_model.pt` semantics unchanged (written on any clean completion); the nuance that a
  stage-1 run exhausting its budget without decaying also writes it is accepted as-is.
- README: new section for the two-stage WSD workflow (stage-1 / stage-2 commands, what
  `lr_decay_fraction` now means, crash-resume behavior) and the local↔W&B lookup via the
  manifest.

## Testing Decisions

- Good tests here assert external behavior of the schedule as a function of (step, wsd state) and
  of the budget helper as a pure function — not internal wiring. The trainer-level pieces are
  covered by the documented two-stage CPU smoke run (acceptance evidence) plus targeted unit tests
  where a pure function exists.
- `tests/test_training_schedule.py` is rewritten: plateau before trigger, cosine decay over
  exactly `D` steps reaching the floor, clamped hold after, warmup ramp unchanged, `warmup=0`
  accepted, `f >= 1` / `f <= 0` rejected.
- New unit tests for the relative-budget helper (including rounding and the `S = 0` degenerate
  case).
- Manifest updater and artifact-metadata builder are pure/testable: test entries-in → JSON-out and
  metadata contents; artifact creation tested against a stubbed wandb; terminal mode asserts no
  wandb interaction.
- Prior art: the existing schedule test file and the general `tests/` conventions.

## Out of Scope

- MLflow or any second tracking stack; W&B artifact upload of periodic `latest.pt` saves
  (best/final only, by decision); RoPE/RMSNorm/real model factory (roadmap, unchanged); MoE
  routing; multi-context domain-doc layout; GitHub/GitLab issue trackers (local markdown only);
  the separate outstanding FFN-flags test ticket from the 2026-08-23 GLU session (tracked
  alongside, not part of this spec's acceptance).

## Further Notes

- `runs/` currently holds dated Hydra run dirs plus stray top-level `latest.pt` /
  `final_model.pt` from the non-Hydra fallback dir; the manifest goes in whichever run dir is
  active per run. The strays are pre-existing and untouched by this work.
- The 2026-08-20 parked decision (RELATIVE vs ABSOLUTE + fixed-D vs fraction) is resolved by this
  spec: **relative**, fraction-of-final-run via the rule of three.

## Acceptance evidence (from spec §2, §4)

- `tests/test_training_schedule.py` rewritten to the WSD behavior (plateau, decay over D, floor
  hold, warmup=0 legal, invalid fraction rejected).
- Unit tests for the relative-budget helper (rounding, `f` validation, `S = 0` degenerate case).
- Unit tests for the manifest updater and artifact-metadata builder (stubbed wandb; terminal mode
  asserts no wandb interaction).
- Two-stage CPU smoke (manual, documented): tiny model, `max_steps=60`, stage 1 stopped ~step 30,
  stage 2 `resume_from=latest start_decay=true` — terminal `lr` column must decay over
  `D=8` steps (S=30, f=0.2) and the run must END at the end of the decay.

## Comments

- 2026-08-23 — Implemented (committed on `main`, subject "WSD two-stage decay +
  checkpoint/run visibility").
  Evidence per acceptance behavior:
  - Schedule/budget behavior: `tests/test_training_schedule.py` (22 tests) —
    warmup ramp, plateau, triggered cosine decay over exactly D steps to the
    floor, clamped hold, warmup=0 legal, `0 < f < 1` rejection, `resolve_decay_budget`
    rounding / `S = 0` degenerate case.
  - Mismatch guard (US10): `tests/test_wsd_resume.py` (`check_resume_consistency`:
    decay-run bypass, steps-per-epoch always enforced, legacy tolerance) + smoke
    (stage 2 resumes with `max_steps=100` vs saved 60, no raise).
  - Crash-mid-decay (US11): `tests/test_wsd_resume.py` (`resolve_wsd_state`
    precedence: checkpoint triggered state > fresh flag) + `tests/test_wsd_checkpoint.py`
    (wsd state in payload, round-trip) + smoke (SIGINT mid-decay at step 34; resume
    continues from trigger step 30 — lr resumes at 1.61e-04, not peak — and ends at 38).
  - Stage marker / computed budget (US8-9, US12): `stage_label` tests; smoke
    terminal shows `WSD decay: trigger step S=30, decay steps D=8`,
    `Optimizer steps: 38 (…; budget=wsd_decay)` and the per-step `lr` column;
    W&B `config.update` (training_stage / wsd_trigger_step / computed_*) is wired
    before `wandb.init`'s first log (terminal mode verified in smoke; unit test
    `test_terminal_mode_performs_no_wandb_interaction` pins no-wandb gating).
  - Artifacts (US13): `tests/test_wandb_artifacts.py` (metadata contents;
    stubbed-wandb create/log failure never raises; terminal mode no-op gate).
  - Manifest (US14-15): `tests/test_run_manifest.py` (entries-in → JSON-out,
    upsert/reuse by file, corrupt-file recovery, atomic write) + smoke run dir
    `runs/<date>/<time>/run_manifest.json` with per-entry stage/tokens/val-loss and
    top-level W&B id/URL (null in terminal mode).
  - README (US16): "Two-stage WSD training" section, parameter table
    (`warmup_fraction`/`min_lr`/`lr_decay_fraction`/`start_decay`), Outputs,
    stop/resume + manifest lookup.
  - Acceptance smoke (spec §2.4), terminal mode, CPU:
    `WANDB_MODE=disabled python train.py model.emb_dim=128 model.n_heads=4
    model.n_layers=2 model.context_length=128 data.seq_len=128 data.stride=64
    data.batch_size=2 data.num_workers=0 training.device=cpu training.max_steps=60
    training.eval_interval=20 training.eval_batches=2 training.max_new_tokens=2
    training.log_interval=1` → SIGINT after step 30; then same dims +
    `training.max_steps=100 training.eval_interval=10000 training.resume_from=latest
    training.start_decay=true` → decays over D=8 steps (S=30, f=0.2), ends at 38.
    (runs/ is git-ignored, so transcripts live outside the repo.)
  KNOWN (no code change; per spec or pre-existing):
  - US6 "extend pretraining in increments" works via early-stop (Ctrl-C)
    increments; a stage-1 run that exhausted its budget cannot be extended with a
    larger stable budget (the spec explicitly bypasses the guard ONLY for decay
    runs). Extending an exhausted run would require re-deciding that.
  - Spec §"warmup fix" premise (`warmup_steps` knob / `not warmup_steps` bug)
    was written against the pre-WIP code; the WIP baseline had already renamed it
    to `training.warmup_fraction`. The validation fix itself (reject only `w < 0`,
    decay runs run `warmup = 0`) is implemented and tested.
  - `min_lr` default (`1e-5`) / strict-positive validation and the
    `start_context` default were changed by earlier WIP, not this ticket.
  - 3 pre-existing baseline test failures (test_mha constructor arity,
    test_model_config `n_kv_heads`) are GQA-related test drift, out of scope.
  - `AGENTS.md` (untracked, user-maintained) was updated in place (invariants,
    checkpoint/manifest description, Agent-skills block) and intentionally left
    uncommitted.
  - The FFN-flags test ticket (`.scratch/ffn-flags-tests/`) is tracked alongside
    per build step 2 and is not part of this spec's acceptance.