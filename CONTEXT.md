# mini-llm-lab

A from-scratch decoder-only LLM lab. The training side is split into small,
single-concern modules so each concept (schedule, checkpointing, logging) can
be read in one file; this glossary pins the names those modules share.

## Language

**Schedule**:
The pure math that decides how many optimizer steps a run takes and what
learning rate each step uses: budget resolution, WSD trigger state, decay
budget, the LR lambda, and the stage label. Embodied by `training.schedule`.
_Avoid_: LR plan, budget logic, learning-rate schedule module.

**Checkpointing**:
Everything that saves, loads, resolves, or validates a restart checkpoint:
the on-disk payload format, resume-path resolution, resume-consistency
validation, and the state-restore procedures. Embodied by
`training.checkpointing`.
_Avoid_: restart, resume (a verb, not the concept), snapshot.

**Log backend**:
The resolved destination of a run's metrics and samples — W&B or the
terminal — selected by the W&B/terminal adapters behind one logger interface,
so trainer code contains no backend branches. Embodied by
`training.log_backend`.
_Avoid_: logger, logging, tracking, wandb mode.

**Training loop**:
The trainer's outer method: iterate the dataloader, move batches to the
device, run one optimizer step, then handle due logging, evaluation, and
saves at their intervals. Lives in the trainer module (`training.trainer`);
it is not a separate module.
_Avoid_: epoch loop, fit, training engine.

**Run manifest**:
The local `run_manifest.json` in a run directory mapping every checkpoint
file to its save stats and the W&B run it belongs to — the local↔W&B lookup.
Written in terminal mode as well. Embodied by `training.run_manifest`.
_Avoid_: checkpoint index, tracking file, run log.

**WSD state**:
The frozen on-disk trigger-state dict `{"triggered", "step"}` persisted in
every checkpoint so a crash mid-decay resumes from the original trigger step.
The format is owned and documented by `training.schedule`; it never changes
on disk.
_Avoid_: decay flag, wsd dict, trigger flag.