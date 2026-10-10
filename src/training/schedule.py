"""The training schedule: pure, stateless budget and learning-rate math.

This module answers "what LR does this run use" in one place:

- ``resolve_phase_budget`` — the explicit stable/decay endpoint for a run.
- ``build_lr_lambda`` — the LR curve handed to the scheduler: linear warmup,
  then a plateau until the decay triggers, then a cosine decay to the floor.
- ``stage_label`` — the run's stage marker (e.g. ``stage-2-decay``).

Resumability policy (single source of truth): a checkpoint is resumable only
if its ``wsd`` entry carries a ``target_step`` key — the decision-rich phase
budget (``PhaseBudget.as_dict()`` plus the ``triggered``/``step``
compatibility aliases). Checkpoints written by older trainers (a ``wsd``
dict with only ``triggered``/``step``, or no ``wsd`` key at all) are NOT
resumable; ``_saved_phase_budget`` raises a clear ``ValueError`` naming the
missing key. (This matches ``training.checkpointing``: old checkpoints are
not resumable, start a fresh run instead.)
"""

import math
from dataclasses import dataclass, replace

from data.budget import complete_optimizer_steps, parse_token_budget
from training.checkpointing import check_resume_consistency


@dataclass(frozen=True)
class PhaseBudget:
    """A resolved phase endpoint in optimizer-step units."""

    mode: str
    stage: str
    start_step: int
    target_step: int | None
    requested_tokens: int | None
    effective_tokens: int | None
    decay_start_step: int | None = None
    decay_steps: int | None = None
    # F9: None is the "unresolved" sentinel. resolve_budget fills it for a
    # fresh stable start (from warmup_fraction) and the exact_resume branch
    # fills it from the saved wsd entry; explicit values (including 0) are
    # respected. as_dict persists None as null; consumers treat a restored
    # None as 0.
    warmup_steps: int | None = None

    def as_dict(self) -> dict:
        """Return the checkpoint/logging representation (None warmup
        persists as null)."""
        return {
            "mode": self.mode,
            "stage": self.stage,
            "start_step": self.start_step,
            "target_step": self.target_step,
            "requested_tokens": self.requested_tokens,
            "effective_tokens": self.effective_tokens,
            "decay_start_step": self.decay_start_step,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
        }


def _phase_state(budget: PhaseBudget) -> dict:
    """Add the old WSD aliases to a new phase state for compatibility."""
    state = budget.as_dict()
    state.update(
        {
            "triggered": budget.stage == "decay",
            "step": budget.decay_start_step,
        }
    )
    return state


def phase_state(budget: PhaseBudget) -> dict:
    """Return the persisted WSD state for a resolved phase budget."""
    return _phase_state(budget)


def _saved_step(checkpoint: dict | None) -> int:
    return int(checkpoint.get("step", 0)) if checkpoint is not None else 0


def _saved_phase_budget(
    checkpoint: dict | None,
    tokens_per_step: int,
    warmup_fraction: float = 0.0,
) -> PhaseBudget | None:
    """Recover an explicit phase endpoint from a saved checkpoint.

    The saved ``wsd`` entry must carry a ``target_step`` key (the phase
    budget written by the current trainer); legacy entries with only the
    ``triggered``/``step`` aliases are not resumable and raise. A fresh run
    (``checkpoint is None``) returns None; a one-pass checkpoint
    (``target_step is None``) returns None so the exact-resume fallback uses
    the saved ``total_steps``.

    ``warmup_steps`` is restored from the saved entry; old-format entries
    without the key fall back to the historical rule (recompute from
    ``warmup_fraction * total_steps`` for a fresh stable start, else 0).
    """
    if checkpoint is None:
        return None

    saved_step = _saved_step(checkpoint)
    saved_wsd = checkpoint.get("wsd")
    if not isinstance(saved_wsd, dict):
        saved_wsd = {}

    if "target_step" not in saved_wsd:
        raise ValueError(
            "The checkpoint's wsd entry does not carry a target_step value; "
            "it was written by an older trainer and is not resumable with "
            "this trainer. Start a fresh run instead."
        )

    target_step = saved_wsd.get("target_step")
    if isinstance(target_step, int) and target_step >= saved_step:
        stage = str(saved_wsd.get("stage", "stable"))
        if stage not in {"stable", "decay"}:
            raise ValueError(
                "The checkpoint carries an unknown WSD stage; cannot resume "
                f"safely: {stage!r}"
            )
        start_step = int(saved_wsd.get("start_step", saved_step if stage == "decay" else 0))
        requested_tokens = saved_wsd.get("requested_tokens")
        effective_tokens = saved_wsd.get("effective_tokens")
        if effective_tokens is None:
            effective_tokens = (target_step - start_step) * tokens_per_step
        if requested_tokens is None:
            requested_tokens = effective_tokens
        decay_start = saved_wsd.get("decay_start_step")
        decay_steps = saved_wsd.get("decay_steps")
        if stage == "decay":
            decay_start = int(decay_start if decay_start is not None else saved_step)
            decay_steps = int(
                decay_steps
                if decay_steps is not None
                else target_step - decay_start
            )
        # F9: an explicit null (the unresolved sentinel persisted by
        # as_dict) counts as absent and falls back to the historical rule.
        if saved_wsd.get("warmup_steps") is not None:
            warmup_steps = int(saved_wsd["warmup_steps"])
        else:
            # Old-format checkpoint: the historical rule — warmup only on a
            # fresh stable start (stage stable, start step 0).
            is_fresh_stable_start = stage == "stable" and start_step == 0
            warmup_steps = (
                int(warmup_fraction * int(target_step))
                if is_fresh_stable_start
                else 0
            )
        return PhaseBudget(
            mode=str(saved_wsd.get("mode", "checkpoint")),
            stage=stage,
            start_step=start_step,
            target_step=int(target_step),
            requested_tokens=int(requested_tokens),
            effective_tokens=int(effective_tokens),
            decay_start_step=decay_start,
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
        )
    return None


def resolve_phase_budget(training_cfg, checkpoint, tokens_per_step: int) -> PhaseBudget:
    """Resolve the manual stable/decay endpoint for a run.

    A saved unfinished phase always wins.  A stable checkpoint can be
    manually switched to decay, and a completed decay can be followed by a
    new explicit decay request.  Otherwise a resume can request either an
    additional stable budget or an additional decay budget.  A fresh run can
    optionally set ``train_tokens``; when omitted, ``target_step=None`` tells
    the trainer to use the available cached train capacity.
    """
    tokens_per_step = int(tokens_per_step)
    if tokens_per_step <= 0:
        raise ValueError(f"tokens_per_step must be positive; got {tokens_per_step}")

    resume_step = _saved_step(checkpoint)
    resume_mode = str(getattr(training_cfg, "resume_mode", "exact"))
    if resume_mode not in {"exact", "continue"}:
        raise ValueError("training.resume_mode must be 'exact' or 'continue'")

    start_decay = bool(getattr(training_cfg, "start_decay", False))
    saved_budget = _saved_phase_budget(
        checkpoint,
        tokens_per_step,
        warmup_fraction=float(getattr(training_cfg, "warmup_fraction", 0.0)),
    )
    if saved_budget is not None:
        # An active decay is authoritative: re-running the launcher after a
        # crash must continue the original cosine endpoint.  A saved stable
        # phase, however, is intentionally overridable by the manual
        # start_decay trigger.
        decay_is_active = (
            saved_budget.stage == "decay"
            and resume_step < (saved_budget.target_step or 0)
        )
        if decay_is_active or not start_decay:
            # An unfinished stable continuation resumes its saved endpoint.
            # Once that endpoint is complete, resume_mode=continue
            # intentionally starts a new additive stable extension.
            can_start_new_stable_extension = (
                saved_budget.stage == "stable"
                and resume_mode == "continue"
                and resume_step >= (saved_budget.target_step or 0)
            )
            if not can_start_new_stable_extension:
                return saved_budget

    if start_decay:
        if checkpoint is None:
            raise ValueError("training.start_decay requires training.resume_from")
        if resume_mode == "continue":
            raise ValueError(
                "Stable continuation cannot start decay; use "
                "training.resume_mode=exact with training.start_decay=true."
            )
        steps, effective = complete_optimizer_steps(
            getattr(training_cfg, "decay_tokens", None),
            tokens_per_step,
            "training.decay_tokens",
        )
        budget = PhaseBudget(
            mode="decay_tokens",
            stage="decay",
            start_step=resume_step,
            target_step=resume_step + steps,
            requested_tokens=parse_token_budget(
                getattr(training_cfg, "decay_tokens"),
                "training.decay_tokens",
                allow_none=False,
            ),
            effective_tokens=effective,
            decay_start_step=resume_step,
            decay_steps=steps,
        )
        return budget

    continue_tokens = getattr(training_cfg, "continue_tokens", None)
    if continue_tokens is not None:
        if checkpoint is None or resume_mode != "continue":
            raise ValueError(
                "training.continue_tokens requires "
                "training.resume_mode=continue with training.resume_from."
            )
        steps, effective = complete_optimizer_steps(
            continue_tokens,
            tokens_per_step,
            "training.continue_tokens",
        )
        return PhaseBudget(
            mode="stable_continue",
            stage="stable",
            start_step=resume_step,
            target_step=resume_step + steps,
            requested_tokens=parse_token_budget(
                continue_tokens, "training.continue_tokens", allow_none=False
            ),
            effective_tokens=effective,
        )

    if checkpoint is not None:
        saved_total = checkpoint.get("total_steps")
        if saved_total is not None:
            saved_total = int(saved_total)
            # F9: this branch is the one-pass default (the saved phase had
            # target_step None), so use the PERSISTED warmup — the LR curve
            # must survive later warmup_fraction edits. An absent key
            # (old-format checkpoint) falls back to the historical rule:
            # fresh stable start -> fraction * total.
            saved_wsd = checkpoint.get("wsd")
            if not isinstance(saved_wsd, dict):
                saved_wsd = {}
            saved_warmup = saved_wsd.get("warmup_steps")
            if saved_warmup is not None:
                warmup_steps = int(saved_warmup)
            else:
                warmup_steps = int(
                    float(getattr(training_cfg, "warmup_fraction", 0.0))
                    * saved_total
                )
            return PhaseBudget(
                mode="exact_resume",
                stage="stable",
                start_step=0,
                target_step=saved_total,
                requested_tokens=saved_total * tokens_per_step,
                effective_tokens=saved_total * tokens_per_step,
                warmup_steps=warmup_steps,
            )

    train_tokens = getattr(training_cfg, "train_tokens", None)
    if train_tokens is not None:
        steps, effective = complete_optimizer_steps(
            train_tokens,
            tokens_per_step,
            "training.train_tokens",
        )
        return PhaseBudget(
            mode="train_tokens",
            stage="stable",
            start_step=0,
            target_step=steps,
            requested_tokens=parse_token_budget(
                train_tokens, "training.train_tokens", allow_none=False
            ),
            effective_tokens=effective,
        )

    return PhaseBudget(
        mode="one_pass",
        stage="stable",
        start_step=0,
        target_step=None,
        requested_tokens=None,
        effective_tokens=None,
    )


def build_lr_lambda(warmup_steps, peak_lr, min_lr, phase: PhaseBudget):
    """Linear warmup, then — for a decay phase — a cosine decay from
    ``phase.decay_start_step`` over ``phase.decay_steps`` down to min_lr.
    Reads the phase directly (no wsd-state dict, no ``triggered``/``step``
    aliases). A None warmup (the unresolved sentinel, F9) means no warmup."""
    warmup_steps = int(warmup_steps if warmup_steps is not None else 0)
    peak_lr = float(peak_lr)
    min_lr = float(min_lr)

    if warmup_steps < 0:
        raise ValueError(f"warmup steps must be >= 0; got {warmup_steps}")
    if not 0.0 < min_lr <= peak_lr:
        raise ValueError(
            f"training.min_lr must be greater than 0 and at most training.lr; "
            f"got min_lr={min_lr}, lr={peak_lr}"
        )

    min_scale = min_lr / peak_lr
    decay = phase.stage == "decay"
    decay_start = phase.decay_start_step
    decay_steps = max(1, int(phase.decay_steps)) if phase.decay_steps else 1

    def lr_lambda(step):
        # Warm up linearly, then decay to min_lr by the decay end and hold.
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        if decay and decay_start is not None:
            progress = min(max(step - decay_start, 0) / decay_steps, 1.0)
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_scale + (1.0 - min_scale) * cosine_scale
        return 1.0

    return lr_lambda


@dataclass(frozen=True)
class Budget:
    """The run's resolved step/token budget (spec 5.9): built once by the
    assembly (the trainer constructor until T6.2); replaces the per-attribute
    budget state on the trainer."""

    phase: PhaseBudget
    total_steps: int
    steps_per_pass: int
    available_steps_per_pass: int
    tokens_per_step: int
    budget_name: str
    continuation_run: bool
    decay_run: bool
    total_train_tokens: int


def stage_label(resumed, decay_triggered):
    """Return the W&B stage marker for the run (e.g. ``stage-2-decay``)."""
    stage = "stage-2" if resumed else "stage-1"
    return f"{stage}-{'decay' if decay_triggered else 'stable'}"


_BUDGET_NAMES = {
    "one_pass": "one_pass",
    "train_tokens": "train_tokens",
    "stable_continue": "continue_tokens",
    "decay_tokens": "wsd_decay_tokens",
    "legacy_decay": "wsd_decay_legacy",
    "exact_resume": "exact_resume",
}


def resolve_budget(
    training_cfg,
    checkpoint: dict | None,
    *,
    steps_per_pass: int,
    available_steps_per_pass: int,
    tokens_per_step: int,
    phase: PhaseBudget | None = None,
) -> Budget:
    """Resolve the run's full Budget (spec 5.9) and validate resume
    geometry — pure, no trainer, no I/O (T6.3: moved out of
    ``Trainer._resolve_budget`` so assembly can validate BEFORE any
    logging/W&B side effect, P3).

    ``phase`` may be pre-resolved by the caller (the loader needs the
    endpoint before it is built); without it, it is resolved here.
    Warmup is filled once for fresh stable starts and persisted on the
    phase, so the resume LR curve is independent of later
    ``warmup_fraction`` edits; a namespace without the key means no warmup.
    """
    resume_mode = str(getattr(training_cfg, "resume_mode", "exact"))
    if resume_mode not in {"exact", "continue"}:
        raise ValueError("training.resume_mode must be 'exact' or 'continue'")

    phase = phase or resolve_phase_budget(training_cfg, checkpoint, tokens_per_step)
    continuation_run = phase.mode == "stable_continue"
    decay_run = phase.stage == "decay"

    total_steps = (
        phase.target_step if phase.target_step is not None else steps_per_pass
    )
    budget_name = _BUDGET_NAMES.get(phase.mode, phase.mode)

    # F9: fill the sentinel only — an explicit value (including 0) is
    # respected; phases that keep None (decay, continuations) persist null
    # and resume as warmup 0.
    if (
        phase.stage == "stable"
        and phase.start_step == 0
        and phase.warmup_steps is None
    ):
        phase = replace(
            phase,
            warmup_steps=int(
                float(getattr(training_cfg, "warmup_fraction", 0.0)) * total_steps
            ),
        )

    budget = Budget(
        phase=phase,
        total_steps=total_steps,
        steps_per_pass=steps_per_pass,
        available_steps_per_pass=available_steps_per_pass,
        tokens_per_step=tokens_per_step,
        budget_name=budget_name,
        continuation_run=continuation_run,
        decay_run=decay_run,
        total_train_tokens=total_steps * tokens_per_step,
    )

    if checkpoint is not None:
        check_resume_consistency(
            saved_total_steps=checkpoint.get("total_steps"),
            total_steps=total_steps,
            saved_steps_per_pass=checkpoint.get("steps_per_pass"),
            steps_per_pass=steps_per_pass,
            saved_tokens_per_step=checkpoint.get("tokens_per_step"),
            tokens_per_step=tokens_per_step,
            decay_run=decay_run,
            continuation_run=continuation_run,
            allow_capacity_growth=continuation_run or decay_run,
            available_steps_per_pass=available_steps_per_pass,
        )

    if budget.total_steps > steps_per_pass:
        raise ValueError(
            f"The training budget ({budget.total_steps:,} steps) exceeds one "
            f"pass over the data file ({steps_per_pass:,} steps). The launcher "
            "should have expanded the cache before loader construction; check "
            "data.max_tokens and the phase budget, then resume from the last "
            "checkpoint."
        )

    return budget
