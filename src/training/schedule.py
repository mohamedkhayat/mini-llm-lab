"""The training schedule: pure, stateless budget and learning-rate math.

This module answers "what LR does this run use" in one place:

- ``resolve_wsd_state`` — the legacy two-field WSD trigger state retained for
  older callers and checkpoints.
- ``resolve_phase_budget`` — the explicit stable/decay endpoint for a run.
- ``resolve_decay_budget`` — the legacy fraction-based helper retained only
  for checkpoints created by the previous WSD implementation.
- ``build_lr_lambda`` — the LR curve handed to the scheduler: linear warmup,
  then a plateau until the decay triggers, then a cosine decay to the floor.
- ``stage_label`` — the run's stage marker (e.g. ``stage-2-decay``).

New checkpoints store a decision-rich phase budget under the ``"wsd"`` key
and retain ``{"triggered": bool, "step": int | None}`` as compatibility
aliases. ``target_step`` and the requested/effective token counts make a
resume independent of the original command line. Legacy checkpoints may omit
the key entirely or contain only the two aliases; both are migrated by
``_saved_phase_budget``.
"""

import math
from dataclasses import dataclass

from data.budget import complete_optimizer_steps, parse_token_budget


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

    def as_dict(self) -> dict:
        """Return the checkpoint/logging representation."""
        return {
            "mode": self.mode,
            "stage": self.stage,
            "start_step": self.start_step,
            "target_step": self.target_step,
            "requested_tokens": self.requested_tokens,
            "effective_tokens": self.effective_tokens,
            "decay_start_step": self.decay_start_step,
            "decay_steps": self.decay_steps,
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
) -> PhaseBudget | None:
    """Recover an explicit phase endpoint or a legacy saved decay endpoint."""
    if checkpoint is None:
        return None

    saved_step = _saved_step(checkpoint)
    saved_wsd = checkpoint.get("wsd")
    if not isinstance(saved_wsd, dict):
        saved_wsd = {}

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
        return PhaseBudget(
            mode=str(saved_wsd.get("mode", "checkpoint")),
            stage=stage,
            start_step=start_step,
            target_step=int(target_step),
            requested_tokens=int(requested_tokens),
            effective_tokens=int(effective_tokens),
            decay_start_step=decay_start,
            decay_steps=decay_steps,
        )

    # Checkpoints written by the previous implementation have only
    # {triggered, step}; their total_steps already contains the chosen decay
    # endpoint.  Preserve that endpoint without consulting the removed config
    # fraction.
    if saved_wsd.get("triggered") is True:
        saved_total = checkpoint.get("total_steps")
        saved_trigger = saved_wsd.get("step")
        if (
            isinstance(saved_total, int)
            and isinstance(saved_trigger, int)
            and saved_total >= saved_trigger >= 0
        ):
            decay_steps = saved_total - saved_trigger
            return PhaseBudget(
                mode="legacy_decay",
                stage="decay",
                start_step=saved_trigger,
                target_step=saved_total,
                requested_tokens=decay_steps * tokens_per_step,
                effective_tokens=decay_steps * tokens_per_step,
                decay_start_step=saved_trigger,
                decay_steps=decay_steps,
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
    saved_budget = _saved_phase_budget(checkpoint, tokens_per_step)
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
            return PhaseBudget(
                mode="exact_resume",
                stage="stable",
                start_step=0,
                target_step=saved_total,
                requested_tokens=saved_total * tokens_per_step,
                effective_tokens=saved_total * tokens_per_step,
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


def resolve_continuation_steps(training_cfg, tokens_per_step):
    """Convert a stable continuation's additional token budget to steps."""
    requested_tokens = getattr(training_cfg, "continue_tokens", None)
    if requested_tokens is None:
        raise ValueError(
            "training.resume_mode=continue requires training.continue_tokens."
        )
    return complete_optimizer_steps(
        requested_tokens,
        tokens_per_step,
        "training.continue_tokens",
    )[0]


def resolve_decay_budget(steps_done, decay_fraction):
    """Derive the WSD decay budget from where stage 1 stopped.

    The decay occupies the final ``decay_fraction`` of the whole run, so for a
    trigger at step ``S`` the decay length is ``D = round(S * f / (1 - f))``
    and the total budget is ``S + D`` (rule of three). ``f`` must be in
    ``(0, 1)``; a degenerate ``S = 0`` still runs at least one decay step.
    """
    steps_done = int(steps_done)
    decay_fraction = float(decay_fraction)

    if steps_done < 0:
        raise ValueError(
            "steps_done must be >= 0 (the trigger step of the decay run); "
            f"got {steps_done}"
        )
    if not 0.0 < decay_fraction < 1.0:
        raise ValueError(
            "training.lr_decay_fraction must be in (0, 1): the decay is the "
            f"final fraction of the run; got {decay_fraction}"
        )

    decay_steps = max(1, round(steps_done * decay_fraction / (1.0 - decay_fraction)))
    return steps_done + decay_steps, decay_steps


def resolve_wsd_state(saved_wsd, start_decay, resume_step):
    """Resolve the live WSD state for a (possibly resumed) run.

    A valid triggered state saved in a checkpoint takes precedence over a
    fresh ``start_decay`` flag, so a crash mid-decay resumes from the
    original trigger step instead of restarting the decay. Legacy
    checkpoints without a ``wsd`` entry (or with an untriggered one) fall
    back to the flag: ``start_decay`` then triggers the decay at the
    resume step.
    """
    saved = saved_wsd if isinstance(saved_wsd, dict) else {}
    saved_step = saved.get("step")
    if (
        saved.get("triggered") is True
        and isinstance(saved_step, int)
        and saved_step >= 0
    ):
        return {"triggered": True, "step": saved_step}
    if start_decay:
        return {"triggered": True, "step": int(resume_step)}
    return {"triggered": False, "step": None}


def build_lr_lambda(warmup_steps, decay_steps, peak_lr, min_lr, wsd_state):
    warmup_steps = int(warmup_steps)
    decay_steps = max(1, int(decay_steps))
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

    def lr_lambda(step):
        # Warm up linearly, then decay to min_lr by decay_end_step and hold.
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        if wsd_state["triggered"] is True and wsd_state["step"] is not None:
            progress = min(max(step - wsd_state["step"], 0) / decay_steps, 1.0)
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        else:
            return 1.0
        return min_scale + (1.0 - min_scale) * cosine_scale

    return lr_lambda


def stage_label(resumed, decay_triggered):
    """Return the W&B stage marker for the run (e.g. ``stage-2-decay``)."""
    stage = "stage-2" if resumed else "stage-1"
    return f"{stage}-{'decay' if decay_triggered else 'stable'}"
