"""The training schedule: pure, stateless budget and learning-rate math.

This module answers "what budget and what LR does this run use" in one place:

- ``resolve_total_steps`` — total-step budget: ``max_steps`` or ``max_tokens``
  win; ``epochs`` is the backwards-compatible fallback; both step and token
  budgets set is an error.
- ``resolve_wsd_state`` — the live WSD trigger state for a (possibly resumed)
  run. A valid triggered state saved in a checkpoint wins over a fresh
  ``start_decay`` flag; legacy checkpoints without a ``wsd`` entry fall back
  to the flag.
- ``resolve_decay_budget`` — the stage-2 decay budget derived from the
  trigger step (rule of three); the fraction must be in ``(0, 1)``.
- ``build_lr_lambda`` — the LR curve handed to the scheduler: linear warmup,
  then a plateau until the decay triggers, then a cosine decay to the floor.
- ``stage_label`` — the run's stage marker (e.g. ``stage-2-decay``).

WSD state format (frozen for on-disk compatibility):
    {"triggered": bool, "step": int | None}
A plain dict under the ``"wsd"`` checkpoint key. ``step`` is the trigger step
once ``triggered`` is true, else ``None``. Legacy checkpoints may omit the key
entirely; both are tolerated on resume. The format never changes on disk, so
old checkpoints keep loading unchanged.
"""

import math


def resolve_total_steps(training_cfg, steps_per_epoch, tokens_per_step):
    """Resolve the training budget, preferring steps/tokens over epochs.

    ``epochs`` remains as a backwards-compatible fallback. Token budgets are
    rounded down to complete optimizer batches so training never exceeds the
    requested budget.
    """
    max_steps = getattr(training_cfg, "max_steps", None)
    max_tokens = getattr(training_cfg, "max_tokens", None)

    if max_steps is not None and max_tokens is not None:
        raise ValueError("Set only one of training.max_steps and training.max_tokens.")

    if max_steps is not None:
        total_steps = int(max_steps)
        budget_name = "max_steps"
    elif max_tokens is not None:
        requested_tokens = int(max_tokens)
        total_steps = requested_tokens // int(tokens_per_step)
        budget_name = "max_tokens"
    else:
        epochs = int(getattr(training_cfg, "epochs", 1))
        total_steps = epochs * int(steps_per_epoch)
        budget_name = "epochs"

    if total_steps <= 0:
        raise ValueError(
            "Training budget must produce at least one optimizer step; "
            "increase training.max_tokens/max_steps or training.epochs."
        )

    return total_steps, budget_name


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
    if saved.get("triggered") is True and isinstance(saved_step, int) and saved_step >= 0:
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