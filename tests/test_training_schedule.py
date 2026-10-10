import math
from types import SimpleNamespace

import pytest

from training.log_backend import resolve_log_backend
from training.schedule import (
    PhaseBudget,
    build_lr_lambda,
    resolve_budget,
    resolve_phase_budget,
)


def _stable(**overrides):
    fields = dict(
        mode="one_pass",
        stage="stable",
        start_step=0,
        target_step=None,
        requested_tokens=None,
        effective_tokens=None,
    )
    fields.update(overrides)
    return PhaseBudget(**fields)


def _decay(start, d, **overrides):
    fields = dict(
        mode="decay_tokens",
        stage="decay",
        start_step=start,
        target_step=start + d,
        requested_tokens=None,
        effective_tokens=None,
        decay_start_step=start,
        decay_steps=d,
    )
    fields.update(overrides)
    return PhaseBudget(**fields)


def test_log_backend_can_be_selected_for_terminal_output():
    assert resolve_log_backend(SimpleNamespace(log_backend="terminal")) == "terminal"
    assert resolve_log_backend(SimpleNamespace(log_backend="term")) == "terminal"
    assert resolve_log_backend(SimpleNamespace(log_backend="wandb")) == "wandb"


def test_disabled_wandb_mode_falls_back_to_terminal():
    cfg = SimpleNamespace(log_backend="wandb")

    assert resolve_log_backend(cfg, wandb_mode="disabled") == "terminal"


def test_unknown_log_backend_is_rejected():
    with pytest.raises(ValueError, match="log_backend"):
        resolve_log_backend(SimpleNamespace(log_backend="tensorboard"))


def test_legacy_checkpoint_without_target_step_is_rejected():
    # Legacy wsd entries (only the triggered/step aliases) carry no
    # target_step; per the single resumability policy they are not
    # resumable — the error names the missing key.
    legacy = {"step": 30, "total_steps": 38, "wsd": {"triggered": True, "step": 30}}

    with pytest.raises(ValueError, match="target_step"):
        resolve_phase_budget(
            SimpleNamespace(resume_mode="exact", start_decay=False),
            legacy,
            tokens_per_step=64,
        )


def test_exact_budget_is_one_pass_over_the_data_file():
    # With no explicit phase budget, the run still defaults to one pass over
    # the prepared train cache (T6.3: the budget math is the pure
    # ``resolve_budget`` function, not a Trainer method).
    budget = resolve_budget(
        SimpleNamespace(
            resume_mode="exact",
            continue_tokens=None,
            start_decay=False,
            train_tokens=None,
        ),
        None,
        steps_per_pass=42,
        available_steps_per_pass=42,
        tokens_per_step=32,
    )

    assert budget.total_steps == 42
    assert budget.budget_name == "one_pass"
    assert budget.total_train_tokens == 42 * 32


def test_active_decay_checkpoint_wins_over_a_new_decay_request():
    cfg = SimpleNamespace(
        resume_mode="exact",
        start_decay=True,
        decay_tokens="99k",
    )
    checkpoint = {
        "step": 17,
        "total_steps": 40,
        "wsd": {
            "mode": "decay_tokens",
            "stage": "decay",
            "start_step": 12,
            "target_step": 40,
            "requested_tokens": 1_792,
            "effective_tokens": 1_792,
            "decay_start_step": 12,
            "decay_steps": 28,
        },
    }

    budget = resolve_phase_budget(cfg, checkpoint, tokens_per_step=64)

    assert budget.target_step == 40
    assert budget.decay_start_step == 12
    assert budget.decay_steps == 28
    assert budget.requested_tokens == 1_792


def test_manual_decay_overrides_a_saved_stable_endpoint():
    cfg = SimpleNamespace(
        resume_mode="exact",
        start_decay=True,
        decay_tokens="1k",
    )
    checkpoint = {
        "step": 12,
        "total_steps": 20,
        "wsd": {
            "mode": "train_tokens",
            "stage": "stable",
            "start_step": 0,
            "target_step": 20,
            "requested_tokens": 1_280,
            "effective_tokens": 1_280,
            "decay_start_step": None,
            "decay_steps": None,
        },
    }

    budget = resolve_phase_budget(cfg, checkpoint, tokens_per_step=64)

    assert budget.mode == "decay_tokens"
    assert budget.start_step == 12
    assert budget.target_step == 27
    assert budget.decay_steps == 15


def test_completed_decay_can_start_a_new_explicit_decay_budget():
    cfg = SimpleNamespace(
        resume_mode="exact",
        start_decay=True,
        decay_tokens="1k",
    )
    checkpoint = {
        "step": 25,
        "total_steps": 25,
        "wsd": {
            "mode": "decay_tokens",
            "stage": "decay",
            "start_step": 10,
            "target_step": 25,
            "requested_tokens": 960,
            "effective_tokens": 960,
            "decay_start_step": 10,
            "decay_steps": 15,
        },
    }

    budget = resolve_phase_budget(cfg, checkpoint, tokens_per_step=64)

    assert budget.start_step == 25
    assert budget.target_step == 40
    assert budget.decay_start_step == 25


def test_stable_continuation_budget_is_additive_and_suffix_aware():
    cfg = SimpleNamespace(
        resume_mode="continue",
        start_decay=False,
        continue_tokens="1k",
    )

    budget = resolve_phase_budget(
        cfg,
        {
            "step": 10,
            "wsd": {
                "mode": "train_tokens",
                "stage": "stable",
                "start_step": 0,
                "target_step": 10,
                "requested_tokens": 640,
                "effective_tokens": 640,
            },
        },
        tokens_per_step=64,
    )

    assert budget.mode == "stable_continue"
    assert budget.target_step == 25
    assert budget.effective_tokens == 960


def test_fresh_train_tokens_define_the_stable_endpoint():
    cfg = SimpleNamespace(
        resume_mode="exact",
        start_decay=False,
        train_tokens="1k",
    )

    budget = resolve_phase_budget(cfg, None, tokens_per_step=64)

    assert budget.mode == "train_tokens"
    assert budget.target_step == 15
    assert budget.effective_tokens == 960


def test_trainer_stable_continuation_extends_the_saved_step_budget():
    # T6.3: the budget math is the pure ``resolve_budget`` function — same
    # continuation geometry, no Trainer.
    budget = resolve_budget(
        SimpleNamespace(
            resume_mode="continue",
            continue_tokens=96,
            max_steps=None,
            max_tokens=None,
            start_decay=False,
        ),
        {
            "step": 7,
            "total_steps": 7,
            "steps_per_pass": 2,
            "tokens_per_step": 32,
            "wsd": {
                "mode": "train_tokens",
                "stage": "stable",
                "start_step": 0,
                "target_step": 7,
                "requested_tokens": 224,
                "effective_tokens": 224,
            },
        },
        steps_per_pass=10,  # post-expansion capacity (single-pass rule)
        available_steps_per_pass=10,
        tokens_per_step=32,
    )

    assert budget.continuation_run is True
    assert budget.total_steps == 10
    assert budget.budget_name == "continue_tokens"


def test_warmup_ramps_linearly_then_holds_the_peak_plateau():
    # Warmup occupies the first 100 steps; before the decay is triggered the
    # schedule stays flat at the peak (WSD "stable" phase).
    lr_lambda = build_lr_lambda(100, 5e-4, 5e-5, _stable())

    assert lr_lambda(0) == pytest.approx(1 / 100)
    assert lr_lambda(99) == pytest.approx(1.0)
    assert lr_lambda(100) == pytest.approx(1.0)
    assert lr_lambda(500) == pytest.approx(1.0)
    assert lr_lambda(2000) == pytest.approx(1.0)


def test_warmup_of_zero_is_accepted():
    # Decay runs resume after warmup already happened, so warmup = 0 must be
    # legal (the old `not warmup_steps` check rejected it).
    lr_lambda = build_lr_lambda(0, 5e-4, 5e-5, _stable())

    assert lr_lambda(0) == pytest.approx(1.0)
    assert lr_lambda(30) == pytest.approx(1.0)


def test_negative_warmup_is_rejected():
    with pytest.raises(ValueError, match="warmup steps"):
        build_lr_lambda(-1, 5e-4, 5e-5, _stable())


def test_min_lr_above_peak_is_rejected():
    with pytest.raises(ValueError, match="min_lr"):
        build_lr_lambda(0, 5e-4, 1e-3, _stable())


def test_triggered_decay_reaches_the_floor_after_exactly_d_steps_and_holds():
    # S = 30, D = 8, floor at 10% of peak: cosine over 8 steps from step 30.
    lr_lambda = build_lr_lambda(0, 5e-4, 5e-5, _decay(30, 8))

    # Decays start at the peak at the trigger step...
    assert lr_lambda(30) == pytest.approx(1.0)
    # ...and drop monotonically (halfway: scale = floor + 0.9 * 0.5 = 0.55).
    assert lr_lambda(34) == pytest.approx(0.55)
    assert lr_lambda(31) < 1.0
    assert lr_lambda(35) < lr_lambda(31)
    # The floor is reached exactly at the end of the decay...
    assert lr_lambda(38) == pytest.approx(0.1)
    # ...and held flat afterwards.
    assert lr_lambda(39) == pytest.approx(0.1)
    assert lr_lambda(1000) == pytest.approx(0.1)

    # Actual LR values (scale x peak) at the key points.
    assert 5e-4 * lr_lambda(31) < 5e-4
    assert 5e-4 * lr_lambda(38) == pytest.approx(5e-5)


def test_untriggered_state_holds_peak_even_past_the_trigger_step():
    lr_lambda = build_lr_lambda(0, 5e-4, 5e-5, _stable())

    assert lr_lambda(30) == pytest.approx(1.0)
    assert lr_lambda(50) == pytest.approx(1.0)


def test_decay_counter_depends_only_on_trigger_step_and_d():
    # The same trigger step and decay length must give identical decay values
    # regardless of any "total steps" notion (relative WSD budget).
    first = build_lr_lambda(0, 1e-3, 1e-4, _decay(100, 10))
    second = build_lr_lambda(0, 1e-3, 1e-4, _decay(100, 10))

    for step in (100, 105, 110, 150):
        assert first(step) == pytest.approx(second(step))
    # Sanity: the halfway value is 10% + 90% * 0.5 = 55%.
    assert first(105) == pytest.approx(0.55)


def test_wsd_cosine_shape_matches_the_reference_formula():
    # Cross-check the lambda against the closed form at several points.
    peak, floor = 5e-4, 5e-5
    s, d = 30, 8
    lr_lambda = build_lr_lambda(0, peak, floor, _decay(s, d))
    min_scale = floor / peak
    for step in range(s, s + d + 5):
        progress = min(max(step - s, 0) / d, 1.0)
        expected = min_scale + (1.0 - min_scale) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        assert lr_lambda(step) == pytest.approx(expected, rel=1e-12)
