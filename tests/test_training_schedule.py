import math
from types import SimpleNamespace

import pytest

from training.log_backend import resolve_log_backend
from training.trainer import Trainer
from training.schedule import (
    build_lr_lambda,
    resolve_phase_budget,
    resolve_continuation_steps,
    resolve_decay_budget,
)


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


def test_exact_budget_is_one_pass_over_the_data_file():
    # With no explicit phase budget, the run still defaults to one pass over
    # the prepared train cache.
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(
        training=SimpleNamespace(
            resume_mode="exact",
            continue_tokens=None,
            start_decay=False,
        )
    )
    trainer.resume_checkpoint = None
    trainer.steps_per_pass = 42
    trainer.tokens_per_step = 32

    trainer._resolve_budget()

    assert trainer.total_steps == 42
    assert trainer.budget_name == "one_pass"
    assert trainer.total_train_tokens == 42 * 32


def test_stable_continuation_uses_additional_complete_optimizer_steps():
    cfg = SimpleNamespace(continue_tokens=1000)

    assert resolve_continuation_steps(cfg, tokens_per_step=32) == 31


def test_stable_continuation_requires_a_complete_optimizer_step():
    cfg = SimpleNamespace(continue_tokens=31)

    with pytest.raises(ValueError, match="complete optimizer batch"):
        resolve_continuation_steps(cfg, tokens_per_step=32)


def test_manual_decay_uses_an_explicit_additional_token_budget():
    cfg = SimpleNamespace(
        resume_mode="exact",
        start_decay=True,
        decay_tokens="1k",
    )

    budget = resolve_phase_budget(
        cfg,
        {"step": 10, "total_steps": 10, "wsd": {"triggered": False}},
        tokens_per_step=64,
    )

    assert budget.mode == "decay_tokens"
    assert budget.stage == "decay"
    assert budget.start_step == 10
    assert budget.target_step == 25
    assert budget.requested_tokens == 1_000
    assert budget.effective_tokens == 960
    assert budget.decay_steps == 15


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

    budget = resolve_phase_budget(cfg, {"step": 10}, tokens_per_step=64)

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


def test_legacy_triggered_checkpoint_keeps_its_saved_endpoint():
    cfg = SimpleNamespace(resume_mode="exact", start_decay=False)

    budget = resolve_phase_budget(
        cfg,
        {"step": 30, "total_steps": 38, "wsd": {"triggered": True, "step": 30}},
        tokens_per_step=64,
    )

    assert budget.mode == "legacy_decay"
    assert budget.target_step == 38
    assert budget.decay_steps == 8


def test_trainer_stable_continuation_extends_the_saved_step_budget():
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(
        training=SimpleNamespace(
            resume_mode="continue",
            continue_tokens=96,
            max_steps=None,
            max_tokens=None,
            start_decay=False,
        )
    )
    trainer.resume_checkpoint = {
        "step": 7,
        "total_steps": 7,
        "steps_per_pass": 2,
        "tokens_per_step": 32,
        "wsd": {"triggered": False, "step": None},
    }
    trainer.steps_per_pass = 2
    trainer.tokens_per_step = 32

    trainer._resolve_budget()

    assert trainer.continuation_run is True
    assert trainer.total_steps == 10
    assert trainer.budget_name == "continue_tokens"


def test_warmup_ramps_linearly_then_holds_the_peak_plateau():
    # Warmup occupies the first 100 steps; before the decay is triggered the
    # schedule stays flat at the peak (WSD "stable" phase).
    lr_lambda = build_lr_lambda(
        warmup_steps=100,
        decay_steps=50,
        peak_lr=5e-4,
        min_lr=5e-5,
        wsd_state={"triggered": False, "step": None},
    )

    assert lr_lambda(0) == pytest.approx(1 / 100)
    assert lr_lambda(99) == pytest.approx(1.0)
    assert lr_lambda(100) == pytest.approx(1.0)
    assert lr_lambda(500) == pytest.approx(1.0)
    assert lr_lambda(2000) == pytest.approx(1.0)


def test_warmup_of_zero_is_accepted():
    # Decay runs resume after warmup already happened, so warmup = 0 must be
    # legal (the old `not warmup_steps` check rejected it).
    lr_lambda = build_lr_lambda(
        warmup_steps=0,
        decay_steps=8,
        peak_lr=5e-4,
        min_lr=5e-5,
        wsd_state={"triggered": False, "step": None},
    )

    assert lr_lambda(0) == pytest.approx(1.0)
    assert lr_lambda(30) == pytest.approx(1.0)


def test_negative_warmup_is_rejected():
    with pytest.raises(ValueError, match="warmup steps"):
        build_lr_lambda(
            warmup_steps=-1,
            decay_steps=8,
            peak_lr=5e-4,
            min_lr=5e-5,
            wsd_state={"triggered": False, "step": None},
        )


def test_min_lr_above_peak_is_rejected():
    with pytest.raises(ValueError, match="min_lr"):
        build_lr_lambda(
            warmup_steps=0,
            decay_steps=8,
            peak_lr=5e-4,
            min_lr=1e-3,
            wsd_state={"triggered": False, "step": None},
        )


def test_triggered_decay_reaches_the_floor_after_exactly_d_steps_and_holds():
    # S = 30, D = 8, floor at 10% of peak: cosine over 8 steps from step 30.
    lr_lambda = build_lr_lambda(
        warmup_steps=0,
        decay_steps=8,
        peak_lr=5e-4,
        min_lr=5e-5,
        wsd_state={"triggered": True, "step": 30},
    )

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
    lr_lambda = build_lr_lambda(
        warmup_steps=0,
        decay_steps=8,
        peak_lr=5e-4,
        min_lr=5e-5,
        wsd_state={"triggered": False, "step": 30},
    )

    assert lr_lambda(30) == pytest.approx(1.0)
    assert lr_lambda(50) == pytest.approx(1.0)


def test_decay_counter_depends_only_on_trigger_step_and_d():
    # The same trigger step and decay length must give identical decay values
    # regardless of any "total steps" notion (relative WSD budget).
    first = build_lr_lambda(
        warmup_steps=0,
        decay_steps=10,
        peak_lr=1e-3,
        min_lr=1e-4,
        wsd_state={"triggered": True, "step": 100},
    )
    second = build_lr_lambda(
        warmup_steps=0,
        decay_steps=10,
        peak_lr=1e-3,
        min_lr=1e-4,
        wsd_state={"triggered": True, "step": 100},
    )

    for step in (100, 105, 110, 150):
        assert first(step) == pytest.approx(second(step))
    # Sanity: the halfway value is 10% + 90% * 0.5 = 55%.
    assert first(105) == pytest.approx(0.55)


def test_decay_budget_derives_total_and_d_from_the_trigger_step():
    # Rule of three: the decay is the final fraction `f` of the whole run, so
    # D = round(S * f / (1 - f)) and total = S + D.
    total, decay_steps = resolve_decay_budget(30, 0.2)
    assert decay_steps == 8
    assert total == 38

    total, decay_steps = resolve_decay_budget(1000, 0.2)
    assert decay_steps == 250
    assert total == 1250


def test_decay_budget_rounds_to_the_nearest_step():
    # 10 * 0.4 / 0.6 = 6.67 -> D = 7 (rounds up, not truncation)...
    total, decay_steps = resolve_decay_budget(10, 0.4)
    assert decay_steps == 7
    assert total == 17
    # ...while 10 * 0.3 / 0.7 = 4.29 -> D = 4 (rounds down, not ceiling).
    total, decay_steps = resolve_decay_budget(10, 0.3)
    assert decay_steps == 4
    assert total == 14


def test_decay_budget_gives_at_least_one_decay_step_for_an_empty_run():
    total, decay_steps = resolve_decay_budget(0, 0.2)
    assert decay_steps == 1
    assert total == 1


@pytest.mark.parametrize("fraction", (0.0, 1.0, 1.5, -0.2))
def test_decay_budget_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError, match="lr_decay_fraction"):
        resolve_decay_budget(30, fraction)


def test_decay_budget_rejects_negative_trigger_step():
    with pytest.raises(ValueError, match="steps_done"):
        resolve_decay_budget(-1, 0.2)


def test_wsd_cosine_shape_matches_the_reference_formula():
    # Cross-check the lambda against the closed form at several points.
    peak, floor = 5e-4, 5e-5
    s, d = 30, 8
    lr_lambda = build_lr_lambda(
        warmup_steps=0,
        decay_steps=d,
        peak_lr=peak,
        min_lr=floor,
        wsd_state={"triggered": True, "step": s},
    )
    min_scale = floor / peak
    for step in range(s, s + d + 5):
        progress = min(max(step - s, 0) / d, 1.0)
        expected = min_scale + (1.0 - min_scale) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        assert lr_lambda(step) == pytest.approx(expected, rel=1e-12)
