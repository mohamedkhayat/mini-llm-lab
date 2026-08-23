import pytest

from training.checkpointing import check_resume_consistency
from training.schedule import resolve_wsd_state, stage_label


def test_fresh_flag_triggers_the_decay_at_the_resume_step():
    state = resolve_wsd_state(None, start_decay=True, resume_step=30)

    assert state == {"triggered": True, "step": 30}


def test_no_checkpoint_state_and_no_flag_stays_untriggered():
    state = resolve_wsd_state(None, start_decay=False, resume_step=30)

    assert state == {"triggered": False, "step": None}


def test_legacy_checkpoint_without_wsd_key_falls_back_to_the_flag():
    state = resolve_wsd_state({}, start_decay=True, resume_step=30)

    assert state == {"triggered": True, "step": 30}


def test_saved_triggered_state_wins_over_the_flag():
    # A crash mid-decay must resume from the original trigger step, not
    # re-trigger at the resume step, even if the flag is set again.
    state = resolve_wsd_state(
        {"triggered": True, "step": 12}, start_decay=True, resume_step=30
    )

    assert state == {"triggered": True, "step": 12}


def test_saved_triggered_state_survives_even_without_the_flag():
    state = resolve_wsd_state(
        {"triggered": True, "step": 12}, start_decay=False, resume_step=30
    )

    assert state == {"triggered": True, "step": 12}


def test_saved_untriggered_state_defers_to_the_flag():
    # A stable checkpoint with start_decay on starts the decay now (at the
    # resume step); a saved "not yet triggered" state does not block it.
    state = resolve_wsd_state(
        {"triggered": False, "step": None}, start_decay=True, resume_step=30
    )

    assert state == {"triggered": True, "step": 30}


def test_malformed_saved_state_falls_back_to_the_flag():
    state = resolve_wsd_state(
        {"triggered": True, "step": None}, start_decay=True, resume_step=30
    )

    assert state == {"triggered": True, "step": 30}


def test_matching_resume_budget_is_accepted():
    check_resume_consistency(
        saved_total_steps=60,
        total_steps=60,
        saved_steps_per_epoch=40,
        steps_per_epoch=40,
        decay_run=False,
    )


def test_non_decay_resume_with_different_total_budget_is_rejected():
    with pytest.raises(ValueError, match="total step budget"):
        check_resume_consistency(
            saved_total_steps=60,
            total_steps=100,
            saved_steps_per_epoch=40,
            steps_per_epoch=40,
            decay_run=False,
        )


def test_decay_run_budget_mismatch_is_accepted():
    # Stage 2 re-derives its budget from the trigger step, so a different
    # configured total must not raise.
    check_resume_consistency(
        saved_total_steps=60,
        total_steps=100,
        saved_steps_per_epoch=40,
        steps_per_epoch=40,
        decay_run=True,
    )


def test_steps_per_epoch_mismatch_is_rejected_even_for_decay_runs():
    with pytest.raises(ValueError, match="batches per"):
        check_resume_consistency(
            saved_total_steps=60,
            total_steps=100,
            saved_steps_per_epoch=40,
            steps_per_epoch=80,
            decay_run=True,
        )


def test_legacy_checkpoint_without_saved_budgets_is_accepted():
    check_resume_consistency(
        saved_total_steps=None,
        total_steps=100,
        saved_steps_per_epoch=None,
        steps_per_epoch=40,
        decay_run=False,
    )


def test_stage_labels():
    assert stage_label(resumed=False, decay_triggered=False) == "stage-1-stable"
    assert stage_label(resumed=True, decay_triggered=False) == "stage-2-stable"
    assert stage_label(resumed=True, decay_triggered=True) == "stage-2-decay"