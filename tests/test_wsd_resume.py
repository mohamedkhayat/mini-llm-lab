import pytest

from training.checkpointing import check_resume_consistency
from training.schedule import stage_label


def test_matching_resume_budget_is_accepted():
    check_resume_consistency(
        saved_total_steps=60,
        total_steps=60,
        saved_steps_per_pass=40,
        steps_per_pass=40,
        saved_tokens_per_step=16,
        tokens_per_step=16,
        decay_run=False,
    )


def test_non_decay_resume_with_different_total_budget_is_rejected():
    with pytest.raises(ValueError, match="total step budget"):
        check_resume_consistency(
            saved_total_steps=60,
            total_steps=100,
            saved_steps_per_pass=40,
            steps_per_pass=40,
            saved_tokens_per_step=16,
            tokens_per_step=16,
            decay_run=False,
        )


def test_decay_run_budget_mismatch_is_accepted():
    # Stage 2 re-derives its budget from the trigger step, so a different
    # configured total must not raise.
    check_resume_consistency(
        saved_total_steps=60,
        total_steps=100,
        saved_steps_per_pass=40,
        steps_per_pass=40,
        saved_tokens_per_step=16,
        tokens_per_step=16,
        decay_run=True,
    )


def test_stable_continuation_budget_mismatch_is_accepted():
    check_resume_consistency(
        saved_total_steps=60,
        total_steps=91,
        saved_steps_per_pass=40,
        steps_per_pass=40,
        saved_tokens_per_step=16,
        tokens_per_step=16,
        decay_run=False,
        continuation_run=True,
    )


def test_steps_per_pass_mismatch_is_rejected_even_for_decay_runs():
    with pytest.raises(ValueError, match="batches per"):
        check_resume_consistency(
            saved_total_steps=60,
            total_steps=100,
            saved_steps_per_pass=40,
            steps_per_pass=80,
            saved_tokens_per_step=16,
            tokens_per_step=16,
            decay_run=True,
        )


def test_phase_extension_allows_a_larger_reprocessed_cache():
    check_resume_consistency(
        saved_total_steps=60,
        total_steps=100,
        saved_steps_per_pass=40,
        steps_per_pass=80,
        saved_tokens_per_step=16,
        tokens_per_step=16,
        decay_run=True,
        allow_capacity_growth=True,
    )


def test_phase_extension_still_rejects_a_smaller_cache():
    with pytest.raises(ValueError, match="batches per"):
        check_resume_consistency(
            saved_total_steps=60,
            total_steps=100,
            saved_steps_per_pass=40,
            steps_per_pass=20,
            saved_tokens_per_step=16,
            tokens_per_step=16,
            decay_run=True,
            allow_capacity_growth=True,
        )


def test_phase_cap_can_be_smaller_than_an_unchanged_physical_cache():
    check_resume_consistency(
        saved_total_steps=60,
        total_steps=30,
        saved_steps_per_pass=60,
        steps_per_pass=30,
        saved_tokens_per_step=16,
        tokens_per_step=16,
        decay_run=True,
        allow_capacity_growth=True,
        available_steps_per_pass=60,
    )


def test_stage_labels():
    assert stage_label(resumed=False, decay_triggered=False) == "stage-1-stable"
    assert stage_label(resumed=True, decay_triggered=False) == "stage-2-stable"
    assert stage_label(resumed=True, decay_triggered=True) == "stage-2-decay"
