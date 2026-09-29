import pytest

from data.budget import (
    complete_optimizer_steps,
    minimum_cache_tokens,
    parse_token_budget,
    split_cache_tokens,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (100, 100),
        ("100", 100),
        ("100m", 100_000_000),
        ("1.2b", 1_200_000_000),
        ("64K", 64_000),
        ("0.5t", 500_000_000_000),
    ],
)
def test_token_budget_parser_accepts_plain_and_decimal_suffix_values(value, expected):
    assert parse_token_budget(value, "training.decay_tokens") == expected


@pytest.mark.parametrize("value", [None, "", "1.2", "1.2x", -1, "-1b", True])
def test_token_budget_parser_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="training.decay_tokens"):
        parse_token_budget(value, "training.decay_tokens", allow_none=False)


def test_token_budget_parser_allows_optional_none():
    assert parse_token_budget(None, "training.train_tokens", allow_none=True) is None


def test_complete_optimizer_steps_rounds_down_and_reports_effective_tokens():
    assert complete_optimizer_steps("1.2b", 131_072, "training.decay_tokens") == (
        9155,
        9155 * 131_072,
    )


def test_complete_optimizer_steps_rejects_budget_smaller_than_one_step():
    with pytest.raises(ValueError, match="complete optimizer step"):
        complete_optimizer_steps("100m", 131_072_000, "training.decay_tokens")


def test_cache_split_matches_total_and_validation_ratio():
    assert split_cache_tokens(10_000, 0.1) == (9_000, 1_000)
    assert split_cache_tokens(10_001, 0.1) == (9_000, 1_001)


def test_minimum_cache_tokens_holds_requested_train_capacity():
    total = minimum_cache_tokens(10_900, 0.1)
    train, val = split_cache_tokens(total, 0.1)

    assert train >= 10_900
    assert train + val == total
    assert split_cache_tokens(total - 1, 0.1)[0] < 10_900
