"""Shared token-budget parsing and cache-split arithmetic.

The training configuration has several token quantities with different
meanings: the prepared cache is split into train/eval tokens, while phase
budgets are rounded to complete optimizer steps.  Keeping those conversions in
one dependency-free module prevents the CLI, trainer, and preparation code from
silently disagreeing about a large budget.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR


_TOKEN_VALUE_RE = re.compile(
    r"^\s*(?P<number>(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<suffix>[kmbt]?)\s*$",
    re.IGNORECASE,
)
_SUFFIX_SCALE = {
    "": Decimal(1),
    "k": Decimal(1_000),
    "m": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "t": Decimal(1_000_000_000_000),
}


def _as_decimal(value, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a token count, got boolean {value!r}")

    if isinstance(value, int):
        return Decimal(value)

    if isinstance(value, float):
        # Decimal(str(...)) avoids importing the binary floating-point residue
        # into the token count while still accepting Hydra numeric nodes.
        try:
            return Decimal(str(value))
        except InvalidOperation as exc:  # pragma: no cover - defensive
            raise ValueError(f"{field} must be a token count, got {value!r}") from exc

    if isinstance(value, str):
        match = _TOKEN_VALUE_RE.fullmatch(value)
        if match is None:
            raise ValueError(
                f"{field} must be a plain integer or a value such as 1.2b/100m; "
                f"got {value!r}"
            )
        try:
            number = Decimal(match.group("number"))
        except InvalidOperation as exc:  # pragma: no cover - regex is restrictive
            raise ValueError(f"{field} must be a token count, got {value!r}") from exc
        return number * _SUFFIX_SCALE[match.group("suffix").lower()]

    raise ValueError(
        f"{field} must be a plain integer or a value such as 1.2b/100m; "
        f"got {value!r}"
    )


def parse_token_budget(value, field: str, *, allow_none: bool = True) -> int | None:
    """Parse a token count from an integer, numeric node, or decimal suffix.

    Suffixes are decimal and case-insensitive: ``k``, ``m``, ``b``, and ``t``.
    Fractional tokens are rejected after scaling so a value like ``1.2b`` is
    valid while ``1.2`` is not.
    """

    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field} must be provided")

    decimal_value = _as_decimal(value, field)
    if decimal_value < 0:
        raise ValueError(f"{field} must be non-negative; got {value!r}")
    integral_value = decimal_value.to_integral_value(rounding=ROUND_FLOOR)
    if decimal_value != integral_value:
        raise ValueError(f"{field} must resolve to a whole number of tokens; got {value!r}")
    return int(integral_value)


def split_cache_tokens(total_tokens: int, val_ratio: float) -> tuple[int, int]:
    """Return ``(train_tokens, val_tokens)`` for a total cache size."""

    total_tokens = int(total_tokens)
    if total_tokens < 0:
        raise ValueError(f"total cache tokens must be non-negative; got {total_tokens}")

    try:
        ratio = Decimal(str(val_ratio))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio!r}") from exc
    if not Decimal(0) <= ratio < Decimal(1):
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio!r}")

    train_tokens = int(
        (Decimal(total_tokens) * (Decimal(1) - ratio)).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )
    return train_tokens, total_tokens - train_tokens


def minimum_cache_tokens(required_train_tokens: int, val_ratio: float) -> int:
    """Find the smallest total cache containing the requested train prefix."""

    required_train_tokens = int(required_train_tokens)
    if required_train_tokens < 0:
        raise ValueError(
            "required train tokens must be non-negative; "
            f"got {required_train_tokens}"
        )
    if required_train_tokens == 0:
        return 0

    try:
        ratio = Decimal(str(val_ratio))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio!r}") from exc
    if not Decimal(0) <= ratio < Decimal(1):
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio!r}")

    train_fraction = Decimal(1) - ratio
    candidate = int(
        (Decimal(required_train_tokens) / train_fraction).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    while split_cache_tokens(candidate, ratio)[0] < required_train_tokens:
        candidate += 1
    while candidate > 0 and split_cache_tokens(candidate - 1, ratio)[0] >= required_train_tokens:
        candidate -= 1
    return candidate


def complete_optimizer_steps(
    requested_tokens,
    tokens_per_step: int,
    field: str,
) -> tuple[int, int]:
    """Resolve a positive phase budget to ``(steps, effective_tokens)``."""

    requested = parse_token_budget(requested_tokens, field, allow_none=False)
    tokens_per_step = int(tokens_per_step)
    if tokens_per_step <= 0:
        raise ValueError(f"tokens_per_step must be positive; got {tokens_per_step}")
    if requested <= 0:
        raise ValueError(f"{field} must be positive; got {requested_tokens!r}")

    steps = requested // tokens_per_step
    if steps <= 0:
        raise ValueError(
            f"{field} must contain at least one complete optimizer batch "
            f"(one complete optimizer step; {tokens_per_step:,} tokens); "
            f"got {requested_tokens!r}"
        )
    return steps, steps * tokens_per_step
