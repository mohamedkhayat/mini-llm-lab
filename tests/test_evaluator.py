"""T3.2: Evaluator — token-weighted loss, GPU-side accumulation, one
host sync at the end."""

import math

import pytest
import torch

from training.evaluator import EvalResult, Evaluator


class _Loader:
    """Yields (x, y) pairs; x carries the token count, y the per-token
    values a deterministic loss_fn averages."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


def _batch(n_tokens, value):
    x = torch.zeros(1, n_tokens, dtype=torch.long)
    y = torch.full((1, n_tokens), float(value))
    return x, y


def _mean_loss_fn(x, y):
    return y.float().mean()


def _evaluator(batches, num_batches=None):
    # A real (empty) module: eval_mode needs the .training flag.
    return Evaluator(
        model=torch.nn.Module(), loss_fn=_mean_loss_fn, loader=_Loader(batches),
        num_batches=num_batches,
    )


def test_eval_token_weighted_unequal_batches():
    """Unequal batch sizes: the reported loss is the token-weighted mean,
    not the plain mean of per-batch losses (4-tok batch @2.0, 8-tok batch
    @4.0 -> (2*4 + 4*8) / 12 = 10/3, while the plain mean is 3.0)."""
    result = _evaluator([_batch(4, 2.0), _batch(8, 4.0)]).run()

    assert result.loss == pytest.approx(10.0 / 3.0)
    assert result.tokens == 12
    assert result.ppl == pytest.approx(math.exp(10.0 / 3.0))
    # std([2.0, 4.0], unbiased) = sqrt(2); n=2
    assert result.stderr == pytest.approx(math.sqrt(2.0) / math.sqrt(2))


def test_eval_equal_batches_matches_manual_mean():
    """Equal batch sizes: the token-weighted mean equals the manual mean
    of per-batch losses."""
    result = _evaluator([_batch(4, 2.0), _batch(4, 4.0)]).run()

    assert result.loss == pytest.approx(3.0)
    assert result.tokens == 8
    assert isinstance(result, EvalResult)


def test_eval_num_batches_caps_the_scan():
    """num_batches limits the scan (the periodic-eval budget); None scans
    the whole split."""
    loader = _Loader([_batch(4, 1.0), _batch(4, 3.0), _batch(4, 5.0)])
    capped = _evaluator([b for b in [_batch(4, 1.0), _batch(4, 3.0), _batch(4, 5.0)]],
                        num_batches=2).run()
    assert capped.loss == pytest.approx(2.0)
    assert capped.tokens == 8

    full = _evaluator([_batch(4, 1.0), _batch(4, 3.0), _batch(4, 5.0)]).run()
    assert full.loss == pytest.approx(3.0)
    assert full.tokens == 12


def test_eval_empty_loader_is_nan():
    result = _evaluator([]).run()
    assert math.isnan(result.loss)
    assert result.tokens == 0
