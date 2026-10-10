"""Evaluation and sampling.

``Evaluator`` runs a bounded (or full) scan of an eval loader: the
token-weighted loss sum and the per-batch losses are accumulated on the
model's device and moved to the host exactly once, at the end (P7 — the
old per-batch ``loss.item()`` syncs are gone).

``Sampler`` produces one generation sample; generation settings come
from a plain ``gen_cfg`` namespace (the generation config group proper
arrives in T7.2 — until then the trainer builds the namespace from
``cfg.training.start_context/max_new_tokens`` + the model cfg's
temperature/top_k). It never reads ``cfg.model``.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from data.tokenizer import text_to_token_ids, token_ids_to_text


@contextmanager
def eval_mode(model: torch.nn.Module):
    """Run the body with the model in eval mode, restoring the previous
    training flag on exit (including on exception)."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class EvalResult:
    loss: float
    ppl: float
    stderr: float
    tokens: int


class Evaluator:
    """A validation scan: ``run()`` returns the token-weighted mean loss,
    perplexity, standard error of the batch losses, and token count.

    ``num_batches=None`` scans the whole split (end-of-run eval); a bound
    is the periodic-eval budget. The loader is expected to yield batches
    already on the model's device (the memmap loaders do).
    """

    def __init__(self, model, loss_fn, loader, num_batches=None, device=None):
        self._model = model
        self._loss_fn = loss_fn
        self._loader = loader
        self._num_batches = num_batches
        self._device = device

    @torch.no_grad()
    def run(self) -> EvalResult:
        with eval_mode(self._model):
            return self._scan()

    def _scan(self) -> EvalResult:
        batch_losses = []
        loss_sum = None  # token-weighted sum, kept on device
        tokens = 0
        n = 0
        for x, y in self._loader:
            if self._num_batches is not None and n >= self._num_batches:
                break
            if self._device is not None:
                x, y = x.to(self._device), y.to(self._device)
            loss = self._loss_fn(x, y)
            batch_losses.append(loss.detach())
            weight = x.numel()
            loss_sum = (
                loss.detach() * weight
                if loss_sum is None
                else loss_sum + loss.detach() * weight
            )
            tokens += weight
            n += 1

        if n == 0:
            return EvalResult(float("nan"), float("nan"), float("nan"), 0)

        assert loss_sum is not None  # n > 0 means at least one batch was added
        batch = torch.stack(batch_losses)
        mean_gpu = loss_sum / tokens
        stderr_gpu = batch.std(unbiased=True) / math.sqrt(n)
        # The single host sync of the whole scan.
        mean, stderr = torch.stack([mean_gpu, stderr_gpu]).tolist()
        return EvalResult(mean, math.exp(mean), stderr, tokens)


class Sampler:
    """One generation sample, wrapped in ``eval_mode`` and ``no_grad``.

    Generation settings come from ``gen_cfg`` (a plain namespace with
    ``start_context``, ``max_new_tokens``, ``temperature``, ``top_k``);
    ``context_length`` comes from the model cfg via the constructor.
    """

    def __init__(self, model, tokenizer, gen_cfg, context_length, autocast_ctx):
        self._model = model
        self._tokenizer = tokenizer
        self._gen_cfg = gen_cfg
        self._context_length = context_length
        self._autocast_ctx = autocast_ctx
        self._device = next(model.parameters()).device

    def sample(self) -> str:
        with eval_mode(self._model):
            encoded = text_to_token_ids(
                self._gen_cfg.start_context, self._tokenizer, device=self._device
            )
            with torch.no_grad():
                token_ids = self._model.generate(
                    encoded,
                    context_length=self._context_length,
                    max_new_tokens=self._gen_cfg.max_new_tokens,
                    autocast_context=self._autocast_ctx,
                    temperature=self._gen_cfg.temperature,
                    top_k=self._gen_cfg.top_k,
                )
            return token_ids_to_text(token_ids, self._tokenizer).replace("\n", " ")
