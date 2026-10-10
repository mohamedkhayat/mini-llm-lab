"""T1.2: eval_mode is exception-safe (P8)."""

import torch
import torch.nn as nn

from training.evaluator import eval_mode


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.training = True


def test_eval_mode_exception_safe_restores_train_mode():
    model = _ToyModel()
    model.train()
    try:
        with eval_mode(model):
            assert model.training is False
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert model.training is True, "exception left the model in eval mode"


def test_eval_mode_restores_the_previous_training_flag():
    model = _ToyModel()
    model.eval()  # entered while in eval mode
    with eval_mode(model):
        assert model.training is False
    assert model.training is False, "eval mode must be preserved after exit"


def test_trainer_uses_eval_mode_in_evaluate(tiny_cfg, monkeypatch):
    """The trainer's evaluate path uses eval_mode: an exception in the loss
    path leaves the model in train mode. (T6.3: the loop-only trainer's
    evaluate path is the Evaluator's run — the context manager lives there.)"""
    import pytest

    from training.assembly import build_trainer

    tiny_cfg.training.train_tokens = 2 * 64
    trainer = build_trainer(tiny_cfg)
    assert trainer.model.training is True  # fresh models are in train mode

    def _boom(x, y):  # type: ignore[no-untyped-def]
        raise RuntimeError("eval boom")

    monkeypatch.setattr(trainer.evaluator, "_loss_fn", _boom)
    with pytest.raises(RuntimeError, match="eval boom"):
        trainer.evaluator.run()
    assert trainer.model.training is True, "evaluate() left the model in eval mode"
