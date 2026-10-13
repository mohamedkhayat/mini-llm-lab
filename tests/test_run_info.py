"""T5.1: RunInfo and RunLogger.init(info) — the 9-kwarg list is gone."""

import dataclasses
from typing import Any, cast

import pytest

from training.log_backend import RunInfo, TerminalLogger, WandbLogger


def _info(**overrides):
    fields = dict(
        name="exp",
        config={"training": {"exp_name": "exp"}},
        saved_config=None,
        resumed_run_id=None,
        computed={"computed_total_steps": 38},
        stage_marker={"training_stage": "stage-1-stable"},
        parameter_metrics={
            "model/parameters_total": 1000,
            "model/parameters_trainable": 900,
            "model/parameters_non_trainable": 100,
            "model/parameter_memory_mb": 0.0038,
        },
        step=38,
    )
    fields.update(overrides)
    return RunInfo(**cast("dict[str, Any]", fields))


def test_run_info_is_a_frozen_dataclass_with_the_eight_fields():
    info = _info()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.name = "x"  # pyright: ignore[reportAttributeAccessIssue]
    assert {f.name for f in dataclasses.fields(info)} == {
        "name",
        "config",
        "saved_config",
        "resumed_run_id",
        "computed",
        "stage_marker",
        "parameter_metrics",
        "step",
    }


def test_terminal_logger_accepts_run_info():
    TerminalLogger().init(_info())  # no exception, no output contract


class _FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.config_updates = []
        self.config: Any = None

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = None

    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def test_wandb_logger_unpacks_run_info():
    fake = _FakeWandb()
    fake.config = type("C", (), {"update": staticmethod(lambda d: None)})()
    logger = WandbLogger(wandb_module=fake)
    logger.init(_info())

    assert fake.init_calls == [
        {
            "project": "mini-llm-lab",
            "name": "exp",
            "config": {"training": {"exp_name": "exp"}},
        }
    ]


class _FakeWandbConfig:
    """Emulates wandb's resume rule: updating a key whose restored value
    differs raises, exactly like wandb.sdk.lib.config_util.ConfigError."""

    def __init__(self, values):
        self._values = dict(values)
        self.updated = None

    def items(self):
        return self._values.items()

    def update(self, d, allow_val_change=False):
        for k, v in d.items():
            if k in self._values and self._values[k] != v and not allow_val_change:
                raise ValueError(
                    f'Attempted to change value of key "{k}" from '
                    f"{self._values[k]!r} to {v!r}"
                )
            self._values[k] = v
        self.updated = d


def test_wandb_logger_resume_skips_changed_original_computed_values():
    """A resumed run restores the original run's W&B config. Recomputed
    values that differ (budget mode one_pass -> exact_resume) must be
    skipped, not overwritten, or W&B raises ConfigError. Equal and absent
    keys are still set."""
    fake = _FakeWandb()
    original = {
        "computed_budget_name": "one_pass",
        "computed_phase_target_step": None,
        "training_stage": "stage-1-stable",
        "model_parameters_total": 1000,
    }
    fake.config = _FakeWandbConfig(original)
    logger = WandbLogger(wandb_module=fake)
    logger.init(
        _info(
            saved_config={"training": {"exp_name": "original"}},
            resumed_run_id="abc123",
            computed={
                "computed_budget_name": "exact_resume",
                "computed_phase_target_step": 46920,
                "computed_total_steps": 46920,
            },
        )
    )

    # Conflicting keys keep their restored original values.
    assert fake.config._values["computed_budget_name"] == "one_pass"
    assert fake.config._values["computed_phase_target_step"] is None
    # Equal and absent keys pass through.
    assert fake.config._values["computed_total_steps"] == 46920
    assert fake.config._values["training_stage"] == "stage-1-stable"
    # The update payload never carried a conflicting value.
    assert "computed_budget_name" not in fake.config.updated
    assert "computed_phase_target_step" not in fake.config.updated
    assert fake.config.updated["computed_total_steps"] == 46920


def test_wandb_logger_resumed_run_keeps_original_config_and_name():
    """Behavior preserved from the kwarg era: a resumed run keeps the
    original run's config and name in W&B."""
    fake = _FakeWandb()
    fake.config = _FakeWandbConfig({})
    logger = WandbLogger(wandb_module=fake)
    saved_config = {"training": {"exp_name": "original"}}
    logger.init(
        _info(
            name="renamed-on-relaunch",
            config={"training": {"exp_name": "renamed-on-relaunch"}},
            saved_config=saved_config,
            resumed_run_id="abc123",
        )
    )

    assert fake.init_calls == [
        {
            "project": "mini-llm-lab",
            "name": "original",
            "config": saved_config,
            "id": "abc123",
            "resume": "must",
        }
    ]
