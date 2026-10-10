"""T5.1: RunInfo and RunLogger.init(info) — the 9-kwarg list is gone."""

import dataclasses

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
    return RunInfo(**fields)


def test_run_info_is_a_frozen_dataclass_with_the_eight_fields():
    info = _info()
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.name = "x"
    assert {f.name for f in dataclasses.fields(info)} == {
        "name", "config", "saved_config", "resumed_run_id",
        "computed", "stage_marker", "parameter_metrics", "step",
    }


def test_terminal_logger_accepts_run_info():
    TerminalLogger().init(_info())  # no exception, no output contract


class _FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.config_updates = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = None

    config = None

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


def test_wandb_logger_resumed_run_keeps_original_config_and_name():
    """Behavior preserved from the kwarg era: a resumed run keeps the
    original run's config and name in W&B."""
    fake = _FakeWandb()
    fake.config = type("C", (), {"update": staticmethod(lambda d: None)})()
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
