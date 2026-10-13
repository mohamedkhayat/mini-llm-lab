import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from chat import _planned_training_description


def test_planned_description_parses_suffixed_max_tokens():
    cfg = OmegaConf.create({"data": {"max_tokens": "10b"}})
    assert (
        _planned_training_description(cfg) == "≤ 10.00B tokens (one pass over the data)"
    )


def test_planned_description_plain_int_max_tokens():
    cfg = OmegaConf.create({"data": {"max_tokens": 6000000000}})
    assert (
        _planned_training_description(cfg) == "≤ 6.00B tokens (one pass over the data)"
    )


def test_planned_description_no_max_tokens():
    assert _planned_training_description(OmegaConf.create({})) == "budget unknown"
