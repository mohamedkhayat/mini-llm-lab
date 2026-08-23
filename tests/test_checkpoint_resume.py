import torch
from torch import nn
from torch.utils.data import Dataset

from data.dataloader import EpochRandomSampler
from training.checkpointing import (
    atomic_torch_save,
    load_checkpoint,
)
from training.trainer import get_model_parameter_metrics


class RangeDataset(Dataset):
    def __len__(self):
        return 17

    def __getitem__(self, index):
        return index


def sampler_order(sampler):
    return list(iter(sampler))


def test_epoch_sampler_recreates_each_data_pass():
    first = EpochRandomSampler(RangeDataset(), seed=42)
    second = EpochRandomSampler(RangeDataset(), seed=42)

    first.set_epoch(3)
    second.set_epoch(3)
    assert sampler_order(first) == sampler_order(second)

    second.set_epoch(4)
    assert sampler_order(first) != sampler_order(second)


def test_checkpoint_write_is_atomic_and_readable(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_torch_save({"step": 12, "tensor": torch.tensor([1, 2, 3])}, path)

    checkpoint = load_checkpoint(path)
    assert checkpoint["step"] == 12
    torch.testing.assert_close(checkpoint["tensor"], torch.tensor([1, 2, 3]))
    assert not list(tmp_path.glob(".latest.pt.tmp.*"))


def test_parameter_metrics_count_tied_parameters_once():
    model = nn.Module()
    model.first = nn.Linear(4, 4, bias=False)
    model.second = nn.Linear(4, 4, bias=False)
    model.second.weight = model.first.weight
    model.frozen = nn.Parameter(torch.ones(3), requires_grad=False)

    metrics = get_model_parameter_metrics(model)

    assert metrics["model/parameters_total"] == 19
    assert metrics["model/parameters_trainable"] == 16
    assert metrics["model/parameters_non_trainable"] == 3
