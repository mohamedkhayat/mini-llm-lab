from typing import Any

import torch
from torch.utils.data import IterableDataset

from data.tokenizer import get_tokenizer
from datasets import load_dataset

class HFStreamingWindowDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: str | None,
        split: str,
        text_column: str,
        seq_len: int,
        stride: int,
        val_ratio: float,
        split_name: str,
        max_samples: int,
        seed: int = 42,
        shuffle: bool = True,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0 or stride <= 0:
            raise ValueError("seq_len and stride must be positive")
        if split_name not in {"train", "val"}:
            raise ValueError("split_name must be 'train' or 'val'")
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")

        self.dataset_name = str(dataset_name)
        self.dataset_config = dataset_config
        self.split = str(split)
        self.text_column = str(text_column)
        self.seq_len = int(seq_len)
        self.val_ratio = float(val_ratio)

        self.split_name = split_name
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.max_tokens = max_tokens

        self.dataset = load_dataset(self.dataset_config.hf_dataset, split=split, )

    def __len__(self) -> int:
        return self.max_samples

    def __iter__(self, idx):
        pass
