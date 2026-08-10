import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """Sliding-window dataset over a flat tensor of token IDs.

    Windows are sliced on-access (no pre-materialization).
    Works for both in-memory corpora and streamed chunk buffers.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int, stride: int) -> None:
        super().__init__()
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional tensor")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride
        # Each example needs seq_len input tokens plus one target token.
        self._n_samples = max(0, (len(token_ids) - seq_len - 1) // stride + 1)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(f"dataset index out of range: {idx}")
        i = idx * self.stride
        x = self.token_ids[i : i + self.seq_len]
        y = self.token_ids[i + 1 : i + self.seq_len + 1]
        return x, y


def get_train_val_split(
    token_ids: torch.Tensor, val_ratio: float, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic split of a flat token tensor by index."""
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    n = len(token_ids)
    split = int(n * (1 - val_ratio))
    return token_ids[:split], token_ids[split:]
