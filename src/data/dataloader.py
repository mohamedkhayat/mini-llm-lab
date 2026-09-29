import json
import numpy as np
import torch


class MemmapDataLoader:
    """Sequential scan over a prepared ``.bin`` token file (train/eval split).

    One optimizer step consumes ``batch_size * block_size`` contiguous tokens.
    The train split is a single pass from the saved resume cursor — iteration
    ends at the file's end (the trainer rejects budgets that would rescan);
    the eval split restarts at batch 0 on every iteration. ``state_dict`` /
    ``load_state_dict`` carry the resume cursor (``current_step``).
    """

    def __init__(
        self, data_dir, split, batch_size, block_size, device, max_tokens=None
    ):
        if split not in {"train", "eval"}:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        # the trailing tokens of the last incomplete batch are dropped
        # (total_batches below), so no padding is needed

        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size  # block size is just seq_len
        self.device = str(device)
        self.batch_tokens = batch_size * block_size

        # load the meta data json (dtype of the .bin files)
        meta_path = self.data_dir / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32

        # and the .bin file for the split
        data_path = self.data_dir / ("train.bin" if split == "train" else "eval.bin")
        if not data_path.is_file():
            raise FileNotFoundError(
                f"{data_path.name} not found in {data_dir}. Run data "
                "preparation first (train.py does this automatically)."
            )
        self.data = np.memmap(data_path, dtype, "r")

        if max_tokens is not None:
            self.total_tokens = min(max_tokens, len(self.data))
        else:
            self.total_tokens = len(self.data)

        self.total_batches = (
            self.total_tokens - 1
        ) // self.batch_tokens  # to ensure the offset is valid
        self.current_step = 0

    def _get_batch(self):
        # we start at element current batch * batch_size
        start_idx = self.current_step * self.batch_tokens
        end_idx = start_idx + self.batch_tokens + 1

        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        x = chunk[:-1].view(self.batch_size, self.block_size)
        y = chunk[1:].view(self.batch_size, self.block_size)

        self.current_step += 1
        if self.device == "cuda":
            return (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        if self.split != "train":
            self.current_step = 0
        while self.current_step < self.total_batches:
            yield self._get_batch()

    def state_dict(self):
        return {"current_step" : self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]
