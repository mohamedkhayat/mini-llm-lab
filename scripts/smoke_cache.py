"""Write a tiny fake token cache at data/smoke/gpt2 for CPU smoke runs.

The trainer skips dataset preparation when the cache is valid, so this lets
`python train.py data.hf_dataset=smoke ...` run end-to-end with no download.
"""

import json
from pathlib import Path

import numpy as np

TRAIN, VAL = 6336, 704

d = Path("data/smoke/gpt2")
d.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
rng.integers(1, 50257, TRAIN, dtype=np.uint16).tofile(d / "train.bin")
rng.integers(1, 50257, VAL, dtype=np.uint16).tofile(d / "eval.bin")
meta = {
    "tokenizer_name": "gpt2",
    "hf_dataset": "smoke",
    "hf_config": None,
    "text_column": "text",
    "file_format": "parquet",
    "dtype": "uint16",
    "max_tokens": TRAIN + VAL,
    "train_tokens": TRAIN,
    "val_tokens": VAL,
    "val_ratio": VAL / (TRAIN + VAL),
}
(d / "meta.json").write_text(json.dumps(meta))
print(f"wrote {d}")
