"""T6.3 throughput gate: 200-step CPU run vs the T0.3 baseline.

Recipe (identical to ticket 03's baseline): tiny model (2 layers,
emb_dim=32, vocab=256; batch=2, seq=16, accum=2), ``train_tokens=12800``,
seed 42, terminal logging, CPU fp32. Wall clock around ``train()`` only
(construction excluded); tokens = 12,800. Baseline: 11,464 tok/s (1.12s),
peak RSS 762 MiB. T6.3 must stay within +/-2%.
"""

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import numpy as np
import torch
from omegaconf import OmegaConf

# CPU-only (the shared GPU is memory-starved — same guard as the test
# fixtures): stub the CUDA availability before anything touches a device.
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

from training.assembly import build_trainer

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/t63-throughput")
OUT.mkdir(parents=True, exist_ok=True)

VOCAB, SEQ_LEN = 256, 16
TRAIN_TOKENS, EVAL_TOKENS = 13_000, 512

# Seeded cache (same layout as the tiny_memmap fixture).
data_dir = OUT / "data" / "smoke" / "gpt2"
data_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
rng.integers(1, VOCAB, size=TRAIN_TOKENS).astype(np.uint16).tofile(data_dir / "train.bin")
rng.integers(1, VOCAB, size=EVAL_TOKENS).astype(np.uint16).tofile(data_dir / "eval.bin")
json.dump(
    {
        "tokenizer_name": "gpt2",
        "dtype": "uint16",
        "max_tokens": TRAIN_TOKENS + EVAL_TOKENS,
        "train_tokens": TRAIN_TOKENS,
        "val_tokens": EVAL_TOKENS,
    },
    open(data_dir / "meta.json", "w"),
)

cfg = OmegaConf.create(
    {
        "data": {
            "hf_dataset": "smoke",
            "tokenizer_name": "gpt2",
            "seq_len": SEQ_LEN,
            "max_tokens": TRAIN_TOKENS + EVAL_TOKENS,
        },
        "model": {
            "name": "gpt2",
            "attention": "mha",
            "tie_embeddings": False,
            "vocab_size": VOCAB,
            "context_length": 128,
            "emb_dim": 32,
            "hidden_dim": 64,
            "n_heads": 2,
            "n_kv_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": True,
            "ffn_bias": True,
            "normalization": "layernorm",
            "activation": "gelu",
            "gated": False,
            "equalize_params": False,
            "position_embedding": "absolute",
            "residual_style": "serial",
            "logit_softcap": None,
            "rope_theta": None,
            "temperature": 1.0,
            "top_k": 25,
        },
        "training": {
            "exp_name": "t63-throughput",
            "device": "cpu",
            "compile": False,
            "use_bf16": False,
            "use_tensor_cores": False,
            "log_backend": "terminal",
            "upload_artifacts": False,
            "lr": 5e-4,
            "min_lr": 5e-5,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "max_grad_norm": 1.0,
            "batch_size": 2,
            "accum_steps": 2,
            "train_tokens": 12_800,
            "warmup_fraction": 0.0,
            "eval_interval": 10**9,
            "eval_batches": 2,
            "log_interval": 25,
            "save_interval": None,
            "resume_from": None,
            "resume_mode": "exact",
            "start_decay": False,
            "seed": 42,
            "use_flce": False,
            "start_context": "a",
            "max_new_tokens": 2,
        },
    }
)

import os

os.chdir(OUT)
trainer = build_trainer(cfg)  # construction excluded from the timing
start = time.perf_counter()
trainer.train()
wall = time.perf_counter() - start

tokens = 12_800
peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
result = {
    "wall_seconds": wall,
    "tokens": tokens,
    "tok_per_sec": tokens / wall,
    "peak_rss_mib": peak_rss_mib,
    "baseline_tok_per_sec": 11_464,
    "within_2pct": abs(tokens / wall - 11_464) / 11_464 <= 0.02,
}
print(json.dumps(result, indent=2))
(OUT / "result.json").write_text(json.dumps(result, indent=2))
