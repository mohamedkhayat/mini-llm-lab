"""Regression: periodic eval must not rewind the training data stream.

With ``data.persistent_workers=true`` and ``data.num_workers > 0`` (the
project defaults), PyTorch keeps ONE iterator object per DataLoader for
its whole lifetime: every ``iter(loader)`` calls ``_reset()`` on it, and
``_reset()`` recreates the sampler stream from batch 0 of the current
epoch permutation.

``Trainer.evaluate()`` iterated the TRAIN loader for the
``train/loss_full`` metric, so every eval sent the training stream back
to the top of the pass: the model retrained on the first
``eval_interval`` batches of each epoch over and over, and the loss
curve showed a sawtooth with period == ``training.eval_interval``.

The test simulates the trainer loop exactly (same enumerate + skip
logic, same eval loader selection) on a persistent multi-worker loader
and asserts the training stream never repeats a batch.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from data.dataloader import create_dataloaders


SEQ_LEN = 256
STRIDE = 256
BATCH_SIZE = 4
STEPS = 60
EVAL_EVERY = 20
EVAL_BATCHES = 2
# ~1000 train windows per pass: far more than this run consumes, so no
# legitimate epoch rollover can happen mid-test.
WINDOWS_PER_PASS = 1000


def _write_corpus(tmp_path: Path) -> str:
    # Deterministic pseudo-text; content only needs to tokenize to
    # ~WINDOWS_PER_PASS * STRIDE tokens plus a 10% val tail.
    words = [f"w{i % 997:03d}" for i in range(997)]
    n_windows_total = int(WINDOWS_PER_PASS * 1.1)  # train is 90% of tokens
    n_tokens = n_windows_total * STRIDE + SEQ_LEN
    text = " ".join(words[i % 997] for i in range(n_tokens // 4))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    return str(corpus)


def _make_loader(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "source": "files",
            "files": [_write_corpus(tmp_path)],
            "max_files": None,
            "tokenized_dir": None,
            "hf_dataset": None,
            "hf_config": None,
            "text_column": "text",
            "split": "train",
            "streaming": False,
            "tokenizer_name": "gpt2",
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
            "batch_size": BATCH_SIZE,
            "val_ratio": 0.1,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 1,
            "pin_memory": False,
            "persistent_workers": True,  # the production default that bites
            "seed": 42,
        }
    )
    train_loader, val_loader = create_dataloaders(cfg)
    assert len(train_loader) >= STEPS, "corpus too small for the simulated run"
    return train_loader


def _simulate_trainer_loop(train_loader):
    """Replica of Trainer.train()'s loop + Trainer.evaluate()'s data path.

    Returns one fingerprint (first window of the batch) per training step.
    """
    consumed = []
    step = 0
    batch_in_epoch = 0
    while step < STEPS:
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx < batch_in_epoch:
                continue
            consumed.append(tuple(x[0].tolist()))
            step += 1
            batch_in_epoch = batch_idx + 1

            if step % EVAL_EVERY == 0:
                # Exactly what Trainer.evaluate() selects and iterates:
                eval_loader = getattr(train_loader, "eval_loader", train_loader)
                for i, (xe, ye) in enumerate(eval_loader):
                    if i < EVAL_BATCHES:
                        pass  # (trainer runs the forward pass here)
                    else:
                        break
            if step >= STEPS:
                break
        if step >= STEPS:
            break
    return consumed


def test_periodic_eval_does_not_rewind_train_stream(tmp_path):
    train_loader = _make_loader(tmp_path)
    consumed = _simulate_trainer_loop(train_loader)
    assert len(consumed) == STEPS

    seen = set()
    for idx, fp in enumerate(consumed):
        assert fp not in seen, (
            f"training stream repeated a batch at step {idx}: every "
            f"{EVAL_EVERY}-step eval must not rewind the persistent "
            "multi-worker training iterator to the start of the pass"
        )
        seen.add(fp)
