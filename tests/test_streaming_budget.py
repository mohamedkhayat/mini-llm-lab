"""Behavioral tests for the HF streaming training-budget contract.

A streaming run is driven through the public seams: ``create_dataloaders``
for loader setup and the full ``Trainer(cfg)`` constructor plus
``Trainer.train()`` for run setup/stop. The HF source is replaced with a
small fake ``IterableDataset`` (``datasets.load_dataset`` is monkeypatched)
and the tokenizer with a stub, so the tests run offline with no hub access,
no corpus, and no GPU.
"""

import datasets
import pytest
from omegaconf import OmegaConf

import data.dataloader as dataloader_module
import data.hf_streaming as hf_streaming_module
import training.trainer as trainer_module
from data.dataloader import create_dataloaders
from training.trainer import Trainer

VOCAB = 64
SEQ_LEN = 4
BATCH_SIZE = 2

# Seven rows of eight tokens each (ids 0..55, below VOCAB; the eot token 60
# is appended per row). With val_ratio 0.5 the even rows are validation and
# the odd rows are training: each training row (8 tokens + eot) yields two
# non-overlapping windows of 4, so the three training rows (1, 3, 5) cover a
# 3-step run at two windows per pass without needing to cycle the source.
ROWS = [" ".join(str(row * 8 + k) for k in range(8)) for row in range(7)]


class FakeStreamingSource:
    """Stand-in for an HF streaming ``IterableDataset``.

    Mirrors the native state contract: ``state_dict`` records the cursor
    (index of the next row to yield), ``load_state_dict`` restores it so a
    resumed pass skips already-yielded rows, and ``set_epoch`` is the epoch
    hook the adapter calls when it cycles the source.
    """

    def __init__(self, rows):
        self.rows = list(rows)
        self._next_row = 0

    def state_dict(self):
        return {"next_row": self._next_row}

    def load_state_dict(self, state):
        self._next_row = int(state.get("next_row", 0))

    def set_epoch(self, epoch):
        pass

    def __iter__(self):
        while self._next_row < len(self.rows):
            row = self.rows[self._next_row]
            self._next_row += 1
            yield {"text": row}


class FakeTokenizer:
    """Split-on-whitespace tokenizer; token ids are the literal numbers."""

    eot_token = 35

    def encode(self, text, allowed_special=None, disallowed_special=()):
        return [int(value) for value in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def streaming_cfg(training_overrides=None, data_overrides=None):
    """A complete offline run config for a tiny streaming training run."""
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "gpt2",
                "attention": "mha",
                "tie_embeddings": False,
                "vocab_size": VOCAB,
                "context_length": SEQ_LEN,
                "emb_dim": 32,
                "hidden_dim": 64,
                "n_heads": 2,
                "n_kv_heads": 2,
                "n_layers": 2,
                "drop_rate": 0.0,
                "qkv_bias": True,
                "ffn_bias": True,
                "norm": "layernorm",
                "activation": "gelu",
                "gated": False,
                "equalize_params": False,
                "position_embedding": "absolute",
                "residual_style": "serial",
                "logit_softcap": None,
                "rope_theta": None,
                "temperature": 1.0,
                "top_k": 0,
            },
            "training": {
                "seed": 0,
                "device": "cpu",
                "lr": 1e-2,
                "min_lr": 1e-3,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "warmup_fraction": 0.0,
                "lr_decay_fraction": 0.2,
                "start_decay": False,
                "log_interval": 10,
                "eval_interval": 999,
                "eval_batches": 2,
                "save_interval": 0,
                "max_steps": None,
                "max_tokens": None,
                "epochs": 10,
                "max_new_tokens": 2,
                "start_context": "stub",
                "exp_name": "streaming-budget-test",
                "log_backend": "terminal",
                "use_bf16": False,
                "use_tensor_cores": False,
                "compile": False,
                "compile_mode": "default",
                "resume_from": None,
                "resume_mode": "exact",
                "continue_tokens": None,
                "upload_artifacts": False,
            },
            "data": {
                "source": "hf_dataset",
                "files": [],
                "tokenized_dir": None,
                "hf_dataset": "local/fake-stream",
                "hf_config": None,
                "text_column": "text",
                "split": "train",
                "streaming": True,
                "tokenizer_name": "fake",
                "seq_len": SEQ_LEN,
                "stride": SEQ_LEN,
                "batch_size": BATCH_SIZE,
                "val_ratio": 0.5,
                "shuffle": False,
                "shuffle_buffer_size": 100,
                "drop_last": True,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "seed": 0,
            },
        }
    )
    if data_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": data_overrides}))
    if training_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": training_overrides}))
    return cfg


@pytest.fixture
def streaming_run(monkeypatch, tmp_path):
    """Offline streaming-run harness: fake HF source + stub tokenizer.

    ``streaming_run(training_overrides=..., data_overrides=..., run=True)``
    builds a real ``Trainer`` (full config-driven constructor) and optionally
    runs ``train()`` to completion. The cwd is pinned to ``tmp_path`` so
    checkpoints and run manifests land in the temp dir.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *args, **kwargs: FakeStreamingSource(ROWS)
    )
    fake = FakeTokenizer()
    monkeypatch.setattr(trainer_module, "get_tokenizer", lambda _name: fake)
    monkeypatch.setattr(hf_streaming_module, "get_tokenizer", lambda _name: fake)

    def build(training_overrides=None, data_overrides=None, run=False):
        trainer = Trainer(streaming_cfg(training_overrides, data_overrides))
        if run:
            trainer.train()
        return trainer

    return build


def test_streaming_run_without_an_explicit_budget_fails_clearly(streaming_run):
    """Neither max_steps nor max_tokens: setup must fail loudly instead of
    silently converting training.epochs (a data-pass count) into steps for
    an unbounded stream."""
    with pytest.raises(ValueError, match="explicit training budget"):
        streaming_run()

    with pytest.raises(ValueError, match="training\\.epochs"):
        streaming_run()


def test_streaming_run_rejects_legacy_data_max_tokens_with_migration_message(
    monkeypatch, tmp_path
):
    """data.max_tokens (a per-pass unique-source-token cap) is rejected on
    streaming runs with a message pointing at the training budget; the data
    path never reaches tokenizer or dataset construction."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        hf_streaming_module, "get_tokenizer", lambda _name: FakeTokenizer()
    )
    cfg = streaming_cfg(data_overrides={"max_tokens": 1000})

    with pytest.raises(ValueError, match="training\\.max_tokens"):
        create_dataloaders(cfg.data, training_cfg=cfg.training)

    with pytest.raises(ValueError, match="[Mm]igrat"):
        create_dataloaders(cfg.data, training_cfg=cfg.training)


def test_streaming_run_reports_the_window_token_count_for_a_max_tokens_budget(
    streaming_run, capsys
):
    """training.max_tokens is converted down to complete optimizer steps and
    the run reports the resulting window-token count (not unique source
    tokens)."""
    trainer = streaming_run(training_overrides={"max_tokens": 100})
    out = capsys.readouterr().out

    # One step consumes batch_size x seq_len = 8 window tokens: 100 -> 12
    # complete steps -> 96 window tokens.
    assert trainer.total_steps == 12
    assert trainer.budget_name == "max_tokens"
    assert trainer.total_train_tokens == 96

    assert "Window tokens: 96" in out
    assert "training.max_tokens=100" in out
    assert "window tokens" in out.lower()


def test_streaming_run_reports_the_window_token_total_for_a_max_steps_budget(
    streaming_run, capsys
):
    """A max_steps budget reports its window-token total with the window
    wording (throughput accounting, not unique source tokens)."""
    streaming_run(training_overrides={"max_steps": 7})
    out = capsys.readouterr().out

    assert "Window tokens: 56" in out
    assert "budget=max_steps" in out


def test_streaming_run_rejects_a_max_tokens_budget_smaller_than_one_step(streaming_run):
    """A window-token budget that does not cover one complete optimizer step
    cannot be converted to any step count, so it fails clearly instead of
    silently running zero steps."""
    # One step consumes batch_size x seq_len = 8 window tokens.
    with pytest.raises(ValueError, match="smaller than one complete optimizer step"):
        streaming_run(training_overrides={"max_tokens": 7})


def test_streaming_run_rejects_a_nonpositive_step_budget(streaming_run):
    with pytest.raises(ValueError, match="training\\.max_steps"):
        streaming_run(training_overrides={"max_steps": 0})


def test_streaming_run_with_max_steps_stops_at_the_requested_complete_steps(
    streaming_run, tmp_path
):
    """A streaming run with training.max_steps trains exactly that many
    complete optimizer steps, cycling the source as needed, then finishes
    (final + latest checkpoints, no overrun)."""
    trainer = streaming_run(training_overrides={"max_steps": 3}, run=True)

    assert trainer.step == 3
    assert (tmp_path / "runs" / "final_model.pt").is_file()
    assert (tmp_path / "runs" / "latest.pt").is_file()


def test_streaming_progress_lines_label_throughput_as_window_tokens(
    streaming_run, capsys
):
    """Streaming progress/throughput output identifies the counters as
    window tokens (not unique source tokens) and drops the synthetic
    data-pass label."""
    streaming_run(training_overrides={"max_steps": 2, "log_interval": 1}, run=True)
    out = capsys.readouterr().out

    progress_lines = [
        line for line in out.splitlines() if "loss" in line and "tok/s" in line
    ]

    assert len(progress_lines) == 2
    for line in progress_lines:
        assert "window tokens" in line
        assert "window tok/s" in line
        assert "data pass" not in line


def test_nonstreaming_progress_lines_keep_their_existing_wording(tmp_path, capsys):
    """Nonstreaming runs keep the established progress wording (data pass /
    tokens / tok/s); the window-token labels are streaming-only."""
    from test_training_loop import build_stub_trainer

    trainer = build_stub_trainer(
        tmp_path, training_overrides={"log_interval": 1}, total_steps=2
    )

    trainer.train()
    out = capsys.readouterr().out

    progress_lines = [
        line for line in out.splitlines() if "loss" in line and "tok/s" in line
    ]
    assert len(progress_lines) == 2
    for line in progress_lines:
        assert "data pass" in line
        assert "window tokens" not in line


def test_nonstreaming_paths_keep_ignoring_data_max_tokens(monkeypatch, tmp_path):
    """The nonstreaming paths (files source) retain their existing behavior:
    data.max_tokens is simply not part of their contract and is ignored."""
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(" ".join(str(i % VOCAB) for i in range(400)), encoding="utf-8")
    monkeypatch.setattr(
        dataloader_module, "get_tokenizer", lambda _name: FakeTokenizer()
    )

    cfg = streaming_cfg(
        data_overrides={
            "source": "files",
            "files": [str(corpus)],
            "streaming": False,
            "max_tokens": 1000,
            "val_ratio": 0.1,
        }
    )

    train_loader, val_loader = create_dataloaders(cfg.data, training_cfg=cfg.training)

    assert len(train_loader) > 0
    assert len(val_loader) > 0
