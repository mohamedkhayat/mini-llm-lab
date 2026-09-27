import pytest
import tiktoken
from datasets import Dataset
from omegaconf import OmegaConf

import data.hf_streaming as hf_streaming
from data.dataloader import _create_hf_streaming_dataloaders


class FakeTokenizer:
    eot_token = 999

    def encode(self, text, disallowed_special=()):
        return [int(value) for value in text.split()]


def _source_factory():
    rows = [
        "0 1 2 3 4 5 6 7 8 9 10 11",
        "20 21 22 23 24 25 26 27 28 29 30 31",
        "40 41 42 43 44 45 46 47 48 49 50 51",
        "60 61 62 63 64 65 66 67 68 69 70 71",
    ]
    return Dataset.from_dict({"text": rows}).to_iterable_dataset(num_shards=2)


def _dataset(monkeypatch, max_samples=2, val_ratio=0.5, rows=None, seq_len=4):
    monkeypatch.setattr(hf_streaming, "get_tokenizer", lambda _: FakeTokenizer())
    if rows is None:
        rows = [
            "0 1 2 3 4 5 6 7 8 9 10 11",
            "20 21 22 23 24 25 26 27 28 29 30 31",
            "40 41 42 43 44 45 46 47 48 49 50 51",
            "60 61 62 63 64 65 66 67 68 69 70 71",
        ]
    source_factory = lambda: Dataset.from_dict({"text": rows}).to_iterable_dataset(
        num_shards=1
    )
    return hf_streaming.HFStreamingWindowDataset(
        dataset_name="local/test",
        dataset_config=None,
        split="train",
        text_column="text",
        tokenizer_name="fake",
        stride=4,
        val_ratio=val_ratio,
        split_name="train",
        seq_len=seq_len,
        max_samples=max_samples,
        shuffle=False,
        persistent_state=True,
        source_factory=source_factory,
    )


def test_hf_window_adapter_resumes_mid_document_from_native_source_state(monkeypatch):
    dataset = _dataset(monkeypatch, max_samples=1)
    first = next(iter(dataset))[0].tolist()
    state = dataset.state_dict()

    assert first == [20, 21, 22, 23]
    assert state["row_index"] == 1
    # The row is fully appended to the packing buffer; on resume the row
    # must be re-read from the source but NOT appended again.
    assert state["row_offset"] == hf_streaming.HFStreamingWindowDataset.ROW_APPENDED
    assert state["source_state"] is not None

    resumed = _dataset(monkeypatch, max_samples=2)
    resumed.load_state_dict(state)
    windows = [x.tolist() for x, _ in resumed]

    assert windows == [[24, 25, 26, 27], [28, 29, 30, 31]]


def test_hf_window_adapter_delegates_source_position_without_replaying_rows(
    monkeypatch,
):
    dataset = _dataset(monkeypatch, max_samples=2)
    first = [x.tolist() for x, _ in dataset]
    state = dataset.state_dict()

    resumed = _dataset(monkeypatch, max_samples=2)
    resumed.load_state_dict(state)
    second = [x.tolist() for x, _ in resumed]

    assert first == [[20, 21, 22, 23], [24, 25, 26, 27]]
    # Packed continuation: the next window starts at the buffer remainder
    # (999 = row 1's trailing EOS), not at the head of row 3.
    assert second == [[28, 29, 30, 31], [999, 60, 61, 62]]


def test_hf_window_adapter_rejects_a_different_stream_identity(monkeypatch):
    dataset = _dataset(monkeypatch)
    state = dataset.state_dict()
    state["identity"]["stride"] = 1

    other = _dataset(monkeypatch)
    try:
        other.load_state_dict(state)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("a changed stream identity must be rejected")


def test_hf_window_adapter_resume_ignores_shuffle_buffer_size(monkeypatch):
    # The shuffle buffer is refilled (not restored) on resume, so its size
    # does not affect the resumed data cursor. A checkpoint saved with one
    # buffer size must resume with another (tuned for host-RAM safety).
    dataset = _dataset(monkeypatch)
    state = dataset.state_dict()
    state["identity"]["shuffle_buffer_size"] = 2000  # != the 10000 default

    other = _dataset(monkeypatch)
    other.load_state_dict(state)  # must not raise


def test_packing_spans_short_rows_with_eos_boundary(monkeypatch):
    # Rows shorter than seq_len yield zero within-row windows. Packing must
    # concatenate rows (each already EOS-terminated) into one token stream
    # so short rows still contribute windows.
    # Train rows (odd indices, val_ratio 0.5): "2 3", "6 7", "10 11".
    rows = [f"{i * 2} {i * 2 + 1}" for i in range(6)]
    dataset = _dataset(monkeypatch, max_samples=2, rows=rows)
    windows = [x.tolist() for x, _ in dataset]

    # Packed train stream: [2,3,999, 6,7,999, 10,11,999]; each window
    # spans a row boundary with the EOS separator inside it.
    assert windows == [[2, 3, 999, 6], [7, 999, 10, 11]]


def test_packing_uses_all_rows_without_replay(monkeypatch):
    rows = [f"{i * 2} {i * 2 + 1}" for i in range(6)]
    dataset = _dataset(monkeypatch, max_samples=2, rows=rows)
    windows = [x.tolist() for x, _ in dataset]
    assert len(windows) == 2  # 9 packed tokens -> 2 windows, [999] leftover

    state = dataset.state_dict()
    assert state["buffer"] == [999]  # sub-window remainder carried in state

    # No third window exists in this pass: the remainder is < seq_len + 1
    # and the next (cycled) pass would start a fresh stream.
    assert state["source_pass"] == 0


def test_packing_resumes_across_a_row_boundary(monkeypatch):
    rows = [f"{i * 2} {i * 2 + 1}" for i in range(6)]
    first_ds = _dataset(monkeypatch, max_samples=1, rows=rows)
    first = [x.tolist() for x, _ in first_ds]
    state = first_ds.state_dict()
    assert first == [[2, 3, 999, 6]]

    resumed = _dataset(monkeypatch, max_samples=1, rows=rows)
    resumed.load_state_dict(state)
    second = [x.tolist() for x, _ in resumed]

    # Continues exactly where the first stream stopped: the buffer
    # remainder [7, 999] plus row 5, with no replay and no re-append.
    assert second == [[7, 999, 10, 11]]


def test_state_saved_by_a_different_schema_version_is_rejected(monkeypatch):
    dataset = _dataset(monkeypatch)
    state = dataset.state_dict()
    state["version"] = 1

    other = _dataset(monkeypatch)
    with pytest.raises(ValueError, match="version"):
        other.load_state_dict(state)


class _NoStateShuffled:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter([{"text": row} for row in self.rows])


class _PreShuffleOnlyStateSource:
    """Raw source has the state API; the shuffled wrapper (the object that
    actually iterates) does not."""

    rows = ["0 1 2 3 4 5 6 7 8 9 10 11", "20 21 22 23 24 25 26 27 28 29 30 31"]

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def set_epoch(self, epoch):
        pass

    def shuffle(self, **kwargs):
        return _NoStateShuffled(self.rows)


def test_capability_check_runs_on_the_post_shuffle_source(monkeypatch):
    monkeypatch.setattr(hf_streaming, "get_tokenizer", lambda _: FakeTokenizer())
    dataset = hf_streaming.HFStreamingWindowDataset(
        dataset_name="local/test",
        dataset_config=None,
        split="train",
        text_column="text",
        tokenizer_name="fake",
        seq_len=4,
        stride=4,
        val_ratio=0.5,
        split_name="train",
        max_samples=1,
        shuffle=True,
        persistent_state=True,
        source_factory=_PreShuffleOnlyStateSource,
    )

    with pytest.raises(RuntimeError, match="upgrade datasets"):
        next(iter(dataset))


def _streaming_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "hf_dataset": "local/test",
            "hf_config": None,
            "split": "train",
            "text_column": "text",
            "tokenizer_name": "gpt2",
            "seq_len": 4,
            "stride": 4,
            "val_ratio": 0.5,
            "seed": 42,
            "shuffle": False,
            "shuffle_buffer_size": 10,
            "batch_size": 2,
            "num_workers": 0,
            "drop_last": True,
            "pin_memory": False,
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_eval_datasets_skip_the_hf_shuffle_wrapper():
    # datasets 5.x + pyarrow: the streaming .shuffle() wrapper retains ~10GB
    # of host RAM per live dataset instance. The trainer keeps three
    # instances alive (train, train_eval, val) and every eval materializes
    # the two eval ones, which OOMed a 60GiB machine at the first eval.
    # Eval is a bounded loss estimate over the already-upstream-shuffled
    # corpus, so it must use the unshuffled (flat ~1GB) streaming path.
    cfg = _streaming_cfg(shuffle=True, shuffle_buffer_size=10000)
    training_cfg = OmegaConf.create(
        {"max_steps": 5, "max_tokens": None, "eval_batches": 2}
    )
    train_loader, val_loader = _create_hf_streaming_dataloaders(cfg, training_cfg)
    train_ds = getattr(train_loader, "checkpoint_dataset")
    # The loaders' .dataset is typed as HF Dataset but is our adapter at
    # runtime; getattr mirrors the codebase's dynamic-attribute access
    # pattern (see the checkpoint_dataset tests above).
    eval_train_ds = getattr(getattr(train_loader, "eval_loader"), "dataset")

    assert train_ds.shuffle is True
    assert getattr(eval_train_ds, "shuffle") is False
    assert getattr(val_loader.dataset, "shuffle") is False


def test_streaming_dataloaders_refuse_forked_workers():
    # Single-process only: forked workers import datasets in the child
    # process and break wandb's import hooks (wandb.sdk ForkedError).
    cfg = _streaming_cfg(num_workers=4)
    with pytest.raises(ValueError, match="single-process"):
        _create_hf_streaming_dataloaders(cfg, None)


def test_streaming_train_dataset_spans_the_full_step_budget():
    # The train stream uses one long-lived iterator per pass: max_samples
    # must cover the whole optimizer budget, not just one batch. A
    # per-batch iterator resumes from the saved state on every batch and
    # refills HF's (non-checkpointed) shuffle buffer, dropping roughly
    # buffer_size rows per resume.
    cfg = _streaming_cfg()
    training_cfg = OmegaConf.create({"max_steps": 5, "max_tokens": None})
    train_loader, _ = _create_hf_streaming_dataloaders(cfg, training_cfg)
    # checkpoint_dataset is a dynamic DataLoader attribute (see
    # training.checkpointing); getattr mirrors the codebase access pattern.
    assert getattr(train_loader, "checkpoint_dataset").max_samples == 10  # 5 steps x 2


def test_streaming_train_dataset_spans_the_full_token_budget():
    cfg = _streaming_cfg()
    training_cfg = OmegaConf.create({"max_steps": None, "max_tokens": 80})
    train_loader, _ = _create_hf_streaming_dataloaders(cfg, training_cfg)
    # 80 window tokens / (2 batch x 4 seq_len = 8 tokens/step) = 10 steps
    # -> 20 windows per pass.
    assert getattr(train_loader, "checkpoint_dataset").max_samples == 20


def test_unsupported_val_ratio_is_rejected_at_construction(monkeypatch):
    # round(1/0.4) collapses to the 50/50 floor, so 0.4 cannot be honored.
    with pytest.raises(ValueError, match="val_ratio"):
        _dataset(monkeypatch, val_ratio=0.4)


def test_val_ratios_the_period_formula_honors_are_accepted(monkeypatch):
    assert _dataset(monkeypatch, val_ratio=0.3).val_ratio == 0.3
    assert _dataset(monkeypatch, val_ratio=0.5).val_ratio == 0.5


def test_literal_special_token_text_in_a_row_is_encoded_not_raised(monkeypatch):
    # Corpus rows (finepdfs) can contain the literal gpt2 EOT special string
    # as plain text. tiktoken's default guard raises ValueError on it and
    # killed a 17h streaming run mid-pass; such rows must be encoded as
    # normal text instead. The literal is derived from the tokenizer at
    # runtime (never spelled out in this file).
    enc = tiktoken.get_encoding("gpt2")
    eot_literal = enc.decode([enc.eot_token])
    rows = ["hello world"] * 3 + [f"see {eot_literal} here"] + ["hello world"] * 3
    dataset = _dataset(monkeypatch, max_samples=2, rows=rows, seq_len=8)
    # The real tiktoken guard (the fixture's fake tokenizer would hide it).
    dataset.tokenizer = enc
    windows = [x.tolist() for x, _ in dataset]
    assert len(windows) == 2
