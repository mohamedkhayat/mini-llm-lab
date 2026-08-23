import json

import torch

from data.token_shards import TokenShardDataset, build_token_shards


class FakeTokenizer:
    n_vocab = 32
    eot_token = 31

    def encode(self, text):
        return [index % 30 for index, _ in enumerate(text)]


def test_token_shards_are_written_and_read_as_sliding_windows(tmp_path):
    input_dir = tmp_path / "text"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("abcdefgh", encoding="utf-8")
    (input_dir / "b.txt").write_text("ijklmnop", encoding="utf-8")
    (input_dir / "c.txt").write_text("qrstuvwx", encoding="utf-8")
    cache_dir = tmp_path / "cache"

    manifest = build_token_shards(
        input_dir,
        cache_dir,
        tokenizer_name="fake",
        shard_tokens=5,
        val_ratio=0.25,
        tokenizer=FakeTokenizer(),
    )

    assert manifest["total_tokens"] == 27
    assert manifest["train_tokens"] == 20
    assert len(manifest["shards"]) == 6
    assert json.loads((cache_dir / "manifest.json").read_text())["storage_dtype"] == (
        "uint16"
    )

    train = TokenShardDataset(
        cache_dir,
        seq_len=3,
        stride=3,
        split="train",
        tokenizer_name="fake",
    )
    x, y = train[0]

    assert len(train) == 4
    assert x.dtype == torch.long
    torch.testing.assert_close(x, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(y, torch.tensor([1, 2, 3]))
    torch.testing.assert_close(train[3][0], torch.tensor([6, 7, 31]))


def test_token_cache_rejects_a_different_tokenizer(tmp_path):
    input_dir = tmp_path / "text"
    input_dir.mkdir()
    (input_dir / "book.txt").write_text("abcdefgh", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    build_token_shards(
        input_dir,
        cache_dir,
        tokenizer_name="fake",
        shard_tokens=32,
        val_ratio=0.25,
        tokenizer=FakeTokenizer(),
    )

    try:
        TokenShardDataset(
            cache_dir,
            seq_len=2,
            stride=2,
            split="train",
            tokenizer_name="gpt2",
        )
    except ValueError as exc:
        assert "tokenizer" in str(exc)
    else:
        raise AssertionError("Expected tokenizer mismatch to fail")
