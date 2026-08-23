"""Build and read disk-backed token shards.

The normal text-file loader is intentionally simple, but concatenating the
whole Gutenberg corpus into one Python string does not scale.  This module
provides a small, reusable alternative:

* :func:`build_token_shards` tokenizes one source file at a time and writes
  fixed-size binary token shards plus a JSON manifest.
* :class:`TokenShardDataset` memory-maps those shards and exposes the same
  ``(x, y)`` sliding-window interface as :class:`data.dataset.GPTDataset`.

The on-disk format is deliberately not Parquet.  Training needs contiguous
token ranges, so a flat binary representation avoids row decoding and keeps
the hot path close to the existing dataset implementation.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.tokenizer import get_tokenizer

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
_WORKER_TOKENIZER = None
_WORKER_STORAGE_DTYPE = None


def _encode_file(path: str | os.PathLike[str], tokenizer, storage_dtype: np.dtype):
    """Read and encode one file, returning compact token storage."""
    text = Path(path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    eos_token = getattr(tokenizer, "eot_token", None)
    if eos_token is not None:
        ids.append(int(eos_token))
    if not ids:
        return str(path), np.empty(0, dtype=storage_dtype)

    max_id = max(ids)
    if max_id >= tokenizer.n_vocab or max_id > np.iinfo(storage_dtype).max:
        raise ValueError(f"Tokenizer produced an invalid token in {path}")
    return str(path), np.asarray(ids, dtype=storage_dtype)


def _init_tokenizer_worker(tokenizer_name: str, storage_dtype_name: str) -> None:
    """Initialize one tokenizer per worker process."""
    global _WORKER_TOKENIZER, _WORKER_STORAGE_DTYPE
    _WORKER_TOKENIZER = get_tokenizer(tokenizer_name)
    _WORKER_STORAGE_DTYPE = np.dtype(storage_dtype_name)


def _encode_file_worker(path: str) -> tuple[str, np.ndarray]:
    """Multiprocessing entry point for encoding one file."""
    return _encode_file(path, _WORKER_TOKENIZER, _WORKER_STORAGE_DTYPE)


def _resolve_input_files(input_path: str | os.PathLike[str]) -> list[Path]:
    """Resolve a directory, file, or recursive glob into sorted text files."""
    raw_path = os.fspath(input_path)
    if glob.has_magic(raw_path):
        paths = [Path(path) for path in glob.glob(raw_path, recursive=True)]
    else:
        path = Path(raw_path)
        if path.is_dir():
            paths = list(path.rglob("*.txt"))
        elif path.is_file():
            paths = [path]
        else:
            paths = []

    paths = sorted(path for path in paths if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No text files found for input {input_path!r}")
    return paths


def _storage_types(tokenizer) -> tuple[np.dtype, torch.dtype, str]:
    """Choose the smallest safe integer type for token IDs."""
    if tokenizer.n_vocab <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16), torch.uint16, "uint16"
    if tokenizer.n_vocab <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32), torch.uint32, "uint32"
    raise ValueError(
        f"Tokenizer vocabulary is too large for the shard format: "
        f"{tokenizer.n_vocab} entries"
    )


def _remove_generated_files(output_dir: Path, generated_files: Iterable[Path]) -> None:
    """Remove only files created by a failed shard build."""
    for path in generated_files:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        (output_dir / MANIFEST_NAME).unlink()
    except FileNotFoundError:
        pass


def build_token_shards(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    tokenizer_name: str = "gpt2",
    shard_tokens: int = 64_000_000,
    val_ratio: float = 0.1,
    max_files: int | None = None,
    overwrite: bool = False,
    workers: int = 1,
    tokenizer=None,
) -> dict[str, Any]:
    """Tokenize text files into memory-mappable binary shards.

    Files are read and tokenized one at a time.  With ``workers > 1``, worker
    processes encode files in parallel while the parent keeps output ordering
    and performs all shard writes.  A tokenizer EOS token is inserted after
    every file so a training window does not silently join two books as if they
    were one document.  The train/validation split is a deterministic
    token-position split, matching the existing loader.

    The returned dictionary is also written to ``manifest.json`` in
    ``output_dir``.
    """
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be greater than 0 and less than 1")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive when provided")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers > 1 and tokenizer is not None:
        raise ValueError(
            "A custom tokenizer cannot be used with workers > 1; "
            "pass tokenizer_name instead"
        )

    paths = _resolve_input_files(input_path)
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise ValueError("max_files excluded every input file")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    existing_shards = list(output_dir.glob("shard_*.bin"))
    if (manifest_path.exists() or existing_shards) and not overwrite:
        raise FileExistsError(
            f"Token cache already exists at {output_dir}; use overwrite=True "
            "or choose a new output directory"
        )
    if overwrite:
        for path in existing_shards:
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    tokenizer = tokenizer or get_tokenizer(tokenizer_name)
    np_dtype, torch_dtype, storage_dtype = _storage_types(tokenizer)
    eos_token = getattr(tokenizer, "eot_token", None)
    buffer = np.empty(shard_tokens, dtype=np_dtype)
    shard_fill = 0
    shard_index = 0
    total_tokens = 0
    shard_metadata: list[dict[str, Any]] = []
    generated_files: list[Path] = []

    def flush_shard(token_count: int) -> None:
        nonlocal shard_index
        if token_count <= 0:
            return
        shard_path = output_dir / f"shard_{shard_index:05d}.bin"
        buffer[:token_count].tofile(shard_path)
        generated_files.append(shard_path)
        shard_metadata.append(
            {
                "file": shard_path.name,
                "tokens": int(token_count),
            }
        )
        shard_index += 1

    def consume_encoded_files(encoded_files) -> None:
        nonlocal total_tokens, shard_fill
        for _, values in tqdm(encoded_files, total=len(paths), desc="Tokenizing files"):
            if values.size == 0:
                continue

            total_tokens += int(values.size)
            source_offset = 0
            while source_offset < values.size:
                available = shard_tokens - shard_fill
                take = min(available, values.size - source_offset)
                buffer[shard_fill : shard_fill + take] = values[
                    source_offset : source_offset + take
                ]
                shard_fill += take
                source_offset += take
                if shard_fill == shard_tokens:
                    flush_shard(shard_fill)
                    shard_fill = 0

    try:
        if workers == 1:
            consume_encoded_files(
                _encode_file(path, tokenizer, np_dtype) for path in paths
            )
        else:
            context = mp.get_context("spawn")
            with context.Pool(
                processes=workers,
                initializer=_init_tokenizer_worker,
                initargs=(tokenizer_name, np_dtype.name),
            ) as pool:
                consume_encoded_files(
                    pool.imap(
                        _encode_file_worker,
                        [os.fspath(path) for path in paths],
                        chunksize=1,
                    )
                )

        flush_shard(shard_fill)
    except Exception:
        _remove_generated_files(output_dir, generated_files)
        raise

    train_tokens = int(total_tokens * (1.0 - val_ratio))
    manifest: dict[str, Any] = {
        "format_version": MANIFEST_VERSION,
        "tokenizer": tokenizer_name,
        "vocab_size": int(tokenizer.n_vocab),
        "eos_token_id": int(eos_token) if eos_token is not None else None,
        "storage_dtype": storage_dtype,
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "shard_tokens": int(shard_tokens),
        "total_tokens": int(total_tokens),
        "train_tokens": train_tokens,
        "val_tokens": int(total_tokens - train_tokens),
        "val_ratio": float(val_ratio),
        "source": {
            "input": os.fspath(input_path),
            "file_count": len(paths),
            "max_files": max_files,
            "files": [os.fspath(path) for path in paths],
        },
        "shards": shard_metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and minimally validate a token-shard manifest."""
    path = Path(path)
    manifest_path = path / MANIFEST_NAME if path.is_dir() else path
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported token cache version: {manifest.get('format_version')}"
        )
    if manifest.get("storage_dtype") not in {"uint16", "uint32"}:
        raise ValueError("Manifest has an unsupported storage dtype")
    if not manifest.get("shards"):
        raise ValueError("Manifest does not contain any token shards")
    return manifest


class TokenShardDataset(Dataset):
    """Sliding-window dataset backed by lazily opened memory-mapped shards."""

    def __init__(
        self,
        cache_path: str | os.PathLike[str],
        seq_len: int,
        stride: int,
        split: str = "train",
        tokenizer_name: str | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        cache_path = Path(cache_path)
        manifest_path = (
            cache_path / MANIFEST_NAME if cache_path.is_dir() else cache_path
        )
        self.cache_dir = manifest_path.parent
        self.manifest = load_manifest(manifest_path)
        if tokenizer_name and self.manifest["tokenizer"] != tokenizer_name:
            raise ValueError(
                f"Token cache uses tokenizer {self.manifest['tokenizer']!r}, "
                f"but the run requests {tokenizer_name!r}"
            )

        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.split = split
        if split == "train":
            segment_start = 0
            segment_end = int(self.manifest["train_tokens"])
        else:
            segment_start = int(self.manifest["train_tokens"])
            segment_end = int(self.manifest["total_tokens"])

        torch_dtype_name = self.manifest["torch_dtype"]
        try:
            self._torch_dtype = getattr(torch, torch_dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype {torch_dtype_name!r}") from exc

        self._ranges: list[tuple[int, str, int, int]] = []
        self._cumulative_samples: list[int] = []
        token_offset = 0
        sample_total = 0
        for shard_number, shard in enumerate(self.manifest["shards"]):
            shard_tokens = int(shard["tokens"])
            shard_start = token_offset
            shard_end = shard_start + shard_tokens
            range_start = max(segment_start, shard_start)
            range_end = min(segment_end, shard_end)
            available = range_end - range_start
            sample_count = max(0, (available - self.seq_len - 1) // self.stride + 1)
            if sample_count:
                self._ranges.append(
                    (
                        shard_number,
                        str(shard["file"]),
                        range_start - shard_start,
                        sample_count,
                    )
                )
                sample_total += sample_count
                self._cumulative_samples.append(sample_total)
            token_offset = shard_end

        if sample_total == 0:
            raise ValueError(
                f"Token cache split {split!r} is too short for seq_len={seq_len}"
            )
        self._mmap_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self._cumulative_samples[-1]

    def _get_shard(self, shard_index: int) -> torch.Tensor:
        tokens = self._mmap_cache.get(shard_index)
        if tokens is None:
            shard_number, filename, _, _ = self._ranges[shard_index]
            shard = self.manifest["shards"][shard_number]
            tokens = torch.from_file(
                str(self.cache_dir / filename),
                shared=False,
                size=int(shard["tokens"]),
                dtype=self._torch_dtype,
            )
            self._mmap_cache[shard_index] = tokens
        return tokens

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"dataset index out of range: {index}")
        shard_index = bisect.bisect_right(self._cumulative_samples, index)
        previous_samples = (
            self._cumulative_samples[shard_index - 1] if shard_index else 0
        )
        _, _, local_start, _ = self._ranges[shard_index]
        start = local_start + (index - previous_samples) * self.stride
        tokens = self._get_shard(shard_index)
        x = tokens[start : start + self.seq_len].to(dtype=torch.long)
        y = tokens[start + 1 : start + self.seq_len + 1].to(dtype=torch.long)
        return x, y

    def __getstate__(self):
        """Do not pickle open mappings into spawned DataLoader workers."""
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}
        return state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize text files into memory-mappable training shards."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory, text file, or recursive glob.",
    )
    parser.add_argument("--output", required=True, help="Output token-cache directory.")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=64_000_000,
        help="Tokens per binary shard (default: 64 million).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel tokenizer processes; the parent writes shards in order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing shard_*.bin files and manifest in the output directory.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    manifest = build_token_shards(
        args.input,
        args.output,
        tokenizer_name=args.tokenizer,
        shard_tokens=args.shard_tokens,
        val_ratio=args.val_ratio,
        max_files=args.max_files,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    print(
        f"Wrote {len(manifest['shards']):,} shards with "
        f"{manifest['total_tokens']:,} tokens to {args.output}"
    )


if __name__ == "__main__":
    main()
