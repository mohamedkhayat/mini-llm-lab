import json
import os
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from huggingface_hub import snapshot_download
from hydra.utils import to_absolute_path
from tqdm import tqdm

from data.budget import parse_token_budget, split_cache_tokens
from data.tokenizer import get_tokenizer


_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def get_data_dir(cfg):
    safe_tok_name = cfg.data.tokenizer_name.replace("/", "__")
    safe_ds_name = cfg.data.hf_dataset.replace("/", "__")
    safe_cfg_name = getattr(cfg.data, "hf_config", None)
    path_parts = [safe_ds_name]
    if safe_cfg_name:
        path_parts.append(str(safe_cfg_name).replace("/", "__"))

    return Path(to_absolute_path("data")) / "__".join(path_parts) / safe_tok_name


def _verify_prefix_preserved(old_path, new_path, dtype_name):
    """Compare an old cache prefix with its atomically prepared replacement."""
    dtype = _DTYPES.get(dtype_name)
    if dtype is None:
        raise RuntimeError(
            "Cannot verify cache prefix preservation: the existing cache has "
            f"unknown dtype {dtype_name!r}. Start a fresh run instead."
        )

    itemsize = np.dtype(dtype).itemsize
    old_tokens = old_path.stat().st_size // itemsize
    new_tokens = new_path.stat().st_size // itemsize
    if new_tokens < old_tokens:
        raise RuntimeError(
            "Refusing to replace a resumable cache with a shorter train file; "
            f"old prefix={old_tokens:,}, new prefix={new_tokens:,}."
        )

    old_arr = np.memmap(old_path, dtype=dtype, mode="r")
    new_arr = np.memmap(new_path, dtype=dtype, mode="r")
    try:
        chunk_size = 1_000_000
        for start in range(0, old_tokens, chunk_size):
            end = min(start + chunk_size, old_tokens)
            if not np.array_equal(old_arr[start:end], new_arr[start:end]):
                raise RuntimeError(
                    "Refusing to replace the cache: regenerated tokens do not "
                    f"match the existing train prefix at token {start:,}."
                )
    finally:
        del old_arr
        del new_arr


def prepare_dataset(cfg, num_proc=8, preserve_prefix=False):

    output_dir = get_data_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    eval_path = output_dir / "eval.bin"
    meta_path = output_dir / "meta.json"
    train_tmp_path = output_dir / "train.bin.tmp"
    eval_tmp_path = output_dir / "eval.bin.tmp"
    meta_tmp_path = output_dir / "meta.json.tmp"

    cached_revision = None
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_revision = json.load(f).get("source_revision")
        except (OSError, json.JSONDecodeError):
            cached_revision = None

    snapshot_kwargs = {
        "repo_id": cfg.data.hf_dataset,
        "repo_type": "dataset",
    }
    configured_revision = getattr(cfg.data, "revision", None)
    revision = configured_revision or cached_revision
    if revision:
        snapshot_kwargs["revision"] = revision
    local_dir = snapshot_download(**snapshot_kwargs)
    source_revision = Path(local_dir).name
    dataset = ds.dataset(
        local_dir, format=cfg.data.file_format, exclude_invalid_files=True
    )
    enc = get_tokenizer(cfg.data.tokenizer_name)

    # data.max_tokens is the total cache budget; train/eval sizes are derived
    # once from val_ratio so every consumer can verify the same invariant.
    total_len = parse_token_budget(
        cfg.data.max_tokens, "data.max_tokens", allow_none=False
    )
    train_arr_len, val_arr_len = split_cache_tokens(
        total_len, float(cfg.data.val_ratio)
    )
    if train_arr_len <= 0 or val_arr_len <= 0:
        raise ValueError(
            "data.max_tokens and data.val_ratio must leave positive train and "
            f"validation caches; got total={total_len}, val_ratio={cfg.data.val_ratio}"
        )

    dtype = np.uint16 if enc.max_token_value < 65536 else np.uint32
    dtype_str = "uint16" if dtype == np.uint16 else "uint32"

    # Build into sidecar files. Reprocessing can therefore be interrupted or
    # fail because the source is short without destroying the last valid
    # cache that a checkpoint may still need.
    train_arr = np.memmap(
        train_tmp_path, dtype=dtype, mode="w+", shape=(train_arr_len,)
    )
    val_arr = np.memmap(
        eval_tmp_path, dtype=dtype, mode="w+", shape=(val_arr_len,)
    )
    idx = 0

    current_arr = train_arr
    current_arr_len = train_arr_len
    done = False
    pbar = tqdm(total=total_len, unit="tok", unit_scale=True, desc="Tokenizing")
    for batch in tqdm(
        dataset.to_batches(columns=[cfg.data.text_column], batch_size=10_000)
    ):
        if done:
            break
        texts = batch.column(cfg.data.text_column).to_pylist()
        token_lists = enc.encode_ordinary_batch(
            texts, num_threads=num_proc
        )  # tokenize the text in batches
        for ids in token_lists:  # I go over each tokenized sequence
            ids.append(enc.eot_token)  # and append the eos token
            # A source row may straddle the train/eval boundary. Carry its
            # remainder into the next mmap instead of silently dropping it;
            # this also handles an exact boundary fill without leaving the
            # validation cache unwritten.
            remaining = ids
            while remaining and not done:
                available = current_arr_len - idx
                take = min(len(remaining), available)
                current_arr[idx : idx + take] = remaining[:take]
                idx += take
                pbar.update(take)
                remaining = remaining[take:]

                if idx == current_arr_len:
                    if current_arr is val_arr:
                        done = True
                        break
                    current_arr = val_arr
                    current_arr_len = val_arr_len
                    idx = 0
        if done:
            break
    pbar.close()
    if not done:
        raise RuntimeError(
            f"Source dataset exhausted before reaching token budget: "
            f"wrote {idx} tokens into {'val' if current_arr is val_arr else 'train'}.bin, "
            f"needed {current_arr_len}."
        )
    train_arr.flush()
    val_arr.flush()
    del train_arr
    del val_arr

    if preserve_prefix and train_path.is_file():
        old_dtype = None
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old_dtype = json.load(f).get("dtype")
            except (OSError, json.JSONDecodeError):
                old_dtype = None
        _verify_prefix_preserved(train_path, train_tmp_path, old_dtype)

    meta = {
        "tokenizer_name": cfg.data.tokenizer_name,
        "hf_dataset": cfg.data.hf_dataset,
        "hf_config": getattr(cfg.data, "hf_config", None),
        "source_revision": source_revision,
        "text_column": cfg.data.text_column,
        "file_format": cfg.data.file_format,
        "dtype": dtype_str,
        "max_tokens": total_len,
        "train_tokens": train_arr_len,
        "val_tokens": val_arr_len,
        "val_ratio": cfg.data.val_ratio,
    }

    with open(meta_tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    os.replace(train_tmp_path, train_path)
    os.replace(eval_tmp_path, eval_path)
    os.replace(meta_tmp_path, meta_path)

    print(
        f"Finished writing {train_arr_len:,} train tokens and {val_arr_len:,} eval tokens."
    )
