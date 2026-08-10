import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import GPTDataset, get_train_val_split
from data.tokenizer import get_tokenizer


def _load_text_from_files(files: list[str]) -> str:
    """Glob ``files`` patterns, read every ``.txt``, concatenate."""
    all_paths: list[str] = []
    for pattern in files:
        all_paths.extend(glob.glob(pattern, recursive=True))

    if not all_paths:
        raise FileNotFoundError(f"No files matched: {files}")

    parts: list[str] = []
    for p in tqdm(all_paths, desc="Reading files"):
        with open(p, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


def _load_text_from_hf_dataset(cfg) -> str:
    """Load and concatenate text from a Hugging Face dataset."""
    from datasets import load_dataset

    name = cfg.hf_dataset
    config = cfg.hf_config if cfg.hf_config else None
    split = cfg.split
    text_column = cfg.text_column
    streaming = cfg.streaming

    if streaming:
        ds = load_dataset(name, config, split=split, streaming=True)
        return _stream_chunks_to_string(ds, text_column)

    if config:
        ds = load_dataset(name, config, split=split)
    else:
        ds = load_dataset(name, split=split)

    parts: list[str] = []
    for example in tqdm(ds, desc="Loading HF dataset"):
        text = example[text_column]
        if isinstance(text, list):
            text = " ".join(text)
        parts.append(text)
    return "\n".join(parts)


def _stream_chunks_to_string(ds, text_column: str) -> str:
    """Iterate a streaming HF dataset and buffer text.

    Returns concatenated string. For very large corpora consider
    token-caching (TODO).
    """
    parts: list[str] = []
    for example in tqdm(ds, desc="Streaming HF dataset"):
        text = example[text_column]
        if isinstance(text, list):
            text = " ".join(text)
        parts.append(text)
    return "\n".join(parts)


def _ensure_sample_data() -> str:
    """Download the small sample corpus when no local files are available."""
    verdict_path = Path("./data/the_verdict")
    if not verdict_path.exists():
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
            "main/ch02/01_main-chapter-code/the-verdict.txt"
        )
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, verdict_path)  # noqa: S310
        print(f"Downloaded sample data to {verdict_path}")

    return verdict_path.read_text(encoding="utf-8")


def create_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from a Hydra ``cfg.data`` node.

    Supports two source types via ``cfg.data.source``:

    - ``"files"``: local text files (glob patterns)
    - ``"hf_dataset"``: Hugging Face Hub dataset

    Returns ``(train_loader, val_loader)``.
    """
    source = getattr(cfg, "source", "files")
    seed = int(getattr(cfg, "seed", 42))
    torch.manual_seed(seed)

    # --- Load raw text ---
    if source == "hf_dataset":
        text = _load_text_from_hf_dataset(cfg)
    else:
        files = list(cfg.files) if cfg.files else []
        if files:
            try:
                text = _load_text_from_files(files)
            except FileNotFoundError:
                text = _ensure_sample_data()
        else:
            text = _ensure_sample_data()

    # --- Tokenize ---
    tokenizer = get_tokenizer(cfg.tokenizer_name)
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    print(f"Total tokens: {len(token_tensor):,}")

    # --- Split ---
    val_ratio = float(getattr(cfg, "val_ratio", 0.1))
    train_tokens, val_tokens = get_train_val_split(token_tensor, val_ratio, seed)
    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

    # --- Datasets ---
    seq_len = int(cfg.seq_len)
    stride = int(getattr(cfg, "stride", seq_len))
    train_ds = GPTDataset(train_tokens, seq_len, stride)
    val_ds = GPTDataset(val_tokens, seq_len, stride)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "The tokenized corpus is too short for the configured seq_len and "
            "val_ratio; need at least one train and validation window."
        )
    print(f"Train steps: {len(train_ds):,} | Val steps: {len(val_ds):,}")

    # --- DataLoaders ---
    bs = int(cfg.batch_size)
    nw = int(getattr(cfg, "num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=bool(getattr(cfg, "shuffle", True)),
        drop_last=bool(getattr(cfg, "drop_last", True)),
        num_workers=nw,
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg, "persistent_workers", True))
        if nw > 0
        else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg, "persistent_workers", True))
        if nw > 0
        else False,
    )

    return train_loader, val_loader
