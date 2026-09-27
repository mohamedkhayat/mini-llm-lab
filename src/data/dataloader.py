import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
from tqdm import tqdm

from data.dataset import GPTDataset, get_train_val_split
from data.tokenizer import get_tokenizer


def _load_text_from_files(
    files: list[str], root: Path | None = None, max_files: int | None = None
) -> str:
    """Glob ``files`` patterns, read every ``.txt``, concatenate.

    Patterns are resolved relative to ``root`` (defaults to the launch
    directory) so they keep working after Hydra changes the cwd to the
    run dir. ``max_files`` caps how many files are read (useful for huge
    corpora like gutenberg).
    """
    root = Path.cwd() if root is None else root
    all_paths: list[str] = []
    for pattern in files:
        all_paths.extend(sorted(glob.glob(str(root / pattern), recursive=True)))

    if not all_paths:
        raise FileNotFoundError(f"No files matched: {files} (searched under {root})")

    if max_files is not None:
        all_paths = all_paths[:max_files]

    parts: list[str] = []
    for p in tqdm(all_paths, desc="Reading files"):
        parts.append(Path(p).read_text(encoding="utf-8"))
    return "\n".join(parts)


def _load_text_from_hf_dataset(cfg) -> str:
    """Load a bounded, non-streaming HF dataset into memory."""
    from datasets import load_dataset

    name = cfg.hf_dataset
    config = cfg.hf_config if cfg.hf_config else None
    split = cfg.split
    text_column = cfg.text_column
    if bool(getattr(cfg, "streaming", False)):
        raise ValueError(
            "Streaming Hugging Face data must use the IterableDataset path; "
            "do not call _load_text_from_hf_dataset with streaming=True."
        )

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


def _ensure_sample_data(root: Path | None = None) -> str:
    """Download the small sample corpus when no local files are available.

    Cached under ``<root>/data/the_verdict`` so it is downloaded once and
    reused across runs (the run dir is fresh every time, so a cwd-relative
    path would re-download on every run).
    """
    root = Path.cwd() if root is None else root
    verdict_path = root / "data" / "the_verdict"
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


def _create_tokenized_dataloaders(
    cache_dir: Path, cfg
) -> tuple[DataLoader, DataLoader]:
    from data.token_shards import TokenShardDataset, load_manifest

    manifest = load_manifest(cache_dir)
    print(
        f"Using tokenized cache: {cache_dir} "
        f"({manifest['total_tokens']:,} tokens, "
        f"{len(manifest['shards']):,} shards)"
    )
    train_ds = TokenShardDataset(
        cache_dir,
        seq_len=int(cfg.seq_len),
        stride=int(getattr(cfg, "stride", cfg.seq_len)),
        split="train",
        tokenizer_name=str(cfg.tokenizer_name),
    )
    val_ds = TokenShardDataset(
        cache_dir,
        seq_len=int(cfg.seq_len),
        stride=int(getattr(cfg, "stride", cfg.seq_len)),
        split="val",
        tokenizer_name=str(cfg.tokenizer_name),
    )
    return _create_dataloaders_from_datasets(train_ds, val_ds, cfg)


def _create_hf_streaming_dataloaders(
    cfg, training_cfg
) -> tuple[DataLoader, DataLoader]:
    # Lazy: importing training.schedule at module level would cycle through
    # training/__init__ -> trainer -> data.dataloader.
    from data.hf_streaming import HFStreamingWindowDataset
    from training.schedule import resolve_total_steps

    num_workers = int(getattr(cfg, "num_workers", 0))
    if num_workers > 0:
        raise ValueError(
            "HF streaming runs are single-process: set data.num_workers=0. "
            "Forked DataLoader workers import datasets inside the child "
            "process, which breaks wandb's import hooks; rows are tokenized "
            "on the main thread."
        )
    batch_size = int(cfg.batch_size)
    # The train stream uses ONE long-lived iterator per pass: max_samples
    # covers the whole optimizer budget. HF's shuffle buffer is not
    # checkpointed; with a per-batch iterator, every batch would resume
    # from the saved state and refill the buffer, silently dropping roughly
    # buffer_size rows (and spamming the log) on each resume. With a
    # long-lived iterator the buffer stays alive; it is refilled only on
    # pass boundaries and on a real checkpoint restore (one ~buffer_size
    # row skip, negligible against the budget).
    budget_windows = batch_size
    if training_cfg is not None and (
        getattr(training_cfg, "max_steps", None) is not None
        or getattr(training_cfg, "max_tokens", None) is not None
    ):
        try:
            total_steps, _ = resolve_total_steps(
                training_cfg, 1, batch_size * int(cfg.seq_len)
            )
        except ValueError:
            pass  # Invalid budget: keep the legacy one-batch iterator; the
            # trainer's streaming budget check raises the precise error.
        else:
            budget_windows = int(total_steps) * batch_size
    eval_batches = int(
        getattr(
            training_cfg,
            "eval_batches",
            getattr(training_cfg, "eval_interval", 50),
        )
        if training_cfg is not None
        else 50
    )
    common = {
        "dataset_name": str(cfg.hf_dataset),
        "dataset_config": cfg.hf_config if cfg.hf_config else None,
        "split": str(cfg.split),
        "text_column": str(cfg.text_column),
        "tokenizer_name": str(cfg.tokenizer_name),
        "seq_len": int(cfg.seq_len),
        "stride": int(getattr(cfg, "stride", cfg.seq_len)),
        "val_ratio": float(getattr(cfg, "val_ratio", 0.1)),
        "seed": int(getattr(cfg, "seed", 42)),
        "shuffle": bool(getattr(cfg, "shuffle", True)),
        "shuffle_buffer_size": int(getattr(cfg, "shuffle_buffer_size", 10_000)),
        "max_examples": getattr(cfg, "max_examples", None),
        "max_tokens": getattr(cfg, "max_tokens", None),
    }
    train_ds = HFStreamingWindowDataset(
        **common,
        split_name="train",
        max_samples=budget_windows,
        persistent_state=True,
    )
    # Eval datasets skip the HF .shuffle() wrapper: on datasets 5.x +
    # pyarrow it retains ~10GB of host RAM per live instance, and every
    # eval materializes two of them on top of the train stream, which
    # OOMed a 60GiB machine at the first eval. Eval is a bounded loss
    # estimate over the already-upstream-shuffled corpus, so it uses the
    # unshuffled streaming path (flat ~1GB, measured).
    eval_common = {**common, "shuffle": False}
    val_ds = HFStreamingWindowDataset(
        **eval_common,
        split_name="val",
        max_samples=max(1, eval_batches * batch_size),
        persistent_state=False,
    )
    train_eval_ds = HFStreamingWindowDataset(
        **eval_common,
        split_name="train",
        max_samples=max(1, eval_batches * batch_size),
        persistent_state=False,
    )

    print(
        f"Using streaming HF dataset: {cfg.hf_dataset} "
        "(native IterableDataset state; single-process; "
        "no corpus buffering or project HF shards)"
    )
    train_loader, val_loader = _create_dataloaders_from_datasets(train_ds, val_ds, cfg)
    train_eval_loader, _ = _create_dataloaders_from_datasets(train_eval_ds, val_ds, cfg)
    train_loader.stream_stateful = True
    train_loader.checkpoint_dataset = train_ds
    train_loader.eval_loader = train_eval_loader
    return train_loader, val_loader


def create_dataloaders(cfg, training_cfg=None) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from a Hydra ``cfg.data`` node.

    Supports two source types via ``cfg.source``:

    - ``"files"``: local text files (glob patterns)
    - ``"hf_dataset"``: Hugging Face Hub dataset; ``streaming=true`` uses an
      on-the-fly IterableDataset and never concatenates the corpus

    File patterns are resolved against the launch directory (Hydra's
    ``runtime.cwd``), not the run dir Hydra switches into, so relative
    paths like ``data/gutenberg/data/text/*.txt`` keep working.

    Returns ``(train_loader, val_loader)``.
    """
    source = getattr(cfg, "source", "files")
    seed = int(getattr(cfg, "seed", 42))
    torch.manual_seed(seed)

    if source not in {"files", "hf_dataset"}:
        raise ValueError(
            f"Unknown data source {source!r}; choose 'files' or 'hf_dataset'"
        )

    # Anchor file paths to the directory the user launched from, not the
    # hydra run dir (which is fresh and empty every run).
    try:
        from hydra.core.hydra_config import HydraConfig

        root = Path(HydraConfig.get().runtime.cwd)
    except Exception:
        root = Path.cwd()

    tokenized_dir = getattr(cfg, "tokenized_dir", None)
    if tokenized_dir:
        cache_dir = Path(tokenized_dir)
        if not cache_dir.is_absolute():
            cache_dir = root / cache_dir
        return _create_tokenized_dataloaders(cache_dir, cfg)

    if source == "hf_dataset" and bool(getattr(cfg, "streaming", False)):
        data_max_tokens = getattr(cfg, "max_tokens", None)
        if data_max_tokens is not None:
            raise ValueError(
                "data.max_tokens is no longer part of the HF streaming data "
                "contract: it capped unique source tokens per data pass, which "
                "mixed a source-token limit with the window-token training "
                "budget. Migrate to the training budget: set "
                "training.max_tokens (window tokens, rounded down to complete "
                "optimizer steps of batch_size x seq_len) or "
                "training.max_steps, and remove data.max_tokens."
            )
        return _create_hf_streaming_dataloaders(cfg, training_cfg)

    max_files = int(getattr(cfg, "max_files", None) or 0) or None

    # --- Load raw text ---
    if source == "hf_dataset":
        text = _load_text_from_hf_dataset(cfg)
    else:
        files = list(cfg.files) if cfg.files else []
        if files:
            try:
                text = _load_text_from_files(files, root=root, max_files=max_files)
            except FileNotFoundError:
                text = _ensure_sample_data(root=root)
        else:
            text = _ensure_sample_data(root=root)

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

    return _create_dataloaders_from_datasets(train_ds, val_ds, cfg)


class EpochRandomSampler(Sampler[int]):
    """Deterministically shuffle a dataset once per logical data pass.

    ``DataLoader(shuffle=True)`` creates a fresh random permutation when its
    iterator is created.  That is convenient for ordinary training, but it
    makes it impossible to reconstruct the remainder of a partially consumed
    pass after a restart.  This sampler derives the permutation from
    ``seed + epoch`` instead, so a checkpoint only needs to store the epoch
    and batch cursor.
    """

    def __init__(self, data_source: Dataset, seed: int, shuffle: bool = True):
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if not self.shuffle:
            yield from range(len(self.data_source))
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def _create_dataloaders_from_datasets(
    train_ds: Dataset, val_ds: Dataset, cfg
) -> tuple[DataLoader, DataLoader]:
    """Create loaders for either in-memory or memory-mapped datasets."""
    # --- DataLoaders ---
    bs = int(cfg.batch_size)
    nw = int(getattr(cfg, "num_workers", 0))
    seed = int(getattr(cfg, "seed", 42))

    if isinstance(train_ds, IterableDataset):
        # Streaming datasets are iterated on the main thread only (the
        # dataloader entry point refuses num_workers > 0 for them), so
        # persistent_workers is meaningless here.
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            drop_last=bool(getattr(cfg, "drop_last", True)),
            num_workers=0,
            pin_memory=bool(getattr(cfg, "pin_memory", True)),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            drop_last=False,
            num_workers=0,
            pin_memory=bool(getattr(cfg, "pin_memory", True)),
        )
        return train_loader, val_loader

    train_sampler = EpochRandomSampler(
        train_ds,
        seed=seed,
        shuffle=bool(getattr(cfg, "shuffle", True)),
    )
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        generator=train_generator,
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
        generator=val_generator,
        drop_last=False,
        num_workers=nw,
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg, "persistent_workers", True))
        if nw > 0
        else False,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "The training DataLoader has no batches; reduce batch_size or "
            "set drop_last=false."
        )

    # Dedicated eval loader over the same dataset and a same-seed sampler.
    # Trainer.evaluate() must never call iter() on the train loader itself:
    # with persistent_workers and num_workers > 0, PyTorch reuses ONE
    # iterator object across iter() calls and _reset() rewinds the sampler
    # stream to batch 0 of the epoch permutation, so an eval would send the
    # training stream back to the top of the pass every eval_interval steps
    # (loss-curve sawtooth with period == eval_interval). Same seed+epoch
    # means the same permutation, i.e. the first N batches of the pass.
    eval_sampler = EpochRandomSampler(
        train_ds,
        seed=seed,
        shuffle=bool(getattr(cfg, "shuffle", True)),
    )
    train_loader.eval_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=eval_sampler,
        drop_last=bool(getattr(cfg, "drop_last", True)),
        num_workers=0,
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
    )

    return train_loader, val_loader
