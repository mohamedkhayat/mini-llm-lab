import json

import hydra
import omegaconf

from data.budget import minimum_cache_tokens, parse_token_budget, split_cache_tokens
from data.prepare import get_data_dir, prepare_dataset
from training.checkpointing import load_checkpoint, resolve_resume_path
from training.schedule import resolve_phase_budget
from training.trainer import Trainer


def validate_resume_data_identity(cfg, checkpoint):
    """Reject a checkpoint whose source identity changed before data restore."""
    if checkpoint is None:
        return
    saved_cfg = checkpoint.get("cfg", {})
    saved_data = saved_cfg.get("data", {}) if isinstance(saved_cfg, dict) else {}
    fields = (
        "hf_dataset",
        "hf_config",
        "revision",
        "text_column",
        "file_format",
        "tokenizer_name",
    )
    mismatches = []
    for field in fields:
        if field not in saved_data or not hasattr(cfg.data, field):
            continue
        if saved_data[field] != getattr(cfg.data, field):
            mismatches.append(
                f"{field}: checkpoint={saved_data[field]!r}, "
                f"config={getattr(cfg.data, field)!r}"
            )
    if mismatches:
        raise ValueError(
            "Cannot resume safely: the checkpoint data identity changed; "
            "restarting from a different token stream would invalidate the "
            "saved cursor. "
            + "; ".join(mismatches)
        )


def is_dataset_valid(data_dir, cfg):

    train_bin, val_bin, meta_json = (
        data_dir / "train.bin",
        data_dir / "eval.bin",
        data_dir / "meta.json",
    )

    if not (train_bin.is_file() and val_bin.is_file() and meta_json.is_file()):
        return False

    try:
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta["tokenizer_name"] != cfg.data.tokenizer_name:
            print("tokenizer changed in config. Re tokenizing...")
            return False

        needed_total = parse_token_budget(cfg.data.max_tokens, "data.max_tokens", allow_none=False)
        val_ratio = float(cfg.data.val_ratio)
        needed_train, needed_val = split_cache_tokens(needed_total, val_ratio)

        cached_train = meta.get("train_tokens", 0)
        cached_val = meta.get("val_tokens", 0)
        cached_total = meta.get("max_tokens", cached_train + cached_val)
        dtype_sizes = {"uint16": 2, "uint32": 4}
        dtype_size = dtype_sizes.get(meta.get("dtype"))
        identity_fields = ("hf_dataset", "hf_config", "text_column", "file_format")
        has_identity_metadata = any(field in meta for field in identity_fields)
        # Caches from the immediately preceding memmap preparer predate the
        # identity fields. Their dataset/tokenizer identity is still encoded
        # by the cache path, so keep them reusable; once any identity field is
        # present, require the complete new contract.
        identity_matches = not has_identity_metadata or all(
            meta.get(field) == getattr(cfg.data, field, None)
            for field in identity_fields
        )

        expected_train, expected_val = split_cache_tokens(
            int(cached_total), meta.get("val_ratio", val_ratio)
        )
        ratio_matches = abs(float(meta.get("val_ratio", val_ratio)) - val_ratio) < 1e-12
        metadata_matches = (
            int(cached_total) == int(cached_train) + int(cached_val)
            and int(cached_train) == expected_train
            and int(cached_val) == expected_val
            and ratio_matches
            and identity_matches
            and dtype_size is not None
            and train_bin.stat().st_size == int(cached_train) * dtype_size
            and val_bin.stat().st_size == int(cached_val) * dtype_size
        )

        if not metadata_matches or cached_train < needed_train or cached_val < needed_val:
            print(
                f"Cache metadata/capacity is incompatible: has "
                f"{cached_train:,} train / {cached_val:,} val tokens, "
                f"but run requires {needed_train:,} / {needed_val:,}. "
                "Re-tokenizing..."
            )
            return False

        print(
            f"Reusing cache! Run needs {needed_train:,} train tokens; "
            f"cache has {cached_train:,}."
        )
        return True

    except Exception as e:
        print(f"Error reading {meta_json} : {e}. Re tokenizing...")
        return False


def tokens_per_step(cfg):
    """Return the effective train tokens consumed by one optimizer update."""
    return (
        int(cfg.training.batch_size)
        * int(cfg.data.seq_len)
        * int(getattr(cfg.training, "accum_steps", 1))
    )


def prepare_cache_for_run(cfg, checkpoint=None):
    """Validate or expand the cache for the selected phase endpoint."""
    validate_resume_data_identity(cfg, checkpoint)
    requested_total = parse_token_budget(
        cfg.data.max_tokens, "data.max_tokens", allow_none=False
    )
    requested_train, _ = split_cache_tokens(requested_total, float(cfg.data.val_ratio))
    step_tokens = tokens_per_step(cfg)
    phase_budget = resolve_phase_budget(cfg.training, checkpoint, step_tokens)
    required_train = (
        phase_budget.target_step * step_tokens + 1
        if phase_budget.target_step is not None
        else None
    )
    effective_total = requested_total
    can_expand = phase_budget.mode in {
        "stable_continue",
        "decay_tokens",
        "legacy_decay",
    }
    if required_train is not None and required_train > requested_train:
        required_total = minimum_cache_tokens(
            required_train, float(cfg.data.val_ratio)
        )
        if not can_expand:
            raise ValueError(
                f"{phase_budget.mode} requires {required_train:,} train tokens, "
                f"but data.max_tokens={requested_total:,} only provides its "
                f"configured train prefix. Increase data.max_tokens to at least "
                f"{required_total:,}; only additive continuation/decay phases "
                "expand the cache automatically."
            )
        effective_total = max(effective_total, required_total)

    if effective_total != requested_total:
        print(
            f"Phase endpoint needs {required_train:,} train tokens; "
            f"expanding preprocessing budget from {requested_total:,} to "
            f"{effective_total:,} total tokens to preserve val_ratio={cfg.data.val_ratio}."
        )
        cfg.data.max_tokens = effective_total

    data_dir = get_data_dir(cfg)
    if not is_dataset_valid(data_dir, cfg):
        prepare_dataset(cfg, preserve_prefix=checkpoint is not None)
    return phase_budget


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    resume_path = resolve_resume_path(cfg.training)
    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    prepare_cache_for_run(cfg, checkpoint)

    print(omegaconf.OmegaConf.to_yaml(cfg))
    print("-" * 60)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
