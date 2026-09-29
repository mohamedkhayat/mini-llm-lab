import asyncio
from contextlib import nullcontext
import gc
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Slider
from omegaconf import DictConfig, OmegaConf
import torch

from data.tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text
from models.gpt import GptModel


PROJECT_ROOT = Path(__file__).resolve().parent
_configured_runs_dir = Path(os.environ.get("MINI_LLM_RUNS_DIR", "runs"))
RUNS_DIR = (
    _configured_runs_dir
    if _configured_runs_dir.is_absolute()
    else PROJECT_ROOT / _configured_runs_dir
).resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    modified_at: float
    cfg: DictConfig | None


@dataclass
class ModelBundle:
    path: Path
    tokenizer: Any
    model: GptModel
    cfg: DictConfig
    autocast_context: Any
    summary: str


def _cfg_value(cfg: DictConfig | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    return OmegaConf.select(cfg, key, default=default)


def _load_run_config(checkpoint_path: Path) -> DictConfig | None:
    """Read Hydra metadata without opening the large checkpoint tensor file."""
    candidates = (
        checkpoint_path.parent / ".hydra" / "config.yaml",
        checkpoint_path.parent / "config.yaml",
    )
    for config_path in candidates:
        if config_path.exists():
            return OmegaConf.load(config_path)
    return None


def discover_checkpoints(runs_dir: Path = RUNS_DIR) -> list[CheckpointInfo]:
    """Find checkpoints and sort newest first using filesystem timestamps."""
    if not runs_dir.exists():
        return []

    checkpoints = []
    for path in runs_dir.rglob("*.pt"):
        if not path.is_file():
            continue
        checkpoints.append(
            CheckpointInfo(
                path=path.resolve(),
                modified_at=path.stat().st_mtime,
                cfg=_load_run_config(path),
            )
        )

    return sorted(
        checkpoints,
        key=lambda item: (item.modified_at, str(item.path)),
        reverse=True,
    )


MODEL_CATALOG = discover_checkpoints()
MODEL_BY_PATH = {str(item.path): item for item in MODEL_CATALOG}


def _format_count(value: int | None) -> str:
    if value is None:
        return "unknown"
    value = int(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,}"


def _planned_training_description(cfg: DictConfig | None) -> str:
    data_max_tokens = _cfg_value(cfg, "data.max_tokens")
    if data_max_tokens is not None:
        return f"≤ {_format_count(int(data_max_tokens))} tokens (one pass over the data)"
    return "budget unknown"


def _checkpoint_label(info: CheckpointInfo) -> str:
    model_name = _cfg_value(info.cfg, "model.name", "gpt2")
    emb_dim = _cfg_value(info.cfg, "model.emb_dim", "?")
    n_layers = _cfg_value(info.cfg, "model.n_layers", "?")
    context_length = _cfg_value(info.cfg, "model.context_length", "?")
    timestamp = datetime.fromtimestamp(info.modified_at).strftime("%Y-%m-%d %H:%M")
    run_name = info.path.parent.relative_to(RUNS_DIR)
    return (
        f"{timestamp} | {run_name} | {info.path.name} | "
        f"{model_name} {emb_dim}d/{n_layers}L | ctx {context_length} | "
        f"{_planned_training_description(info.cfg)}"
    )


def _checkpoint_options() -> dict[str, str]:
    options = {}
    for info in MODEL_CATALOG:
        label = _checkpoint_label(info)
        if label in options:
            label = f"{label} | {info.path}"
        options[label] = str(info.path)
    return options


def _load_checkpoint_config(
    checkpoint_path: Path, checkpoint: dict[str, Any]
) -> DictConfig:
    cfg = checkpoint.get("cfg")
    if cfg is not None:
        return OmegaConf.create(cfg)

    sidecar_cfg = _load_run_config(checkpoint_path)
    if sidecar_cfg is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no cfg and no Hydra config sidecar."
        )
    return sidecar_cfg


def _load_state_dict(model: GptModel, state_dict: dict[str, Any]) -> None:
    """Load regular and compiled checkpoints."""
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as first_error:
        compiled_prefix = "_orig_mod."
        if not any(key.startswith(compiled_prefix) for key in state_dict):
            raise first_error

    unwrapped_state_dict = {
        key.removeprefix(compiled_prefix): value for key, value in state_dict.items()
    }
    model.load_state_dict(unwrapped_state_dict, strict=True)


def _configure_inference_runtime(cfg: DictConfig) -> tuple[Any, str]:
    """Apply the checkpoint's precision and Tensor Core settings for inference."""
    requested_bf16 = bool(_cfg_value(cfg, "training.use_bf16", False))
    requested_tensor_cores = bool(_cfg_value(cfg, "training.use_tensor_cores", False))

    if DEVICE.type != "cuda":
        return (
            nullcontext(),
            "BF16 autocast: disabled (CUDA unavailable)\n"
            "**Tensor Cores:** disabled (CUDA unavailable)",
        )

    capability = torch.cuda.get_device_capability()
    tensor_cores_enabled = requested_tensor_cores and capability[0] >= 8
    if tensor_cores_enabled:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False

    if requested_bf16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "This checkpoint requests BF16 inference, but the current "
                "CUDA device does not support BF16."
            )
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    return (
        autocast_context,
        f"BF16 autocast: {'enabled' if requested_bf16 else 'disabled'}\n"
        f"**Tensor Cores:** "
        f"{'enabled' if tensor_cores_enabled else 'disabled'} "
        f"(compute capability {capability[0]}.{capability[1]})",
    )


def _model_summary(
    checkpoint_path: Path,
    model: GptModel,
    cfg: DictConfig,
    checkpoint: dict[str, Any],
    runtime_summary: str,
) -> str:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    step = checkpoint.get("step")
    batch_size = _cfg_value(cfg, "training.batch_size")
    seq_len = _cfg_value(cfg, "data.seq_len")
    accum_steps = int(_cfg_value(cfg, "training.accum_steps") or 1)
    if step is not None and batch_size is not None and seq_len is not None:
        trained_tokens = int(step) * int(batch_size) * int(seq_len) * accum_steps
        token_text = _format_count(trained_tokens)
    else:
        token_text = _planned_training_description(cfg)

    return (
        f"**Model:** `{checkpoint_path.name}`\n"
        f"**Run:** `{checkpoint_path.parent.relative_to(RUNS_DIR)}`\n"
        f"**Parameters:** {_format_count(parameter_count)}\n"
        f"**Attention:** `{_cfg_value(cfg, 'model.attention', 'mha')}`\n"
        f"**Tied embeddings:** "
        f"{'enabled' if bool(_cfg_value(cfg, 'model.tie_embeddings', False)) else 'disabled'}\n"
        f"**Context:** {_cfg_value(cfg, 'model.context_length', 'unknown')} tokens\n"
        f"**Tokenizer:** `{_cfg_value(cfg, 'data.tokenizer_name', 'gpt2')}`\n"
        f"**Training tokens:** {token_text}\n"
        f"**Device:** `{DEVICE}`\n"
        f"{runtime_summary}"
    )


def get_model_and_tokenizer(model_path: str | Path) -> ModelBundle:
    """Load a selected checkpoint and its tokenizer from saved config metadata."""
    checkpoint_path = Path(model_path).resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    cfg = _load_checkpoint_config(checkpoint_path, checkpoint)
    tokenizer_name = str(_cfg_value(cfg, "data.tokenizer_name", "gpt2"))
    tokenizer = get_tokenizer(tokenizer_name)
    autocast_context, runtime_summary = _configure_inference_runtime(cfg)

    model = GptModel(cfg.model)
    _load_state_dict(model, checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    summary = _model_summary(checkpoint_path, model, cfg, checkpoint, runtime_summary)
    del checkpoint
    return ModelBundle(
        path=checkpoint_path,
        tokenizer=tokenizer,
        model=model,
        cfg=cfg,
        autocast_context=autocast_context,
        summary=summary,
    )


def _generation_settings(settings: dict[str, Any]) -> dict[str, Any]:
    temperature = max(0.0, min(2.0, float(settings.get("temperature", 0.6))))
    max_new_tokens = max(1, min(512, int(settings.get("max_new_tokens", 50))))
    top_k_value = int(settings.get("top_k", 25))
    top_k = None if top_k_value <= 0 else top_k_value
    return {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
    }


def _chat_settings() -> cl.ChatSettings:
    options = _checkpoint_options()
    if not options:
        raise FileNotFoundError(
            f"No .pt checkpoints found under {RUNS_DIR}. Run training first."
        )

    default_path = next(iter(options.values()))
    return cl.ChatSettings(
        inputs=[
            Select(
                id="model",
                label="Model checkpoint",
                items=options,
                initial_value=default_path,
                tooltip="Newest checkpoints are listed first.",
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.6,
                min=0.0,
                max=2.0,
                step=0.05,
                description="0 is greedy/deterministic generation.",
            ),
            Slider(
                id="max_new_tokens",
                label="Max new tokens",
                initial=50,
                min=1,
                max=512,
                step=1,
            ),
            Slider(
                id="top_k",
                label="Top-k",
                initial=25,
                min=0,
                max=1000,
                step=1,
                description="0 disables top-k filtering.",
            ),
        ]
    )


def _clear_session_model() -> None:
    bundle = cl.user_session.get("model_bundle")
    cl.user_session.set("model_bundle", None)
    if bundle is not None:
        del bundle
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def _selected_checkpoint(value: Any) -> Path:
    path = Path(str(value)).resolve()
    if str(path) not in MODEL_BY_PATH:
        raise ValueError("That checkpoint is not available in the model selector.")
    return path


async def _load_for_session(model_path: str | Path) -> ModelBundle:
    path = _selected_checkpoint(model_path)
    await cl.Message(content=f"Loading `{path.name}`...").send()
    _clear_session_model()
    bundle = await asyncio.to_thread(get_model_and_tokenizer, path)
    cl.user_session.set("model_bundle", bundle)
    cl.user_session.set("model_path", str(path))
    await cl.Message(content=f"Loaded model.\n\n{bundle.summary}").send()
    return bundle


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        settings = await _chat_settings().send()
    except FileNotFoundError as error:
        await cl.Message(content=str(error)).send()
        return

    cl.user_session.set("generation_settings", _generation_settings(settings))
    await _load_for_session(settings["model"])


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    cl.user_session.set("generation_settings", _generation_settings(settings))
    selected_path = _selected_checkpoint(settings["model"])
    current_path = cl.user_session.get("model_path")
    if str(selected_path) != current_path:
        await _load_for_session(selected_path)


@cl.on_chat_end
def on_chat_end() -> None:
    _clear_session_model()


@cl.on_message
async def main(message: cl.Message) -> None:
    bundle = cl.user_session.get("model_bundle")
    if bundle is None:
        await cl.Message(
            content="No model is loaded. Select a checkpoint first."
        ).send()
        return

    generation = cl.user_session.get("generation_settings") or {
        "temperature": 0.6,
        "max_new_tokens": 50,
        "top_k": 25,
    }
    context_length = int(_cfg_value(bundle.cfg, "model.context_length", 1024))
    token_ids = bundle.model.generate(
        token_idx=text_to_token_ids(message.content, bundle.tokenizer, DEVICE),
        max_new_tokens=generation["max_new_tokens"],
        context_length=context_length,
        autocast_context=bundle.autocast_context,
        top_k=generation["top_k"],
        temperature=generation["temperature"],
    )
    text = token_ids_to_text(token_ids, bundle.tokenizer)
    await cl.Message(content=text).send()
