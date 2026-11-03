"""Record token-by-token generations from real checkpoints for the README GIFs.

For every sampled token this stores the decoded piece, its probability, and the
top-5 candidates the model was choosing between, so the animation shows what
the model actually computed, not a re-typed string.

  probe : final checkpoint, several prompts/seeds -> pick one by eye
  final : final checkpoint, one prompt, with top-5 traces  -> stream.json
  race  : same prompt + seed across 4 checkpoints          -> race.json

Run (GPU if available):
  .venv/bin/python figs/readme/record_generations.py probe
  .venv/bin/python figs/readme/record_generations.py final "<prompt>" <seed>
  .venv/bin/python figs/readme/record_generations.py race  "<prompt>" <seed>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.tokenizer import get_tokenizer  # noqa: E402
from models.gpt import GptModel  # noqa: E402

RUN_DIR = ROOT / "runs" / "2026-10-13" / "19-16-08"
CACHE = Path(os.environ.get("CKPT_CACHE", "/tmp/mini-llm-lab-ckpt"))
OUT = ROOT / "figs" / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW = 90
RACE_CHECKPOINTS = ["step_2000", "step_6000", "step_14000", "final_model"]


def load(name):
    path = RUN_DIR / f"{name}.pt"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{name}.pt"
        if not path.exists():
            src = RUN_DIR / f"{name}.pt.zst"
            subprocess.run(
                ["zstd", "-d", "-q", "-f", str(src), "-o", str(path)], check=True
            )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    cfg = OmegaConf.create(ck["cfg"])
    model = GptModel(cfg.model)
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    meta = {"step": int(ck["step"]), "tokens_seen": int(ck["tokens_seen"])}
    return model.to(DEVICE).eval(), meta


@torch.no_grad()
def sample(model, tok, prompt, seed, trace=True):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    steps, pending = [], b""
    for _ in range(MAX_NEW):
        logits = model(ids[:, -model.context_length :])[:, -1, :].float()
        logits = logits / TEMPERATURE
        kth = torch.topk(logits, TOP_K).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen)
        # A token can be half of a UTF-8 character: hold bytes until they decode.
        pending += tok.decode_single_token_bytes(nxt.item())
        try:
            piece, pending = pending.decode("utf-8"), b""
        except UnicodeDecodeError:
            piece = ""
        step = {"text": piece, "p": round(probs[0, nxt].item(), 4)}
        if trace:
            top = torch.topk(probs[0], 5)
            step["top"] = [
                [
                    tok.decode_single_token_bytes(i).decode("utf-8", "replace"),
                    round(p, 4),
                ]
                for p, i in zip(top.values.tolist(), top.indices.tolist())
            ]
        steps.append(step)
        ids = torch.cat([ids, nxt], dim=1)
    return steps


def main():
    mode = sys.argv[1]
    tok = get_tokenizer("gpt2")
    OUT.mkdir(parents=True, exist_ok=True)
    settings = {"temperature": TEMPERATURE, "top_k": TOP_K, "max_new_tokens": MAX_NEW}

    if mode == "probe":
        model, _ = load("final_model")
        prompts = [
            "The mitochondria is",
            "Photosynthesis is the process by which",
            "The French Revolution began in",
            "To understand how a computer works,",
            "Black holes form when",
            "Water boils at a lower temperature at high altitude because",
        ]
        for prompt in prompts:
            for seed in (1, 7, 42):
                text = "".join(
                    s["text"] for s in sample(model, tok, prompt, seed, False)
                )
                print(f"--- [{seed}] {prompt}{text!r}\n")
        return

    prompt, seed = sys.argv[2], int(sys.argv[3])
    if mode == "final":
        model, meta = load("final_model")
        data = {
            **settings,
            "prompt": prompt,
            "seed": seed,
            **meta,
            "steps": sample(model, tok, prompt, seed),
        }
        (OUT / "stream.json").write_text(json.dumps(data, indent=1))
    elif mode == "race":
        runs = []
        for name in RACE_CHECKPOINTS:
            model, meta = load(name)
            runs.append(
                {
                    "checkpoint": name,
                    **meta,
                    "steps": sample(model, tok, prompt, seed, False),
                }
            )
            del model
            torch.cuda.empty_cache()
        data = {**settings, "prompt": prompt, "seed": seed, "runs": runs}
        (OUT / "race.json").write_text(json.dumps(data, indent=1))
    print("ok")


if __name__ == "__main__":
    main()
