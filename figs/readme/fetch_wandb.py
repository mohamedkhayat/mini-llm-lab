"""Snapshot W&B run hgi0v79p into figs/data/ so the README figures rebuild offline.

Writes:
  wandb_history_hgi0v79p.json  every logged train/val/grad row
  run_summary.json             headline numbers + model config straight from W&B
  run_manifest.json            the run's local checkpoint manifest (if present)

Run: WANDB_API_KEY=... .venv/bin/python figs/readme/fetch_wandb.py
"""

import json
import shutil
import statistics
from pathlib import Path

import wandb

RUN = "mohamedkhayat025-none/mini-llm-lab/hgi0v79p"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figs" / "data"
MANIFEST = ROOT / "runs" / "2026-10-13" / "19-16-08" / "run_manifest.json"

KEYS = [
    "train/loss",
    "train/lr",
    "train/tok_per_sec",
    "train/grad_norm",
    "train/grad_norm_max",
    "train/tokens_seen",
    "val/loss",
    "val/stderr",
    *[f"grad/layer/{i:02d}" for i in range(12)],
    *[
        f"grad/group/{g}"
        for g in ("attn_q", "attn_k", "attn_v", "attn_out", "ffn", "norm", "emb_head")
    ],
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run = wandb.Api(timeout=120).run(RUN)
    rows = []
    for row in run.scan_history(page_size=2000):
        d = {k: row[k] for k in KEYS + ["step"] if row.get(k) is not None}
        if len(d) > 1:
            rows.append(d)
    (OUT / "wandb_history_hgi0v79p.json").write_text(json.dumps(rows))

    s = run.summary
    tps = [r["train/tok_per_sec"] for r in rows if "train/tok_per_sec" in r]
    summary = {
        "run": RUN,
        "url": run.url,
        "gpu": (run.metadata or {}).get("gpu"),
        "parameters": s["model/parameters_total"],
        "steps": s["step"],
        "tokens_seen": s["train/tokens_seen"],
        "elapsed_hours": s["run/elapsed_hours"],
        "final_loss": s["final/loss"],
        "final_ppl": s["final/ppl"],
        "final_stderr": s["final/stderr"],
        "final_eval_tokens": s["final/tokens"],
        "median_tok_per_sec": statistics.median(tps),
        "peak_gpu_gb": s["system/gpu_memory_peak_allocated_gb"],
        "tokens_per_step": run.config["computed_tokens_per_step"],
        "model_cfg": run.config["model"],
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=1))
    if MANIFEST.exists():
        shutil.copy(MANIFEST, OUT / "run_manifest.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "model_cfg"}, indent=1))


if __name__ == "__main__":
    main()
