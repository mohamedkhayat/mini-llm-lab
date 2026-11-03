"""Build every README visual from recorded data. No numbers are typed by hand.

Inputs (all committed under figs/data/):
  wandb_history_hgi0v79p.json  full W&B history of run hgi0v79p (fetch_wandb.py)
  stream.json / race.json      token-by-token generations (record_generations.py)

Outputs (figs/readme/):
  hero-{light,dark}.svg          banner with headline numbers + real val-loss curve
  training-{light,dark}.svg      loss / LR / throughput / grad-norm small multiples
  grad-layers-{light,dark}.svg   per-layer gradient-norm heatmap
  architecture-{light,dark}.svg  block diagram + parameter budget (counted from GptModel)
  stream.gif                     final model streaming tokens with its top-5 choices
  race.gif                       four checkpoints streaming the same prompt

Run:  .venv/bin/python figs/readme/build_assets.py
Fonts (Inter, JetBrains Mono) are downloaded once into figs/readme/.fonts/.
"""

import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "figs" / "data"
FONTS = HERE / ".fonts"
PREVIEW = Path(os.environ["PREVIEW_DIR"]) if os.environ.get("PREVIEW_DIR") else None
sys.path.insert(0, str(ROOT / "src"))

# Google Fonts static instances (stable gstatic URLs).
FONT_URLS = {
    "Inter-400.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf",
    "Inter-500.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuI6fMZg.ttf",
    "Inter-600.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuGKYMZg.ttf",
    "Inter-800.ttf": "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuDyYMZg.ttf",
    "JBMono-400.ttf": "https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxjPQ.ttf",
    "JBMono-500.ttf": "https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8-qxjPQ.ttf",
    "JBMono-700.ttf": "https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8L6tjPQ.ttf",
}

# Theme tokens. Series hues validated with the dataviz palette validator
# (light: #2a78d6/#eb6834 on #ffffff, dark: #3987e5/#e8692f on #0d1117).
THEMES = {
    "light": {
        "bg": "#ffffff",
        "panel": "#f6f8fa",
        "ink": "#1f2328",
        "muted": "#59636e",
        "faint": "#8c959f",
        "grid": "#d8dee4",
        "band": "#f3f5f7",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "blue_soft": "#9ec3ee",
        "ramp": ["#f2f7fd", "#b9d4f3", "#6ea3e4", "#2a78d6", "#174b8f", "#0b2650"],
    },
    "dark": {
        "bg": "#0d1117",
        "panel": "#151b23",
        "ink": "#f0f6fc",
        "muted": "#9198a1",
        "faint": "#656c76",
        "grid": "#262c36",
        "band": "#141a22",
        "blue": "#3987e5",
        "orange": "#e8692f",
        "blue_soft": "#24508a",
        "ramp": ["#111824", "#16325a", "#1f5498", "#3987e5", "#8cbcf3", "#e3effd"],
    },
}


# --------------------------------------------------------------------- setup
def ensure_fonts():
    FONTS.mkdir(exist_ok=True)
    for name, url in FONT_URLS.items():
        path = FONTS / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "svg.fonttype": "path",  # glyphs as paths: renders identically on GitHub
            "font.family": "Inter",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "svg.hashsalt": "mini-llm-lab",  # deterministic SVG ids
        }
    )


def mono(weight=400):
    return {"family": "JetBrains Mono", "weight": weight}


def load_history():
    rows = json.loads((DATA / "wandb_history_hgi0v79p.json").read_text())
    rows.sort(key=lambda r: r["step"])
    return rows


def series(rows, key):
    pts = [
        (
            r["step"],
            r["train/tokens_seen"] if "train/tokens_seen" in r else None,
            r[key],
        )
        for r in rows
        if key in r
    ]
    return pts


def tokens_at(rows, step):
    """Tokens seen at an optimizer step (constant tokens/step for the run)."""
    per = rows[0]["train/tokens_seen"] / rows[0]["step"]
    return step * per


def phases(rows):
    """Warmup end and decay start, read off the logged LR curve."""
    lr = [(r["step"], r["train/lr"]) for r in rows if "train/lr" in r]
    peak = max(v for _, v in lr)
    warm_end = next(s for s, v in lr if v >= peak * 0.999)
    decay_start = next(s for s, v in lr if s > warm_end and v < peak * 0.999)
    # the logged point is the first *after* decay began; step back one interval
    return warm_end, decay_start - (lr[1][0] - lr[0][0]), lr[-1][0]


def final_eval():
    manifest = json.loads((ROOT / "figs" / "data" / "run_manifest.json").read_text())
    return manifest["checkpoints"]["final_model.pt"]


def fmt_b(x):
    return f"{x / 1e9:.0f}B" if x else "0"


def save(fig, name):
    if PREVIEW:  # also write a PNG to look at (not committed)
        fig.savefig(
            PREVIEW / name.replace(".svg", ".png"),
            facecolor=fig.get_facecolor(),
            dpi=100,
        )
    fig.savefig(
        HERE / name,
        format="svg",
        facecolor=fig.get_facecolor(),
        metadata={"Date": None},
        dpi=200,
    )
    plt.close(fig)
    print("wrote", name)


def style_axes(ax, t):
    ax.set_facecolor(t["bg"])
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(font_manager.FontProperties(**mono()))
        lbl.set_fontsize(8.5)
    ax.grid(axis="y", color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------- hero
def hero(rows, theme):
    t = THEMES[theme]
    fev = final_eval()
    params = json.loads((DATA / "run_summary.json").read_text())
    fig = plt.figure(figsize=(12.8, 4.2), dpi=100)
    fig.patch.set_facecolor(t["bg"])

    fig.text(
        0.045,
        0.82,
        "LLM PRETRAINING FRAMEWORK  ·  PYTORCH",
        color=t["orange"],
        fontsize=10.5,
        **mono(700),
    )
    fig.text(0.043, 0.60, "mini-llm-lab", color=t["ink"], fontsize=50, weight=800)
    fig.text(
        0.045,
        0.495,
        "Modular transformer components, swappable from config,",
        color=t["muted"],
        fontsize=14.5,
        weight=400,
    )
    fig.text(
        0.045,
        0.43,
        "with optional fused kernels and a resumable training stack.",
        color=t["muted"],
        fontsize=14.5,
        weight=400,
    )

    stats = [
        (f"{params['parameters'] / 1e6:.1f}M", "parameters"),
        (f"{fev['tokens_seen'] / 1e9:.2f}B", "tokens"),
        (f"{fev['final_eval']['loss']:.2f}", "val loss"),
        (f"{params['elapsed_hours']:.1f}h", "wall clock"),
    ]
    fig.text(
        0.045,
        0.285,
        "SHOWCASE RUN  ·  A SMALL LM TRAINED WITH IT ON 1× RTX 5090",
        color=t["faint"],
        fontsize=8.5,
        **mono(700),
    )
    fig.add_artist(
        plt.Line2D([0.045, 0.52], [0.265, 0.265], color=t["grid"], linewidth=0.8)
    )
    x = 0.045
    for value, label in stats:
        fig.text(x, 0.14, value, color=t["ink"], fontsize=24, weight=600)
        fig.text(x, 0.07, label.upper(), color=t["faint"], fontsize=8.5, **mono(500))
        x += 0.118

    # Right: the real validation curve, as the banner's picture.
    ax = fig.add_axes([0.58, 0.14, 0.38, 0.70])
    ax.set_facecolor(t["bg"])
    tr = series(rows, "train/loss")
    va = series(rows, "val/loss")
    tx = np.array([tokens_at(rows, s) for s, _, _ in tr]) / 1e9
    ty = np.array([v for _, _, v in tr])
    vx = np.array([tokens_at(rows, s) for s, _, _ in va]) / 1e9
    vy = np.array([v for _, _, v in va])
    ax.plot(tx, ty, color=t["blue"], linewidth=1.0, alpha=0.55)
    ax.plot(vx, vy, color=t["orange"], linewidth=2.2, solid_capstyle="round")
    ax.scatter(
        [vx[-1]],
        [vy[-1]],
        s=46,
        color=t["orange"],
        edgecolor=t["bg"],
        linewidth=2,
        zorder=5,
    )
    ax.set_yscale("log")
    ax.set_ylim(2.55, 10.5)
    ax.set_xlim(0, 12.2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_xticklabels(["0", "4B", "8B", "12B tokens"])
    ax.set_yticks([3, 4, 6, 10])
    ax.set_yticklabels(["3", "4", "6", "10"])
    ax.minorticks_off()
    style_axes(ax, t)
    ax.text(
        0.98,
        0.96,
        "cross-entropy loss",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=t["faint"],
        fontsize=8.5,
        **mono(500),
    )
    ax.annotate(
        "val",
        (vx[3], vy[3]),
        xytext=(vx[3] + 0.7, vy[3] * 1.18),
        color=t["orange"],
        fontsize=9,
        **mono(700),
    )
    ax.annotate(
        "train",
        (tx[40], ty[40]),
        xytext=(tx[40] - 0.15, ty[40] * 0.84),
        color=t["blue"],
        fontsize=9,
        **mono(700),
    )
    save(fig, f"hero-{theme}.svg")


# ------------------------------------------------------------------ training
def training(rows, theme):
    t = THEMES[theme]
    warm_end, decay_start, last = phases(rows)
    fev = final_eval()
    fig = plt.figure(figsize=(12.8, 7.6), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.75, 1],
        hspace=0.42,
        wspace=0.22,
        left=0.06,
        right=0.975,
        top=0.86,
        bottom=0.08,
    )

    fig.text(
        0.06,
        0.945,
        "One run, 22,814 optimizer steps",
        color=t["ink"],
        fontsize=17,
        weight=600,
    )
    fig.text(
        0.06,
        0.905,
        "116.5M-parameter showcase run · 524,288 tokens per step · BF16 · torch.compile · "
        "Liger fused cross-entropy",
        color=t["muted"],
        fontsize=10.5,
    )

    def bands(ax, label=False):
        xs = [
            0,
            tokens_at(rows, warm_end),
            tokens_at(rows, decay_start),
            tokens_at(rows, last),
        ]
        names = ["WARMUP", "STABLE", "DECAY"]
        for i in range(3):
            if i != 1:
                ax.axvspan(
                    xs[i] / 1e9,
                    xs[i + 1] / 1e9,
                    color=t["band"] if theme == "light" else t["panel"],
                    zorder=0,
                    linewidth=0,
                )
            if label:
                ax.text(
                    (xs[i] + xs[i + 1]) / 2e9,
                    1.0,
                    names[i],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    color=t["faint"],
                    fontsize=8,
                    **mono(700),
                )
        ax.set_xlim(0, xs[-1] / 1e9)

    def xb(ax):
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
        ax.set_xticklabels(["0", "2B", "4B", "6B", "8B", "10B", "12B"])

    # Loss
    ax = fig.add_subplot(gs[0, :])
    tr = series(rows, "train/loss")
    va = series(rows, "val/loss")
    tx = np.array([tokens_at(rows, s) for s, _, _ in tr]) / 1e9
    ty = np.array([v for _, _, v in tr])
    vx = np.array([tokens_at(rows, s) for s, _, _ in va]) / 1e9
    vy = np.array([v for _, _, v in va])
    bands(ax, label=True)
    ax.plot(
        tx, ty, color=t["blue"], linewidth=1.0, alpha=0.75, label="train (50-step mean)"
    )
    ax.plot(
        vx,
        vy,
        color=t["orange"],
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markeredgecolor=t["bg"],
        markeredgewidth=1.2,
        label="validation (50-batch probe)",
    )
    ax.set_yscale("log")
    ax.set_ylim(2.42, 10.6)
    ax.set_yticks([3, 4, 5, 6, 8, 10])
    ax.set_yticklabels(["3", "4", "5", "6", "8", "10"])
    ax.minorticks_off()
    xb(ax)
    style_axes(ax, t)
    ax.set_ylabel(
        "cross-entropy (log scale)", color=t["muted"], fontsize=9, **mono(500)
    )
    leg = ax.legend(
        loc="upper right",
        frameon=False,
        borderaxespad=1.2,
        prop=font_manager.FontProperties(**mono(500), size=9),
        bbox_to_anchor=(1.0, 0.93),
    )
    for txt in leg.get_texts():
        txt.set_color(t["muted"])
    # Headline result, labelled with its own measurement scope.
    loss = fev["final_eval"]["loss"]
    fx = fev["tokens_seen"] / 1e9
    ax.scatter(
        [fx],
        [loss],
        marker="D",
        s=52,
        color=t["ink"],
        edgecolor=t["bg"],
        linewidth=1.5,
        zorder=6,
        label="final eval (full 1.37B-token val split)",
    )
    ax.annotate(
        f"final eval on the full 1.37B-token val split:  {loss:.3f}  ·  ppl {math.exp(loss):.1f}",
        xy=(fx, loss),
        xytext=(fx - 0.1, 2.52),
        ha="right",
        va="center",
        color=t["ink"],
        fontsize=10,
        **mono(700),
        arrowprops={
            "arrowstyle": "-",
            "color": t["faint"],
            "linewidth": 0.8,
            "shrinkB": 4,
        },
    )
    ax.text(
        tokens_at(rows, decay_start) / 1e9 + 0.1,
        0.955,
        "resumed +2B",
        transform=ax.get_xaxis_transform(),
        color=t["faint"],
        fontsize=8,
        **mono(500),
        va="top",
    )

    def small(col, key, title, color, scale=1.0, ylog=False, fmt=None, ylim=None):
        ax = fig.add_subplot(gs[1, col])
        pts = series(rows, key)
        x = np.array([tokens_at(rows, s) for s, _, _ in pts]) / 1e9
        y = np.array([v for _, _, v in pts]) * scale
        bands(ax)
        ax.plot(x, y, color=color, linewidth=1.6 if not ylog else 1.0)
        if ylog:
            ax.set_yscale("log")
            ax.minorticks_off()
        if ylim:
            ax.set_ylim(*ylim)
        xb(ax)
        style_axes(ax, t)
        if fmt:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
            for lbl in ax.get_yticklabels():
                lbl.set_fontproperties(font_manager.FontProperties(**mono()))
                lbl.set_fontsize(8.5)
        ax.set_title(
            title, loc="left", color=t["ink"], fontsize=10.5, weight=600, pad=10
        )
        return ax, y

    ax, _ = small(
        0,
        "train/lr",
        "Learning rate — warmup · stable · decay",
        t["blue"],
        scale=1e4,
        ylim=(0, 13.5),
    )
    ax.set_yticks([0, 4, 8, 12])
    ax.set_yticklabels(["0", "4e-4", "8e-4", "1.2e-3"])
    style_axes(ax, t)
    ax, tps = small(
        1,
        "train/tok_per_sec",
        "Throughput (tokens / s)",
        t["blue"],
        scale=1e-3,
        fmt=lambda v, _: f"{v:.0f}K",
        ylim=(100, 160),
    )
    med = float(np.median(tps))
    ax.axhline(med, color=t["faint"], linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(
        0.2,
        med + 2.5,
        f"median {med:.0f}K",
        color=t["muted"],
        fontsize=8.5,
        **mono(500),
    )
    ax, gn = small(
        2, "train/grad_norm", "Gradient norm (pre-clip, log)", t["blue"], ylog=True
    )
    ax.set_yticks([0.1, 0.3, 1, 3, 10])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
    style_axes(ax, t)
    ax.axhline(1.0, color=t["orange"], linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(
        11.9,
        1.12,
        "clip = 1.0",
        ha="right",
        color=t["muted"],
        fontsize=8.5,
        **mono(500),
    )
    save(fig, f"training-{theme}.svg")


# --------------------------------------------------------------- grad layers
def grad_layers(rows, theme):
    t = THEMES[theme]
    pts = [r for r in rows if "grad/layer/00" in r]
    steps = np.array([r["step"] for r in pts])
    sq = np.array([[r[f"grad/layer/{i:02d}"] ** 2 for r in pts] for i in range(12)])
    share = sq / sq.sum(axis=0, keepdims=True)
    k = 6  # smooth over 6 log windows (300 steps): the worst-step sample is noisy
    kernel = np.ones(k) / k
    share = np.array(
        [
            np.convolve(
                np.pad(row, (k // 2, k - 1 - k // 2), mode="edge"), kernel, "valid"
            )
            for row in share
        ]
    )
    fig = plt.figure(figsize=(12.8, 4.6), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0.075, 0.13, 0.80, 0.58])
    cmap = LinearSegmentedColormap.from_list("ramp", t["ramp"])
    x = np.array([tokens_at(rows, st) for st in steps]) / 1e9
    norm = matplotlib.colors.PowerNorm(
        0.5, vmin=0, vmax=100
    )  # sqrt: keeps the small shares visible
    im = ax.pcolormesh(
        x,
        np.arange(12),
        share * 100,
        cmap=cmap,
        shading="nearest",
        norm=norm,
        rasterized=True,
    )  # 5k cells: embed as an image, not 5k SVG paths
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"layer {i}" for i in range(12)])
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xticklabels(["0", "2B", "4B", "6B", "8B", "10B", "12B tokens"])
    style_axes(ax, t)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(
        0.075,
        0.905,
        "Where the gradient lives, layer by layer",
        color=t["ink"],
        fontsize=15,
        weight=600,
    )
    fig.text(
        0.075,
        0.80,
        "Each layer's share of the squared gradient norm (worst step of each 50-step window, "
        "smoothed).\nLayer 0, right after the embedding, takes the largest share throughout, and nearly all "
        "of it for the first ~4B tokens.",
        color=t["muted"],
        fontsize=10,
        linespacing=1.6,
    )
    cax = fig.add_axes([0.895, 0.13, 0.012, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks([0, 8.3, 25, 50, 100])
    cb.set_ticklabels(["0%", "8% (even)", "25%", "50%", "100%"])
    cax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    for lbl in cax.get_yticklabels():
        lbl.set_fontproperties(font_manager.FontProperties(**mono()))
        lbl.set_fontsize(8.5)
    save(fig, f"grad-layers-{theme}.svg")


# -------------------------------------------------------------- architecture
def param_budget():
    from omegaconf import OmegaConf

    from models.gpt import GptModel

    cfg = OmegaConf.create(
        json.loads((DATA / "run_summary.json").read_text())["model_cfg"]
    )
    model = GptModel(cfg)
    groups = {
        "token embedding (tied LM head)": 0,
        "attention (GQA)": 0,
        "SwiGLU feed-forward": 0,
        "norms": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        if "norm" in name:
            groups["norms"] += n
        elif "emb" in name or "lm_head" in name:
            groups["token embedding (tied LM head)"] += n
        elif ".mha." in name:
            groups["attention (GQA)"] += n
        elif ".ffn." in name:
            groups["SwiGLU feed-forward"] += n
        else:
            raise KeyError(f"unclassified parameter {name}")
    total = sum(p.numel() for p in model.parameters())
    return cfg, groups, total


def architecture(theme):
    t = THEMES[theme]
    cfg, groups, total = param_budget()
    assert sum(groups.values()) == total, (groups, total)
    hid = int(cfg.hidden_dim * 2 / 3) if cfg.equalize_params else cfg.hidden_dim
    fig = plt.figure(figsize=(12.8, 7.4), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 740)
    ax.axis("off")
    attn_fill = "#dbe9fa" if theme == "light" else "#12253f"
    ffn_fill = "#fde6da" if theme == "light" else "#33190c"

    def box(x, y, w, h, lines, fill=None, edge=None, ink=None):
        """lines: [(text, size, weight, mono?, color?)] centred top-to-bottom."""
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=8",
                facecolor=fill or t["panel"],
                edgecolor=edge or t["grid"],
                linewidth=1.1,
            )
        )
        step = 18
        y0 = y + h / 2 + step * (len(lines) - 1) / 2
        for i, (text, size, weight, is_mono, color) in enumerate(lines):
            kw = mono(weight) if is_mono else {"weight": weight}
            ax.text(
                x + w / 2,
                y0 - i * step,
                text,
                ha="center",
                va="center",
                fontsize=size,
                color=color or ink or t["ink"],
                **kw,
            )

    def arrow(x, y0, y1):
        ax.annotate(
            "",
            xy=(x, y1),
            xytext=(x, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": t["faint"],
                "linewidth": 1.1,
                "mutation_scale": 10,
            },
        )

    def residual(y_from, y_to, x_side):
        ax.plot(
            [cx, x_side, x_side, cx],
            [y_from, y_from, y_to, y_to],
            color=t["faint"],
            linewidth=1.0,
            solid_joinstyle="round",
        )
        ax.add_patch(
            plt.Circle(
                (cx, y_to),
                7,
                facecolor=t["bg"],
                edgecolor=t["faint"],
                linewidth=1.1,
                zorder=5,
            )
        )
        ax.text(
            cx,
            y_to - 0.5,
            "+",
            ha="center",
            va="center",
            color=t["ink"],
            fontsize=10,
            zorder=6,
        )

    ax.text(60, 690, "Architecture", color=t["ink"], fontsize=18, weight=600)
    ax.text(
        60,
        662,
        "The exact model behind the numbers above, read from the run config.",
        color=t["muted"],
        fontsize=10.5,
    )

    cx, w = 330, 340
    x0 = cx - w / 2
    ax.text(
        cx,
        22,
        f"token ids  (B, {cfg.context_length})  ·  GPT-2 BPE",
        ha="center",
        color=t["faint"],
        fontsize=8.5,
        **mono(500),
    )
    box(
        x0,
        40,
        w,
        50,
        [
            ("Token embedding", 11.5, 600, False, t["bg"]),
            (
                f"{cfg.vocab_size:,} × {cfg.emb_dim}  ·  tied with LM head",
                8.5,
                400,
                True,
                t["faint"],
            ),
        ],
        fill=t["ink"],
        edge=t["ink"],
    )
    arrow(cx, 90, 118)
    # pre-norm block, repeated
    ax.add_patch(
        FancyBboxPatch(
            (x0 - 40, 118),
            w + 80,
            404,
            boxstyle="round,pad=0,rounding_size=12",
            facecolor="none",
            edgecolor=t["faint"],
            linewidth=1.0,
            linestyle=(0, (4, 3)),
        )
    )
    ax.text(
        x0 + w + 40,
        534,
        f"× {cfg.n_layers} layers",
        ha="right",
        va="center",
        color=t["ink"],
        fontsize=12,
        weight=600,
    )
    ax.text(
        x0 - 28,
        504,
        "PRE-NORM BLOCK",
        va="center",
        color=t["faint"],
        fontsize=8,
        **mono(700),
    )
    box(x0, 146, w, 34, [("RMSNorm", 11, 600, False, None)])
    arrow(cx, 180, 200)
    box(
        x0,
        200,
        w,
        86,
        [
            ("Grouped-query attention", 11.5, 600, False, None),
            (
                f"{cfg.n_heads} query heads · {cfg.n_kv_heads} KV heads · causal",
                8.5,
                400,
                True,
                t["muted"],
            ),
            (
                f"RoPE θ={int(cfg.rope_theta):,} · QK-norm · SDPA",
                8.5,
                700,
                True,
                t["blue"],
            ),
        ],
        fill=attn_fill,
        edge=t["blue"],
    )
    arrow(cx, 286, 306)
    residual(134, 314, x0 - 20)
    arrow(cx, 321, 340)
    box(x0, 340, w, 34, [("RMSNorm", 11, 600, False, None)])
    arrow(cx, 374, 394)
    box(
        x0,
        394,
        w,
        78,
        [
            ("SwiGLU feed-forward", 11.5, 600, False, None),
            (
                f"{cfg.emb_dim} → {hid} → {cfg.emb_dim}  ·  SiLU-gated, 3 matmuls",
                8.5,
                400,
                True,
                t["muted"],
            ),
        ],
        fill=ffn_fill,
        edge=t["orange"],
    )
    arrow(cx, 472, 492)
    residual(328, 500, x0 + w + 20)
    arrow(cx, 507, 548)
    box(x0, 548, w, 34, [("RMSNorm", 11, 600, False, None)])
    arrow(cx, 582, 602)
    box(
        x0,
        602,
        w,
        34,
        [("LM head  →  softmax over 50,257", 11, 600, False, t["bg"])],
        fill=t["ink"],
        edge=t["ink"],
    )

    # right column: design choices
    px, pw = 660, 560
    ax.text(px, 610, "Design choices", color=t["ink"], fontsize=13, weight=600)
    notes = [
        (
            "RoPE + QK-norm",
            "Rotary positions instead of a learned table; normalised",
            "queries/keys keep attention logits bounded at LR 1.2e-3.",
        ),
        (
            "GQA  2 : 1",
            f"{cfg.n_kv_heads} KV heads serve {cfg.n_heads} query heads: half the K/V",
            "projections and KV cache, near-MHA quality.",
        ),
        (
            "SwiGLU  ⅔ width",
            f"Gated FFN with hidden {hid} = ⅔ × {cfg.hidden_dim}: same parameter",
            "count as the ungated 4× MLP.",
        ),
        (
            "Tied embeddings",
            "Input embedding and output projection share one",
            f"{cfg.vocab_size:,} × {cfg.emb_dim} matrix.",
        ),
    ]
    ny = 572
    for head, l1, l2 in notes:
        ax.plot([px, px + pw], [ny + 16, ny + 16], color=t["grid"], linewidth=0.7)
        ax.text(
            px, ny - 6, head, color=t["ink"], fontsize=10.5, weight=600, va="center"
        )
        ax.text(px + 150, ny - 6, l1, color=t["muted"], fontsize=9.5, va="center")
        ax.text(px + 150, ny - 26, l2, color=t["muted"], fontsize=9.5, va="center")
        ny -= 62

    # right column: parameter budget
    py = 190
    ax.text(
        px, py + 102, "Where the parameters go", color=t["ink"], fontsize=13, weight=600
    )
    ax.text(
        px,
        py + 80,
        f"{total:,} total, counted from GptModel",
        color=t["muted"],
        fontsize=9.5,
        **mono(),
    )
    colors = [t["ink"], t["blue"], t["orange"], t["faint"]]
    x = px
    for (_, n), col in zip(groups.items(), colors):
        bw = pw * n / total
        if bw > 2:
            ax.add_patch(
                FancyBboxPatch(
                    (x, py + 30),
                    bw - 3,
                    30,
                    boxstyle="round,pad=0,rounding_size=4",
                    facecolor=col,
                    edgecolor="none",
                )
            )
        x += bw
    for i, ((name, n), col) in enumerate(zip(groups.items(), colors)):
        row_y = py - i * 26
        ax.add_patch(
            FancyBboxPatch(
                (px, row_y - 5),
                10,
                10,
                boxstyle="round,pad=0,rounding_size=2",
                facecolor=col,
                edgecolor="none",
            )
        )
        ax.text(px + 20, row_y, name, color=t["ink"], fontsize=9.5, va="center")
        pct = 100 * n / total
        ax.text(
            px + pw,
            row_y,
            f"{n / 1e6:5.1f}M   {pct:4.1f}%"
            if pct >= 0.1
            else f"{n / 1e3:5.1f}K   <0.1%",
            ha="right",
            color=t["muted"],
            fontsize=9,
            va="center",
            **mono(),
        )
    save(fig, f"architecture-{theme}.svg")


# ---------------------------------------------------------------------- GIFs
G = {
    "bg": (13, 17, 23),
    "chrome": (22, 27, 34),
    "line": (48, 54, 61),
    "ink": (230, 237, 243),
    "muted": (139, 148, 158),
    "faint": (88, 96, 105),
    "blue": (57, 135, 229),
    "orange": (232, 105, 47),
    "hl": (61, 38, 24),
}


def pil_font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def window(w, h, title):
    img = Image.new("RGB", (w, h), G["bg"])
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], 14, fill=G["bg"], outline=G["line"])
    d.rounded_rectangle([0, 0, w - 1, 44], 14, fill=G["chrome"])
    d.rectangle([0, 30, w - 1, 44], fill=G["chrome"])
    d.line([0, 44, w, 44], fill=G["line"])
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([18 + i * 22, 16, 30 + i * 22, 28], fill=c)
    f = pil_font("JBMono-500.ttf", 14)
    d.text((w / 2, 22), title, font=f, fill=G["muted"], anchor="mm")
    return img, d


def layout(chunks, cols):
    """Monospace word-wrap. chunks: [(text, tag)] -> [(row, col, char, tag, chunk_index)].

    Wraps on the flattened text so a word split across tokens ("computer" + ".")
    still moves as one unit."""
    flat = [(ch, tag, ci) for ci, (text, tag) in enumerate(chunks) for ch in text]
    out, row, col, i = [], 0, 0, 0
    while i < len(flat):
        ch, tag, ci = flat[i]
        if ch == "\n":
            row, col, i = row + 1, 0, i + 1
            continue
        if ch != " " and (i == 0 or flat[i - 1][0] in " \n"):
            j = i
            while j < len(flat) and flat[j][0] not in " \n":
                j += 1
            if col + (j - i) > cols and col > 0:
                row, col = row + 1, 0
        if col >= cols:
            row, col = row + 1, 0
        if ch == " " and col == 0 and row > 0:
            i += 1
            continue
        out.append((row, col, ch, tag, ci))
        col, i = col + 1, i + 1
    return out


def save_gif(frames, durations, name):
    # One shared palette (taken from the busiest frame) keeps colors stable.
    sample = [frames[0], frames[len(frames) // 2], frames[-1]]
    sheet = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, f in enumerate(sample):
        sheet.paste(f, (0, k * f.height))
    pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(
        HERE / name,
        save_all=True,
        append_images=q[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(
        "wrote",
        name,
        f"{(HERE / name).stat().st_size / 1e6:.1f} MB",
        len(frames),
        "frames",
    )


def stream_gif():
    data = json.loads((DATA / "stream.json").read_text())
    W, H = 1200, 600
    tf, tfb = pil_font("JBMono-400.ttf", 19), pil_font("JBMono-700.ttf", 19)
    small, smallb = pil_font("JBMono-400.ttf", 14), pil_font("JBMono-700.ttf", 14)
    label = pil_font("JBMono-700.ttf", 12)
    cw = tf.getbbox("M")[2]
    lh = 31
    tx0, ty0, cols = 44, 92, int(700 / cw)
    steps = data["steps"]
    chunks = [(data["prompt"], "prompt")] + [(s["text"], "gen") for s in steps]
    cells = layout(chunks, cols)
    title = f"mini-llm-lab  ·  generate  ·  checkpoint step {data['step']:,}"
    px = 800

    def heat(p, bits_max=6.0):
        """Background tint by surprisal -log2(p): 0 bits = none, >= bits_max = full."""
        a = min(1.0, -math.log2(max(p, 1e-9)) / bits_max) * 0.6
        return tuple(
            int(G["bg"][k] + (G["orange"][k] - G["bg"][k]) * a) for k in range(3)
        )

    def frame(n_tokens, show_top, shade=False):
        img, d = window(W, H, title)
        d.text((tx0, 62), "PROMPT + COMPLETION", font=label, fill=G["faint"])
        last = None
        for row, col, ch, tag, ci in cells:
            if ci > n_tokens:
                break
            x, y = tx0 + col * cw, ty0 + row * lh
            is_new = ci == n_tokens and ci > 0 and not shade
            if shade and tag == "gen" and ch != " ":
                d.rectangle(
                    [x, y - 3, x + cw, y + lh - 7], fill=heat(steps[ci - 1]["p"])
                )
            if is_new:
                d.rectangle([x, y - 3, x + cw, y + lh - 7], fill=G["hl"])
            color = (
                G["muted"] if tag == "prompt" else (G["orange"] if is_new else G["ink"])
            )
            d.text((x, y), ch, font=tfb if tag == "prompt" else tf, fill=color)
            last = (x + cw, y)
        if last and not shade:
            d.rectangle(
                [last[0] + 2, last[1] - 1, last[0] + cw - 2, last[1] + lh - 9],
                fill=G["blue"],
            )
        # side panel
        d.line([px - 24, 60, px - 24, H - 70], fill=G["line"])
        d.text((px, 62), "NEXT-TOKEN DISTRIBUTION", font=label, fill=G["faint"])
        if show_top is not None:
            st = steps[show_top]
            d.text(
                (px, 88),
                f"token {show_top + 1:>2} / {len(steps)}",
                font=small,
                fill=G["muted"],
            )
            for i, (tok, p) in enumerate(st["top"]):
                y = 126 + i * 62
                chosen = tok == st["text"]
                shown = repr(tok)[1:-1].replace(" ", "·") or "∅"
                d.text(
                    (px, y),
                    f"{shown[:18]}",
                    font=smallb if chosen else small,
                    fill=G["ink"] if chosen else G["muted"],
                )
                d.text(
                    (px + 360, y),
                    f"{p * 100:5.1f}%",
                    font=small,
                    fill=G["ink"] if chosen else G["muted"],
                    anchor="ra",
                )
                bw = int(360 * p)
                d.rounded_rectangle([px, y + 24, px + 360, y + 34], 4, fill=G["chrome"])
                if bw > 2:
                    d.rounded_rectangle(
                        [px, y + 24, px + bw, y + 34],
                        4,
                        fill=G["orange"] if chosen else G["blue"],
                    )
                if chosen:
                    d.text((px + 362 + 8, y + 22), "◀", font=small, fill=G["orange"])
            picked = repr(st["text"])[1:-1].replace(" ", "·") or "(partial UTF-8 byte)"
            d.text((px, 126 + 5 * 62 + 4), "sampled", font=label, fill=G["faint"])
            d.text(
                (px, 126 + 5 * 62 + 24), f"{picked[:18]}", font=smallb, fill=G["orange"]
            )
            d.text(
                (px + 360, 126 + 5 * 62 + 24),
                f"p = {st['p'] * 100:.1f}%",
                font=small,
                fill=G["orange"],
                anchor="ra",
            )
        if shade:
            d.text(
                (px, 120),
                "Each generated token is shaded by\nits surprisal, −log2 p: how unlikely\n"
                "the model thought its own pick was.\n\nSampling explores the tail; the\n"
                "text still keeps the thread.",
                font=small,
                fill=G["muted"],
                spacing=10,
            )
            for k, bits in enumerate([0, 1.5, 3, 4.5, 6]):
                d.rectangle(
                    [px + k * 44, 300, px + k * 44 + 40, 318],
                    fill=heat(2**-bits),
                    outline=G["line"],
                )
            d.text((px, 326), "0 bits", font=label, fill=G["faint"])
            d.text((px + 216, 326), "6+ bits", font=label, fill=G["faint"], anchor="ra")
            ps = [st["p"] for st in steps]
            d.text(
                (px, 372),
                f"mean p(picked) {sum(ps) / len(ps):.2f}\n"
                f"tokens with p < 10%: {sum(p < 0.1 for p in ps)} / {len(ps)}",
                font=small,
                fill=G["muted"],
                spacing=10,
            )
        # footer
        d.line([0, H - 54, W, H - 54], fill=G["line"])
        foot = (
            f"temperature {data['temperature']}  ·  top-k {data['top_k']}  ·  seed {data['seed']}"
            f"  ·  {data['tokens_seen'] / 1e9:.2f}B tokens seen  ·  real model output, unedited"
        )
        d.text((tx0, H - 34), foot, font=small, fill=G["faint"])
        return img

    frames, durs = [frame(0, None)], [900]
    for i in range(len(steps)):
        frames.append(frame(i, i))  # distribution, before the pick lands
        durs.append(110)
        frames.append(frame(i + 1, i))
        durs.append(70)
    frames.append(frame(len(steps), None, shade=True))
    durs.append(5000)
    save_gif(frames, durs, "stream.gif")


def race_gif(rows):
    data = json.loads((DATA / "race.json").read_text())
    W, H = 1200, 700
    tf = pil_font("JBMono-400.ttf", 14)
    tfb = pil_font("JBMono-700.ttf", 14)
    head = pil_font("Inter-600.ttf", 20)
    small = pil_font("JBMono-400.ttf", 13)
    label = pil_font("JBMono-700.ttf", 12)
    cw, lh = tf.getbbox("M")[2], 22
    colw, gap, x0, top = 270, 22, 34, 186
    cols = int((colw - 8) / cw)
    max_rows = int((H - 80 - top) / lh)
    val = {r["step"]: r["val/loss"] for r in rows if "val/loss" in r}
    runs = []
    for run in data["runs"]:
        chunks = [(data["prompt"], "prompt")] + [
            (s["text"], "gen") for s in run["steps"]
        ]
        cells = [c for c in layout(chunks, cols) if c[0] < max_rows]
        near = min(val, key=lambda s: abs(s - run["step"]))
        runs.append((run, cells, val[near]))
    n = len(data["runs"][0]["steps"])

    def frame(k):
        img, d = window(
            W, H, "mini-llm-lab  ·  same prompt, same seed, four checkpoints"
        )
        d.text((x0, 64), "WATCHING IT LEARN", font=label, fill=G["orange"])
        d.text(
            (x0, 84),
            "Every column runs the identical sampler; only the weights differ.",
            font=pil_font("Inter-400.ttf", 17),
            fill=G["muted"],
        )
        for i, (run, cells, vl) in enumerate(runs):
            cx = x0 + i * (colw + gap)
            final = run["checkpoint"] == "final_model"
            accent = G["orange"] if final else G["blue"]
            d.rounded_rectangle(
                [cx - 12, 124, cx + colw, H - 66],
                10,
                outline=accent if final else G["line"],
                fill=G["bg"],
            )
            d.text((cx, 136), f"step {run['step']:,}", font=head, fill=G["ink"])
            d.text(
                (cx, 162),
                f"{run['tokens_seen'] / 1e9:.2f}B tok · val {vl:.2f}",
                font=small,
                fill=G["muted"],
            )
            last = None
            for row, col, ch, tag, ci in cells:
                if ci > k:
                    break
                x, y = cx + col * cw, top + 8 + row * lh
                d.text(
                    (x, y),
                    ch,
                    font=tfb if tag == "prompt" else tf,
                    fill=G["muted"] if tag == "prompt" else G["ink"],
                )
                last = (x + cw, y)
            if last and k < n:
                d.rectangle(
                    [last[0] + 1, last[1], last[0] + cw - 2, last[1] + lh - 6],
                    fill=accent,
                )
        d.line([0, H - 54, W, H - 54], fill=G["line"])
        foot = (
            f"prompt “{data['prompt']}”  ·  temperature {data['temperature']}  ·  top-k {data['top_k']}"
            f"  ·  seed {data['seed']}  ·  val = 50-batch probe"
        )
        d.text((x0, H - 34), foot, font=small, fill=G["faint"])
        return img

    frames, durs = [frame(0)], [800]
    for k in range(1, n + 1):
        frames.append(frame(k))
        durs.append(85)
    durs[-1] = 5000
    save_gif(frames, durs, "race.gif")


def main():
    ensure_fonts()
    rows = load_history()
    only = set(sys.argv[1:])
    for theme in ("light", "dark"):
        if not only or "hero" in only:
            hero(rows, theme)
        if not only or "training" in only:
            training(rows, theme)
        if not only or "grads" in only:
            grad_layers(rows, theme)
        if not only or "arch" in only:
            architecture(theme)
    if not only or "stream" in only:
        stream_gif()
    if not only or "race" in only:
        race_gif(rows)


if __name__ == "__main__":
    main()
