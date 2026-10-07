"""Static figures for the README, built from finished runs (no model needed).

    uv run python -m experiments.forbidden_word.report \
        --main outputs/forbidden_word/<lens run> \
        --swap outputs/forbidden_word_swap/<swap run> \
        --out reports/figures

All lens figures show layers 0-34; layer 35 (the output distribution) is left out because
log P of a rare token there is dominated by the tail and swings with the CoT source.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Categorical slots validated for colour-vision deficiency (adjacent pairs) on a light surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
REF = "#898781"  # reference series / muted ink
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SUPPRESSION_BAND = (24, 34)
N_LENS_LAYERS = 35  # layers 0..34

CI = list[dict[str, float]]  # per layer: {"mean", "lo", "hi", "n"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": REF,
        "ytick.color": REF,
        "font.size": 10.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


# ---- helpers --------------------------------------------------------------------------------


def _cut(ci: CI) -> CI:
    return ci[:N_LENS_LAYERS]


def _line(ax: Axes, ci: CI, color: str, label: str, dashed: bool = False) -> None:
    ci = _cut(ci)
    x = range(len(ci))
    ax.fill_between(x, [c["lo"] for c in ci], [c["hi"] for c in ci], color=color, alpha=0.14, lw=0)
    ax.plot(x, [c["mean"] for c in ci], color=color, lw=2, ls="--" if dashed else "-", label=label)


def _delta_axes(ax: Axes, title: str, subtitle: str) -> None:
    lo, hi = SUPPRESSION_BAND
    ax.axvspan(lo - 0.5, hi + 0.5, color=BLUE, alpha=0.05, lw=0)
    ax.text(lo, ax.get_ylim()[1], " suppression band (L24–34)", va="top", fontsize=9, color=REF)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ log P(total)")
    ax.set_xlim(-0.5, N_LENS_LAYERS - 0.5)
    ax.set_title(f"{title}\n", loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)
    ax.legend(loc="lower left")


def _figure(width: float = 9.0, height: float = 4.2) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _save(fig: Figure, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ---- figures --------------------------------------------------------------------------------


def fig_swap_contrasts(swap: dict[str, Any], out: Path) -> Path:
    """Same CoT text, different prompts: where does each instruction act?"""
    paired = swap["by_cot"]["forbid"]["paired"]
    fig, ax = _figure()
    _line(ax, paired["forbid-control"], BLUE, "forbid − control")
    _line(ax, paired["forbid-random"], ORANGE, "forbid − random (only the word differs)")
    _line(ax, paired["random-control"], AQUA, "random − control")
    _line(ax, paired["grader-forbid"], YELLOW, "grader − forbid")
    _delta_axes(
        ax,
        "Same text, different instruction: the word is suppressed only in late layers",
        "CoTs written under forbid, re-read under each prompt; mean over 120 problems, 95% CI. "
        "−0.7 ≈ 2× less likely.",
    )
    return _save(fig, out, "swap_contrasts.png")


def fig_text_artifact(main: dict[str, Any], swap: dict[str, Any], out: Path) -> Path:
    """The mid-layer rise in the first analysis came from comparing different texts."""
    fig, ax = _figure()
    _line(
        ax,
        main["lens"]["paired_target_logprob"]["forbid-control"],
        REF,
        "different texts (first analysis)",
        dashed=True,
    )
    _line(ax, swap["by_cot"]["forbid"]["paired"]["forbid-control"], BLUE, "same text (prompt swap)")
    _delta_axes(
        ax,
        "forbid − control: holding the text fixed removes the mid-layer rise",
        "Different texts: each condition read on the CoT it wrote. Same text: forbid CoTs read "
        "under both prompts.",
    )
    return _save(fig, out, "text_artifact.png")


def fig_position_heatmap(swap: dict[str, Any], out: Path) -> Path:
    """Late suppression by relative position within the CoT."""
    grid = swap["by_cot"]["forbid"]["paired_by_bin"]["forbid-random"]
    rows = list(range(16, N_LENS_LAYERS))
    values = [[c["mean"] for c in grid[j]] for j in rows]
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", "#c9403f"])
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.grid(False)
    im = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, aspect="auto", origin="lower")
    for r, row in enumerate(values):
        for k, v in enumerate(row):
            ax.text(k, r, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 else INK)  # fmt: skip
    ax.set_xticks(range(5), ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
    ax.set_yticks(range(len(rows)), [f"L{j}" for j in rows])
    ax.set_xlabel("Position within the CoT")
    ax.set_title("forbid − random by layer and CoT position\n", loc="left")
    ax.text(0, 1.01, "Same text (forbid CoTs). Blue: total less likely under forbid.",
            transform=ax.transAxes, fontsize=9.5, color=INK_2)  # fmt: skip
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Δ log P(total)", color=INK_2)
    cbar.outline.set_visible(False)
    return _save(fig, out, "position_heatmap.png")


def fig_text_rate(main: dict[str, Any], out: Path) -> Path:
    """Text-level effect, length-normalised."""
    order = ["baseline", "control", "forbid", "random", "grader"]
    rate = {c: main["text"]["per_condition"][c]["target_per_1k_chars"] for c in order}
    fig, ax = _figure(7.5, 3.4)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    y = range(len(order))[::-1]
    ax.barh(list(y), [rate[c]["mean"] for c in order], height=0.5, color=BLUE)
    for yy, c in zip(y, order, strict=True):
        r = rate[c]
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=INK_2, lw=1.5)
        ax.text(r["hi"] + 0.05, yy, f"{r['mean']:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(list(y), order)
    ax.set_xlabel("Uses of total per 1,000 CoT characters (mean, 95% CI)")
    ax.set_title("In text, forbidding total halves its rate; forbidding sum does not", loc="left")
    return _save(fig, out, "text_rate.png")


def fig_violations(swap: dict[str, Any], out: Path) -> Path:
    """Where the model breaks the rule, the late filter acts only at its background level."""
    bc = swap["by_cot"]
    fig, ax = _figure()
    _line(ax, bc["control"]["by_class"]["target"]["paired_target_logprob"]["forbid-control"],
          BLUE, "natural uses (control CoTs, next token total)")  # fmt: skip
    _line(ax, bc["forbid"]["by_class"]["target"]["paired_target_logprob"]["forbid-control"],
          ORANGE, "violations (forbid CoTs, next token total)")  # fmt: skip
    _line(ax, bc["forbid"]["by_class"]["neutral"]["paired_target_logprob"]["forbid-control"],
          REF, "ordinary positions (forbid CoTs)", dashed=True)  # fmt: skip
    _delta_axes(
        ax,
        "Violations: the filter does not scale up where the model writes the word",
        "forbid − control on the same text. Meta sentences excluded. Natural uses are strongly "
        "suppressed; violations only at the background level.",
    )
    return _save(fig, out, "violations.png")


def fig_substitutes(swap: dict[str, Any], out: Path) -> Path:
    """Before a substitute, 'total' is a strong hidden runner-up under the prohibition."""
    bc = swap["by_cot"]

    def level(cot: str, cls: str) -> CI:
        return bc[cot]["by_class"][cls]["mean"][cot]["target_logprob"]

    fig, ax = _figure()
    _line(ax, level("forbid", "substitute"), BLUE, "before a substitute, under forbid")
    _line(
        ax, level("control", "substitute"), ORANGE, "before a substitute, in control (natural use)"
    )
    _line(ax, level("forbid", "neutral"), REF, "ordinary positions (forbid)", dashed=True)
    lo, hi = SUPPRESSION_BAND
    ax.axvspan(lo - 0.5, hi + 0.5, color=BLUE, alpha=0.05, lw=0)
    ax.set_xlim(-0.5, N_LENS_LAYERS - 0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("log P(total)")
    ax.set_title(
        "Substitutes: total stays a strong hidden candidate until late layers\n", loc="left"
    )
    ax.text(0, 1.02, "Positions where the next token is sum / combined / overall / altogether / "
            "aggregate; CoT read under its own prompt.", transform=ax.transAxes, fontsize=9.5,
            color=INK_2)  # fmt: skip
    ax.legend(loc="lower left")
    return _save(fig, out, "substitutes.png")


def fig_meta(swap: dict[str, Any], out: Path) -> Path:
    """Talking about the rule raises the word mid-network, then it is suppressed hard."""
    by_class = swap["by_cot"]["forbid"]["by_class"]
    fig, ax = _figure()
    _line(ax, by_class["meta"]["paired_target_logprob"]["forbid-control"], ORANGE,
          "inside meta sentences (talking about the word)")  # fmt: skip
    _line(ax, by_class["neutral"]["paired_target_logprob"]["forbid-control"], BLUE,
          "ordinary positions")  # fmt: skip
    _delta_axes(
        ax,
        "Meta sentences: reasoning about the rule raises total mid-network",
        "forbid − control on the same text (forbid CoTs). Above zero: total more likely under "
        "the prohibition.",
    )
    return _save(fig, out, "meta.png")


def build(main_dir: Path, swap_dir: Path, out: Path) -> list[Path]:
    main = json.loads((main_dir / "summary.json").read_text())
    swap = json.loads((swap_dir / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    return [
        fig_swap_contrasts(swap, out),
        fig_text_artifact(main, swap, out),
        fig_position_heatmap(swap, out),
        fig_text_rate(main, out),
        fig_violations(swap, out),
        fig_substitutes(swap, out),
        fig_meta(swap, out),
    ]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--main", type=Path, required=True, help="lens run (stage 2) directory")
    ap.add_argument("--swap", type=Path, required=True, help="prompt-swap run directory")
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = ap.parse_args(argv)
    for path in build(args.main, args.swap, args.out):
        print(path)


if __name__ == "__main__":
    main()
