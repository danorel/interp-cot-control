"""Static figures for the README, built from finished runs (no model needed).

    uv run python -m experiments.forbidden_word.report \
        --main outputs/forbidden_word/<generation or lens run> \
        --swap outputs/forbidden_word_swap/<swap run> \
        --out reports/figures/total

The words (target, other, substitutes) are read from the swap run's config, so the same code
draws experiment 1 ("total") and experiment 2 ("factor"). Figures whose data a run lacks are
skipped (a generation-only run has no different-text lens).

Lens figures show layers 0-34. Layer 35 (the output distribution) is left out: log P of a rare
token there is dominated by the tail and swings with the CoT source. Lens differences are
plotted on a log scale but labelled as plain multipliers ("2× less likely").
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Categorical slots validated for colour-vision deficiency (adjacent pairs) on a light surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
REF = "#898781"  # reference series / muted ink
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LATE_BAND = (24, 34)  # where the word-specific suppression sits in Qwen3-8B
N_LENS_LAYERS = 35  # layers 0..34
MULTIPLIERS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 2 / 3, 0.8, 1, 1.25, 1.5, 2, 4)

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


@dataclass(frozen=True)
class Words:
    target: str
    other: str
    substitutes: list[str]

    @classmethod
    def from_swap_run(cls, swap_dir: Path) -> Words:
        params = yaml.safe_load((swap_dir / "config.yaml").read_text())["params"]
        return cls(params["target"]["word"], params["other"]["word"], params["substitutes"])

    def q(self, word: str) -> str:
        return f"“{word}”"


# ---- helpers --------------------------------------------------------------------------------


def _line(ax: Axes, ci: CI, color: str, label: str, dashed: bool = False) -> None:
    ci = ci[:N_LENS_LAYERS]
    x = range(len(ci))
    ax.fill_between(x, [c["lo"] for c in ci], [c["hi"] for c in ci], color=color, alpha=0.14, lw=0)
    ax.plot(x, [c["mean"] for c in ci], color=color, lw=2, ls="--" if dashed else "-", label=label)


def _multiplier_label(m: float) -> str:
    if math.isclose(m, 1):
        return "no change"
    return f"{1 / m:g}× less likely" if m < 1 else f"{m:g}× more likely"


def _multiplier_axis(ax: Axes, ylabel: str) -> None:
    """Log-difference axis labelled as plain multipliers."""
    lo, hi = ax.get_ylim()
    ticks = [m for m in MULTIPLIERS if lo <= math.log(m) <= hi]
    for coarse in ((0.8, 1.25), (2 / 3, 1.5)):  # drop the finest steps while crowded
        if len(ticks) > 6:
            ticks = [m for m in ticks if m not in coarse]
    ax.set_yticks([math.log(m) for m in ticks], [_multiplier_label(m) for m in ticks])
    ax.set_ylabel(ylabel)


def _probability_axis(ax: Axes, ylabel: str) -> None:
    """log P axis labelled as percentages."""
    lo, hi = ax.get_ylim()
    ticks = [k for k in range(0, 12) if lo <= math.log(10**-k) <= hi]
    labels = [f"{100 * 10**-k:g}%" if k <= 3 else rf"$10^{{-{k}}}$" for k in ticks]
    ax.set_yticks([math.log(10**-k) for k in ticks], labels)
    ax.set_ylabel(f"{ylabel}, log scale")


def _layer_axes(ax: Axes, band_label: str) -> None:
    lo, hi = LATE_BAND
    ax.axvspan(lo - 0.5, hi + 0.5, color=BLUE, alpha=0.05, lw=0)
    ax.text(lo, ax.get_ylim()[1], f" {band_label}", va="top", fontsize=9, color=REF)
    ax.set_xlabel("Layer (0 = input side, 35 = output)")
    ax.set_xlim(-0.5, N_LENS_LAYERS - 0.5)


def _titles(ax: Axes, title: str, how_to_read: str) -> None:
    """Takeaway as the title; a how-to-read note under it (the pad leaves room for its lines)."""
    lines = how_to_read.count("\n") + 1
    ax.set_title(title, loc="left", pad=10 + 13.5 * lines)
    ax.text(0, 1.012, how_to_read, transform=ax.transAxes, va="bottom", fontsize=9.5, color=INK_2)


def _contrast_figure(
    title: str, how_to_read: str, lines: list[tuple[CI, str, str, bool]]
) -> Figure:
    """Same-text contrasts (prompt A vs prompt B) by layer, on a multiplier axis."""
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for ci, color, label, dashed in lines:
        _line(ax, ci, color, label, dashed)
    ax.axhline(0, color=AXIS, lw=1)
    _multiplier_axis(ax, "P(next word = target): A vs B")
    _layer_axes(ax, "late layers 24–34")
    _titles(ax, title, how_to_read)
    ax.legend(loc="lower left")
    return fig


def _save(fig: Figure, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _class(swap: dict[str, Any], cot: str, cls: str) -> dict[str, Any] | None:
    by_class = swap["by_cot"].get(cot, {}).get("by_class", {})
    return by_class.get(cls) if by_class.get(cls, {}).get("n_cots") else None


# ---- figures --------------------------------------------------------------------------------


def fig_swap_contrasts(swap: dict[str, Any], w: Words, out: Path) -> Path:
    paired = swap["by_cot"]["forbid"]["paired"]
    t, o = w.q(w.target), w.q(w.other)
    fig = _contrast_figure(
        f"Where does the ban act? Same text, two prompts, by layer ({t})",
        f"Each line compares two prompts on the identical CoT (written under the ban): how much "
        f"likelier the model is\nto say {t} next under prompt A than B. Shaded: 95% CI over "
        f"problems. Flat at “no change” = the instruction does nothing there.",
        [
            (paired["forbid-control"], BLUE, "ban vs control (the prohibition)", False),
            (
                paired["forbid-random"],
                ORANGE,
                f"ban on {t} vs ban on {o} (word-specific part)",
                False,
            ),
            (paired["random-control"], AQUA, f"ban on {o} vs control", False),
            (paired["grader-forbid"], YELLOW, "ban + monitoring threat vs ban alone", False),
        ],
    )
    return _save(fig, out, "swap_contrasts.png")


def fig_text_artifact(
    main: dict[str, Any], swap: dict[str, Any], w: Words, out: Path
) -> Path | None:
    if not main.get("lens"):
        return None
    fig = _contrast_figure(
        f"Ban vs control for {w.q(w.target)}: different texts vs the same text",
        f"Ban vs control for {w.q(w.target)}, measured two ways. Dashed: each prompt read on the "
        f"CoT it wrote (texts differ).\nSolid: the same CoT read under both prompts. Only the "
        f"late drop survives when the text is held fixed.",
        [
            (main["lens"]["paired_target_logprob"]["forbid-control"], REF,
             "different texts (first analysis)", True),
            (swap["by_cot"]["forbid"]["paired"]["forbid-control"], BLUE,
             "same text (prompt swap)", False),
        ],
    )  # fmt: skip
    return _save(fig, out, "text_artifact.png")


def fig_position_heatmap(swap: dict[str, Any], w: Words, out: Path) -> Path:
    grid = swap["by_cot"]["forbid"]["paired_by_bin"]["forbid-random"]
    rows = list(range(16, N_LENS_LAYERS))
    values = [[c["mean"] for c in grid[j]] for j in rows]
    lim = math.log(4)
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", "#c9403f"])
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.grid(False)
    im = ax.imshow(values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    for r, row in enumerate(values):
        for k, v in enumerate(row):
            m = math.exp(v)
            label = "same" if abs(v) < 0.05 else (f"÷{1 / m:.1f}" if m < 1 else f"×{m:.1f}")
            ax.text(k, r, label, ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.8 else INK)  # fmt: skip
    ax.set_xticks(range(5), ["first 20%", "20–40%", "40–60%", "60–80%", "last 20%"])
    ax.set_yticks(range(len(rows)), [f"L{j}" for j in rows])
    ax.set_xlabel("Part of the chain of thought")
    ax.set_ylabel("Layer")
    _titles(
        ax,
        f"Ban on {w.q(w.target)} vs ban on {w.q(w.other)}, by layer and part of the CoT",
        f"Ban on {w.q(w.target)} vs ban on {w.q(w.other)}, same CoT. "
        f"Cell: how many times less (÷)\nor more (×) likely {w.q(w.target)} is next. "
        f"Blue = suppressed. Layers 0–15 are all ≈ same.",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks(
        [math.log(m) for m in (1 / 4, 1 / 2, 1, 2, 4)], labels=["÷4", "÷2", "same", "×2", "×4"]
    )
    cbar.outline.set_visible(False)
    return _save(fig, out, "position_heatmap.png")


def fig_text_rate(main: dict[str, Any], w: Words, out: Path) -> Path:
    order = ["baseline", "control", "forbid", "random", "grader"]
    names = {
        "baseline": "no instruction",
        "control": f"{w.q(w.target)} allowed",
        "forbid": f"{w.q(w.target)} banned",
        "random": f"{w.q(w.other)} banned",
        "grader": "banned + monitored",
    }
    rate = {c: main["text"]["per_condition"][c]["target_per_1k_chars"] for c in order}
    cut = 1 - rate["forbid"]["mean"] / rate["control"]["mean"]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    y = list(range(len(order)))[::-1]
    ax.barh(y, [rate[c]["mean"] for c in order], height=0.5, color=BLUE)
    for yy, c in zip(y, order, strict=True):
        r = rate[c]
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=INK_2, lw=1.5)
        ax.text(r["hi"] + 0.04, yy, f"{r['mean']:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [names[c] for c in order])
    ax.set_xlabel(f"Uses of {w.q(w.target)} per 1,000 characters of reasoning (mean, 95% CI)")
    ax.set_title(
        f"Uses of {w.q(w.target)} in the reasoning (the ban cuts it by {cut:.0%})",
        loc="left",
    )
    return _save(fig, out, "text_rate.png")


def fig_violations(swap: dict[str, Any], w: Words, out: Path) -> Path | None:
    natural, violation = _class(swap, "control", "target"), _class(swap, "forbid", "target")
    neutral = _class(swap, "forbid", "neutral")
    if not (natural and violation and neutral):
        return None
    t = w.q(w.target)
    fig = _contrast_figure(
        f"Ban vs control where the next word is {t}: natural uses vs violations",
        f"Ban vs control on the same CoT, at positions where the next word is {t}. Blue: CoTs "
        f"written without a ban,\nwhere the model naturally says {t}. Orange: CoTs written under "
        f"the ban, where it said {t} anyway.",
        [
            (natural["paired_target_logprob"]["forbid-control"], BLUE,
             f"before {t} in control CoTs (natural use)", False),
            (violation["paired_target_logprob"]["forbid-control"], ORANGE,
             f"before {t} in banned CoTs (violation)", False),
            (neutral["paired_target_logprob"]["forbid-control"], REF,
             "all other positions (banned CoTs)", True),
        ],
    )  # fmt: skip
    return _save(fig, out, "violations.png")


def fig_substitutes(swap: dict[str, Any], w: Words, out: Path) -> Path | None:
    banned, natural = _class(swap, "forbid", "substitute"), _class(swap, "control", "substitute")
    neutral = _class(swap, "forbid", "neutral")
    if not (banned and natural and neutral):
        return None
    subs = " / ".join(w.substitutes)
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    _line(ax, banned["mean"]["forbid"]["target_logprob"], BLUE, f"before {subs}, under the ban")
    _line(ax, natural["mean"]["control"]["target_logprob"], ORANGE,
          "before the same words, no ban (natural use)")  # fmt: skip
    _line(ax, neutral["mean"]["forbid"]["target_logprob"], REF, "all other positions (ban)", True)
    _probability_axis(ax, f"P(next word = {w.q(w.target)})")
    _layer_axes(ax, "late layers 24–34")
    _titles(
        ax,
        f"Is {w.q(w.target)} a hidden candidate where the model writes a substitute?",
        f"Positions where the model is about to write {subs}; each CoT read under its own "
        f"prompt.\nA gap between blue and orange = under the ban the model was “thinking” "
        f"{w.q(w.target)} where it wrote the substitute.",
    )
    ax.legend(loc="lower left")
    return _save(fig, out, "substitutes.png")


def fig_meta(swap: dict[str, Any], w: Words, out: Path) -> Path | None:
    meta, neutral = _class(swap, "forbid", "meta"), _class(swap, "forbid", "neutral")
    if not (meta and neutral):
        return None
    t = w.q(w.target)
    fig = _contrast_figure(
        f"Ban vs control inside meta sentences vs ordinary reasoning ({t})",
        f"Ban vs control on the same CoT. Orange: inside meta sentences, where the model talks "
        f"about the word\n(“the problem says not to use the word {t}”). Above “no change” = "
        f"{t} likelier because of the ban.",
        [
            (meta["paired_target_logprob"]["forbid-control"], ORANGE,
             "inside meta sentences (talking about the word)", False),
            (neutral["paired_target_logprob"]["forbid-control"], BLUE,
             "ordinary reasoning", False),
        ],
    )  # fmt: skip
    return _save(fig, out, "meta.png")


def build(main_dir: Path, swap_dir: Path, out: Path) -> list[Path]:
    main = json.loads((main_dir / "summary.json").read_text())
    swap = json.loads((swap_dir / "summary.json").read_text())
    words = Words.from_swap_run(swap_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = [
        fig_swap_contrasts(swap, words, out),
        fig_text_artifact(main, swap, words, out),
        fig_position_heatmap(swap, words, out),
        fig_text_rate(main, words, out),
        fig_violations(swap, words, out),
        fig_substitutes(swap, words, out),
        fig_meta(swap, words, out),
    ]
    return [p for p in made if p is not None]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--main", type=Path, required=True, help="generation or lens run directory")
    ap.add_argument("--swap", type=Path, required=True, help="prompt-swap run directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    for path in build(args.main, args.swap, args.out):
        print(path)


if __name__ == "__main__":
    main()
