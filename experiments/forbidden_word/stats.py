"""Aggregate scored records into per-condition rates and paired (same-problem) contrasts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from experiments.forbidden_word.data import Record

# condition -> problem id -> record
Groups = dict[str, dict[str, Record]]

# (treatment, reference). Control is the main reference: it mentions the word like the
# forbid prompts do, so the difference isolates the prohibition from mere priming.
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("forbid", "control"),
    ("grader", "control"),
    ("random", "control"),
    ("grader", "forbid"),
    ("control", "baseline"),
)

TEXT_METRICS: dict[str, Callable[[Record], float]] = {
    "target_used": lambda r: r.text.target.task > 0,
    "target_count": lambda r: r.text.target.task,
    "other_used": lambda r: r.text.other.task > 0,
    "meta_refusal": lambda r: len(r.text.meta_sentences) > 0,
    "instruction_ref": lambda r: r.text.instruction_refs > 0,
    "monitor_mention": lambda r: r.text.monitor_mentions > 0,
    "accuracy": lambda r: r.text.correct,
    "truncated": lambda r: r.text.truncated,
    "cot_chars": lambda r: r.text.cot_chars,
}


def mean_ci(x: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean with percentile-bootstrap 95% CI (pass per-problem differences for paired CIs)."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    boots = a[np.random.default_rng(seed).integers(0, a.size, (n_boot, a.size))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi), "n": int(a.size)}


def group(records: Iterable[Record]) -> Groups:
    groups: Groups = {}
    for r in records:
        groups.setdefault(r.sample.condition, {})[r.sample.problem.id] = r
    return groups


def paired_diffs(
    groups: Groups, treatment: str, reference: str, value: Callable[[Record], Any]
) -> np.ndarray:
    """value(treatment) - value(reference) per shared problem; skips problems where value is None."""
    a, b = groups[treatment], groups[reference]
    pairs = [(value(a[i]), value(b[i])) for i in a.keys() & b.keys()]
    return np.array([np.subtract(x, y) for x, y in pairs if x is not None and y is not None])


def _contrasts(groups: Groups) -> list[tuple[str, str]]:
    return [(a, b) for a, b in CONTRASTS if a in groups and b in groups]


def summarize_text(groups: Groups) -> dict[str, Any]:
    # "At risk": problems where the unprompted model actually says the word.
    at_risk = {i for i, r in groups.get("baseline", {}).items() if r.text.target.task > 0}
    per_condition = {
        cond: {
            name: mean_ci([float(f(r)) for r in rs.values()]) for name, f in TEXT_METRICS.items()
        }
        | {
            "target_used_at_risk": mean_ci(
                [float(rs[i].text.target.task > 0) for i in at_risk & rs.keys()]
            )
        }
        for cond, rs in groups.items()
    }
    paired = {
        f"{a}-{b}": mean_ci(paired_diffs(groups, a, b, lambda r: r.text.target.task).tolist())
        for a, b in _contrasts(groups)
    }
    return {
        "n_at_risk": len(at_risk),
        "per_condition": per_condition,
        "paired_target_count": paired,
    }


def summarize_lens(groups: Groups) -> dict[str, Any] | None:
    scored = [r.lens for rs in groups.values() for r in rs.values() if r.lens is not None]
    if not scored:
        return None

    def curve(rs: Iterable[Record], field: str) -> list[float]:
        arr = np.array([getattr(r.lens, field) for r in rs if r.lens is not None])
        return arr.mean(0).tolist() if arr.size else []

    def target_logprob(r: Record) -> list[float] | None:
        return r.lens.target_logprob if r.lens is not None else None

    paired = {}
    for a, b in _contrasts(groups):
        diffs = paired_diffs(groups, a, b, target_logprob)  # [n_problems, n_layers]
        paired[f"{a}-{b}"] = (
            [mean_ci(diffs[:, j].tolist()) for j in range(diffs.shape[1])] if diffs.size else []
        )
    return {
        "layers": scored[0].layers,
        "mean_by_condition": {
            cond: {f: curve(rs.values(), f) for f in ("target_logprob", "other_logprob")}
            for cond, rs in groups.items()
        },
        "paired_target_logprob": paired,
    }


def summarize(records: Iterable[Record]) -> dict[str, Any]:
    groups = group(records)
    return {"text": summarize_text(groups), "lens": summarize_lens(groups)}
