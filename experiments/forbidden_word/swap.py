"""Prompt swap: the same CoT text read under different instructions.

The main lens comparison contrasts CoTs that the model *wrote* under different instructions,
so differences mix the instruction's effect with differences between the texts (length,
wording, which positions get sampled). Here each CoT sampled under condition C is re-read
teacher-forced under the prompt of every condition P. Lens positions are chosen once per CoT
and addressed from the end of the text, so every prompt is measured at the same CoT tokens:
P1 − P2 on the same CoT is the instruction's effect alone.

Positions are split by the token that comes next in the CoT:
    neutral     neither the target, the other word, nor a substitute (the main measurement)
    target      the target word: a violation in CoTs written under a prohibition
    substitute  a word the model may use instead of the target ("sum", "overall", ...)

    uv run interp-run experiments/forbidden_word/swap.yaml
"""

from __future__ import annotations

import random
import zlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from tqdm import tqdm

from experiments.forbidden_word.data import Sample, load_samples
from experiments.forbidden_word.lens import CoTLens
from experiments.forbidden_word.params import SwapParams
from experiments.forbidden_word.stats import mean_ci
from interptemp.experiment import Experiment
from interptemp.models.nnterp_model import NnterpModel

# (prompt A, prompt B), reported as A − B on the same CoT. forbid − random is the cleanest:
# both prompts contain a prohibition, only the forbidden word differs.
PROMPT_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("forbid", "control"),
    ("grader", "control"),
    ("random", "control"),
    ("forbid", "random"),
    ("grader", "forbid"),
)
POSITION_CLASSES = ("neutral", "target", "substitute")

Bins = list[list[float | None]]  # [layer][bin]; None = no sampled position in that bin


@dataclass(frozen=True)
class ClassCurves:
    """Lens curves averaged over one CoT's positions of one class, under one prompt."""

    n: int
    target_logprob: list[float]  # [layer] log P(target word)
    substitute_logprob: list[float]  # [layer] log P(any substitute)


@dataclass(frozen=True)
class SwapScore:
    problem_id: str
    cot_condition: str  # condition the CoT was sampled under
    prompt_condition: str  # condition whose prompt it is re-read under
    layers: list[int]
    n_positions: int  # neutral positions
    target_logprob: list[float]  # [layer], mean over neutral positions
    target_logprob_by_bin: Bins  # neutral positions by relative place in the CoT
    other_logprob: list[float]
    classes: dict[str, ClassCurves]  # position class -> curves; absent if the CoT has none


def classify_positions(
    token_ids: Sequence[int], start: int, end: int, token_sets: Mapping[str, Sequence[int]]
) -> dict[str, list[int]]:
    """Positions p in [start, end - 1) by the class of token p + 1 (see module docstring)."""
    target, other, sub = (set(token_sets[k]) for k in ("target", "other", "substitute"))
    classes: dict[str, list[int]] = {c: [] for c in POSITION_CLASSES}
    for p in range(start, end - 1):
        nxt = token_ids[p + 1]
        if nxt in target:
            classes["target"].append(p)
        elif nxt in sub:
            classes["substitute"].append(p)
        elif nxt not in other:
            classes["neutral"].append(p)
    return classes


def subsample(positions: list[int], max_n: int | None, seed: int) -> list[int]:
    if max_n is None or len(positions) <= max_n:
        return positions
    return sorted(random.Random(seed).sample(positions, max_n))


def position_bins(offsets: Sequence[int], cot_len: int, n_bins: int) -> list[int]:
    """Bin of each position by its relative place in the CoT (0 = first 1/n_bins)."""
    return [min(n_bins * o // cot_len, n_bins - 1) for o in offsets]


def bin_means(values: torch.Tensor, bins: Sequence[int], n_bins: int) -> Bins:
    """values: [layer, position] -> per layer, the mean over positions of each bin."""
    b = torch.tensor(bins)
    return [
        [row[b == k].mean().item() if (b == k).any() else None for k in range(n_bins)]
        for row in values
    ]


def score_cot(
    lens: CoTLens,
    source: Sample,
    prompts: Mapping[str, str],
    n_bins: int,
    max_class_positions: int,
    seed: int,
) -> list[SwapScore] | None:
    """Re-read `source`'s completion under each prompt; None if positions can't be aligned."""
    span = lens.cot_span(source.prompt, source.completion)
    if span is None:
        return None
    ids, start, end = span
    # Every prompt must tokenise the completion identically, or the same offsets from the end
    # would point at different tokens.
    for prompt in prompts.values():
        other = lens.cot_span(prompt, source.completion)
        if other is None or other[0][other[1] :] != ids[start:]:
            return None

    found = classify_positions(ids, start, end, lens.token_sets)
    picked = {
        "neutral": subsample(found["neutral"], lens.max_positions, seed),
        "target": subsample(found["target"], max_class_positions, seed + 1),
        "substitute": subsample(found["substitute"], max_class_positions, seed + 2),
    }
    if not picked["neutral"]:
        return None
    # One forward pass per prompt over all classes; remember each class's slice.
    order = [p for c in POSITION_CLASSES for p in picked[c]]
    slices, at = {}, 0
    for c in POSITION_CLASSES:
        slices[c] = slice(at, at + len(picked[c]))
        at += len(picked[c])

    names = list(prompts)
    lp = lens.per_position(
        [prompts[n] + source.completion for n in names], [p - len(ids) for p in order]
    )
    neutral = slices["neutral"]
    bins = position_bins([p - start for p in picked["neutral"]], end - start, n_bins)
    return [
        SwapScore(
            problem_id=source.problem.id,
            cot_condition=source.condition,
            prompt_condition=name,
            layers=lens.layers,
            n_positions=len(picked["neutral"]),
            target_logprob=lp["target"][i][:, neutral].mean(-1).tolist(),
            target_logprob_by_bin=bin_means(lp["target"][i][:, neutral], bins, n_bins),
            other_logprob=lp["other"][i][:, neutral].mean(-1).tolist(),
            classes={
                c: ClassCurves(
                    n=len(picked[c]),
                    target_logprob=lp["target"][i][:, slices[c]].mean(-1).tolist(),
                    substitute_logprob=lp["substitute"][i][:, slices[c]].mean(-1).tolist(),
                )
                for c in POSITION_CLASSES
                if picked[c]
            },
        )
        for i, name in enumerate(names)
    ]


# ---- aggregation -------------------------------------------------------------------------


def _ci_per_layer(curves: list[list[float]]) -> list[dict[str, float]]:
    """Per-layer mean with 95% CI over CoTs (pass per-CoT differences for paired CIs)."""
    n_layers = len(curves[0]) if curves else 0
    return [mean_ci([c[j] for c in curves]) for j in range(n_layers)]


def _diff(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def _summarize_classes(cot_index: Mapping[str, Mapping[str, SwapScore]]) -> dict[str, Any]:
    """cot_index: prompt -> problem -> score, for one CoT condition."""
    prompts = list(cot_index)
    out: dict[str, Any] = {}
    for cls in POSITION_CLASSES:
        first = [s for s in cot_index[prompts[0]].values() if cls in s.classes]
        out[cls] = {
            "n_cots": len(first),
            "n_positions": sum(s.classes[cls].n for s in first),
            "mean": {
                p: {
                    field: _ci_per_layer(
                        [
                            getattr(s.classes[cls], field)
                            for s in cot_index[p].values()
                            if cls in s.classes
                        ]
                    )
                    for field in ("target_logprob", "substitute_logprob")
                }
                for p in prompts
            },
            "paired_target_logprob": {
                f"{a}-{b}": _ci_per_layer(
                    [
                        _diff(sa.classes[cls].target_logprob, sb.classes[cls].target_logprob)
                        for i in sorted(cot_index[a].keys() & cot_index[b].keys())
                        if cls in (sa := cot_index[a][i]).classes
                        and cls in (sb := cot_index[b][i]).classes
                    ]
                )
                for a, b in PROMPT_CONTRASTS
                if a in cot_index and b in cot_index
            },
        }
    return out


def summarize_swap(scores: Iterable[SwapScore], n_bins: int) -> dict[str, Any]:
    """Per CoT condition: mean curves per prompt, paired A − B contrasts over problems per layer
    and per (layer, relative-position bin) on neutral positions, and per position class."""
    index: dict[tuple[str, str], dict[str, SwapScore]] = defaultdict(dict)
    for s in scores:
        index[s.cot_condition, s.prompt_condition][s.problem_id] = s
    if not index:
        return {}
    layers = next(iter(next(iter(index.values())).values())).layers
    cot_conditions = list(dict.fromkeys(c for c, _ in index))

    out: dict[str, Any] = {"layers": layers, "n_bins": n_bins, "by_cot": {}}
    for cot in cot_conditions:
        prompts = [p for c, p in index if c == cot]
        curves = {
            p: torch.tensor([s.target_logprob for s in index[cot, p].values()]).mean(0).tolist()
            for p in prompts
        }
        paired, paired_by_bin = {}, {}
        for a, b in PROMPT_CONTRASTS:
            if a not in prompts or b not in prompts:
                continue
            sa, sb = index[cot, a], index[cot, b]
            shared = sorted(sa.keys() & sb.keys())
            paired[f"{a}-{b}"] = _ci_per_layer(
                [_diff(sa[i].target_logprob, sb[i].target_logprob) for i in shared]
            )
            paired_by_bin[f"{a}-{b}"] = [
                [
                    mean_ci(
                        [
                            xa - xb
                            for i in shared
                            if (xa := sa[i].target_logprob_by_bin[j][k]) is not None
                            and (xb := sb[i].target_logprob_by_bin[j][k]) is not None
                        ]
                    )
                    for k in range(n_bins)
                ]
                for j in range(len(layers))
            ]
        out["by_cot"][cot] = {
            "n_problems": len(index[cot, prompts[0]]),
            "mean_target_logprob": curves,
            "paired": paired,
            "paired_by_bin": paired_by_bin,
            "by_class": _summarize_classes({p: index[cot, p] for p in prompts}),
        }
    return out


# ---- experiment --------------------------------------------------------------------------


class PromptSwapExperiment(Experiment):
    def run(self) -> dict[str, Any]:
        params = SwapParams.model_validate(self.params)
        if not isinstance(self.model, NnterpModel):
            raise TypeError("prompt swap needs the nnterp backend (activations)")

        samples = {s.key: s for s in load_samples(params.generations)}
        problem_ids = list(dict.fromkeys(pid for pid, _ in samples))[: params.n_problems]
        jobs = [
            (pid, c) for pid in problem_ids for c in params.cot_conditions if (pid, c) in samples
        ]
        self.log.info(
            f"{len(jobs)} CoTs x {len(params.prompt_conditions)} prompts from {params.generations}"
        )

        lens = CoTLens(
            self.model,
            target_forms=params.target.forms,
            other_forms=params.other.forms,
            layers=params.lens.layers,
            max_positions=params.lens.max_positions,
            topk=params.lens.topk,
            extra_forms={"substitute": params.substitutes},
        )
        self.log.info(
            "substitute tokens: "
            + ", ".join(
                repr(self.model.tokenizer.decode([i])) for i in lens.token_sets["substitute"]
            )
        )
        scores: list[SwapScore] = []
        n_skipped = 0
        for pid, cot in tqdm(jobs, desc="swap", mininterval=10):
            prompts = {
                p: samples[pid, p].prompt for p in params.prompt_conditions if (pid, p) in samples
            }
            seed = zlib.crc32(f"{pid}/{cot}".encode())
            result = score_cot(
                lens, samples[pid, cot], prompts, params.n_bins, params.max_class_positions, seed
            )
            if result is None:
                n_skipped += 1
            else:
                scores += result

        if n_skipped:
            self.log.warning(f"swap: {n_skipped}/{len(jobs)} CoTs skipped (misaligned/empty)")
        self.save_jsonl("swap_rows.jsonl", [asdict(s) for s in scores])
        return {"n_cots": len(jobs) - n_skipped, "n_skipped": n_skipped} | summarize_swap(
            scores, params.n_bins
        )
