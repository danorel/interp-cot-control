"""Prompt swap: the same CoT text read under different instructions.

The main lens comparison contrasts CoTs that the model *wrote* under different instructions,
so differences mix the instruction's effect with differences between the texts (length,
wording, which positions get sampled). Here each CoT sampled under condition C is re-read
teacher-forced under the prompt of every condition P. Lens positions are chosen once per CoT
and addressed from the end of the text, so every prompt is measured at the same CoT tokens:
P1 − P2 on the same CoT is the instruction's effect alone.

    uv run interp-run experiments/forbidden_word/swap.yaml
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from tqdm import tqdm

from experiments.forbidden_word.data import Sample, load_samples
from experiments.forbidden_word.lens import CoTLens, lens_positions
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

Bins = list[list[float | None]]  # [layer][bin]; None = no sampled position in that bin


@dataclass(frozen=True)
class SwapScore:
    problem_id: str
    cot_condition: str  # condition the CoT was sampled under
    prompt_condition: str  # condition whose prompt it is re-read under
    layers: list[int]
    n_positions: int
    target_logprob: list[float]  # [layer], mean over positions
    target_logprob_by_bin: Bins
    other_logprob: list[float]


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
    lens: CoTLens, source: Sample, prompts: Mapping[str, str], n_bins: int, seed: int
) -> list[SwapScore] | None:
    """Re-read `source`'s completion under each prompt; None if positions can't be aligned."""
    span = lens.cot_span(source.prompt, source.completion)
    if span is None:
        return None
    ids, start, end = span
    positions = lens_positions(ids, start, end, lens.exclude_next, lens.max_positions, seed)
    if not positions:
        return None
    # Every prompt must tokenise the completion identically, or the same offsets from the end
    # would point at different tokens.
    for prompt in prompts.values():
        other = lens.cot_span(prompt, source.completion)
        if other is None or other[0][other[1] :] != ids[start:]:
            return None

    from_end = [p - len(ids) for p in positions]
    bins = position_bins([p - start for p in positions], end - start, n_bins)
    names = list(prompts)
    logprobs = lens.per_position([prompts[n] + source.completion for n in names], from_end)
    return [
        SwapScore(
            problem_id=source.problem.id,
            cot_condition=source.condition,
            prompt_condition=name,
            layers=lens.layers,
            n_positions=len(positions),
            target_logprob=logprobs["target"][i].mean(-1).tolist(),
            target_logprob_by_bin=bin_means(logprobs["target"][i], bins, n_bins),
            other_logprob=logprobs["other"][i].mean(-1).tolist(),
        )
        for i, name in enumerate(names)
    ]


# ---- aggregation -------------------------------------------------------------------------


def _ci_per_layer(diffs: list[list[float]]) -> list[dict[str, float]]:
    n_layers = len(diffs[0]) if diffs else 0
    return [mean_ci([d[j] for d in diffs]) for j in range(n_layers)]


def summarize_swap(scores: Iterable[SwapScore], n_bins: int) -> dict[str, Any]:
    """Per CoT condition: mean curves per prompt, and paired A − B contrasts over problems,
    per layer and per (layer, relative-position bin)."""
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
                [
                    [x - y for x, y in zip(sa[i].target_logprob, sb[i].target_logprob, strict=True)]
                    for i in shared
                ]
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
        )
        scores: list[SwapScore] = []
        n_skipped = 0
        for pid, cot in tqdm(jobs, desc="swap", mininterval=10):
            prompts = {
                p: samples[pid, p].prompt for p in params.prompt_conditions if (pid, p) in samples
            }
            seed = zlib.crc32(f"{pid}/{cot}".encode())
            result = score_cot(lens, samples[pid, cot], prompts, params.n_bins, seed)
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
