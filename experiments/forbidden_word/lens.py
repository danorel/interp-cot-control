"""Logit lens over the CoT: does the residual stream still point at the word where the text
doesn't say it?

The sample is re-run teacher-forced (prompt + completion). At every layer, resid_post at
sampled CoT positions is unembedded (final norm + lm_head) and we read off the probability
mass on the word's tokens. Positions whose *next* token is the word are excluded: there the
lens trivially tracks the text, and we want the latent (non-verbalised) signal.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from experiments.forbidden_word.text import THINK_CLOSE
from interptemp.models.nnterp_model import NnterpModel
from interptemp.sites import resid_post

Unembed = Callable[[torch.Tensor], torch.Tensor]


def variant_token_ids(tokenizer: Any, forms: Sequence[str]) -> list[int]:
    """First token of each surface variant (±leading space, ±capitalised) of each form."""
    variants = {
        prefix + case for f in forms for case in (f, f.capitalize()) for prefix in ("", " ")
    }
    return sorted({tokenizer(v, add_special_tokens=False).input_ids[0] for v in variants})


def lens_positions(
    token_ids: Sequence[int],
    start: int,
    end: int,
    exclude_next: set[int],
    max_n: int | None,
    seed: int = 0,
) -> list[int]:
    """Positions p in [start, end - 1) whose next token is not in `exclude_next`, subsampled."""
    positions = [p for p in range(start, end - 1) if token_ids[p + 1] not in exclude_next]
    if max_n is not None and len(positions) > max_n:
        positions = sorted(random.Random(seed).sample(positions, max_n))
    return positions


@dataclass(frozen=True)
class SetStats:
    logprob: float  # mean over positions of log P(any token of the set)
    topk_hit: float  # fraction of positions with a token of the set in the top-k


@torch.inference_mode()
def lens_stats(
    h: torch.Tensor, unembed: Unembed, token_sets: Mapping[str, Sequence[int]], topk: int
) -> dict[str, SetStats]:
    """h: [n_positions, d_model] residuals of one layer."""
    logits = unembed(h).float()
    log_z = logits.logsumexp(-1)
    top = logits.topk(topk, dim=-1).indices
    stats = {}
    for name, ids in token_sets.items():
        idx = torch.tensor(list(ids))
        logprob = logits[:, idx].logsumexp(-1) - log_z
        hit = torch.isin(top, idx).any(-1).float()
        stats[name] = SetStats(logprob=logprob.mean().item(), topk_hit=hit.mean().item())
    return stats


@dataclass(frozen=True)
class LensScore:
    """Per-layer curves, aligned with `layers`."""

    layers: list[int]
    n_positions: int
    target_logprob: list[float]
    target_topk_hit: list[float]
    other_logprob: list[float]
    other_topk_hit: list[float]


class CoTLens:
    def __init__(
        self,
        model: NnterpModel,
        target_forms: Sequence[str],
        other_forms: Sequence[str],
        layers: Sequence[int] | None,
        max_positions: int | None,
        topk: int,
    ):
        self.model = model
        tok = model.tokenizer
        self.token_sets = {
            "target": variant_token_ids(tok, target_forms),
            "other": variant_token_ids(tok, other_forms),
        }
        self.exclude_next = {i for ids in self.token_sets.values() for i in ids}
        self.layers = list(layers) if layers is not None else list(range(model.num_layers))
        self.sites = resid_post(self.layers)
        self.max_positions, self.topk = max_positions, topk
        self.think_close_id = tok.convert_tokens_to_ids(THINK_CLOSE)

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.model.model  # nnterp StandardizedTransformer; modules outside a trace
        return raw.lm_head._module(raw.ln_final._module(h))

    def cot_positions(self, prompt: str, completion: str, seed: int) -> list[int] | None:
        """Lens positions inside the CoT; None if the prompt doesn't retokenise as a prefix."""
        prompt_ids = self.model.encode([prompt])["input_ids"][0].tolist()
        ids = self.model.encode([prompt + completion])["input_ids"][0].tolist()
        start = len(prompt_ids)
        if ids[:start] != prompt_ids:
            return None
        end = (
            ids.index(self.think_close_id, start)
            if self.think_close_id in ids[start:]
            else len(ids)
        )
        return lens_positions(ids, start, end, self.exclude_next, self.max_positions, seed)

    def score(self, prompt: str, completion: str, seed: int) -> LensScore | None:
        positions = self.cot_positions(prompt, completion, seed)
        if not positions:
            return None
        acts = self.model.activations([prompt + completion], self.sites, positions=positions)
        per_layer = [
            lens_stats(acts[s][0], self.unembed, self.token_sets, self.topk) for s in self.sites
        ]
        return LensScore(
            layers=self.layers,
            n_positions=len(positions),
            target_logprob=[st["target"].logprob for st in per_layer],
            target_topk_hit=[st["target"].topk_hit for st in per_layer],
            other_logprob=[st["other"].logprob for st in per_layer],
            other_topk_hit=[st["other"].topk_hit for st in per_layer],
        )
