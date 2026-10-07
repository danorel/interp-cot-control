from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml

from experiments.forbidden_word.data import Problem, Sample
from experiments.forbidden_word.lens import CoTLens
from experiments.forbidden_word.params import SwapParams
from experiments.forbidden_word.swap import (
    SwapScore,
    bin_means,
    position_bins,
    score_cot,
    summarize_swap,
)

CONFIG = Path(__file__).parents[1] / "experiments/forbidden_word/swap.yaml"


class CharLens:
    """Stand-in for CoTLens: one token per character, '|' closes the CoT, and the 'lens value'
    at a position is the code of the character there — so equal values = same token read."""

    def __init__(self) -> None:
        self.exclude_next = {ord("T")}  # 'T' plays the forbidden word
        self.max_positions = None
        self.layers = [0]

    def cot_span(self, prompt: str, completion: str) -> tuple[list[int], int, int]:
        ids = [ord(c) for c in prompt + completion]
        start = len(prompt)
        end = ids.index(ord("|"), start) if ord("|") in ids[start:] else len(ids)
        return ids, start, end

    def per_position(self, texts: list[str], positions: list[int]) -> dict[str, torch.Tensor]:
        codes = torch.tensor([[[float(ord(t[p])) for p in positions]] for t in texts])
        return {"target": codes, "other": torch.zeros_like(codes)}


def _sample(cond: str, prompt: str, completion: str) -> Sample:
    return Sample(Problem("p", "q", None, False), cond, prompt, completion)


def test_prompts_of_different_length_read_the_same_cot_tokens():
    source = _sample("forbid", "PROMPT-FORBID:", "abTcd|answer")
    prompts = {"forbid": "PROMPT-FORBID:", "control": "CTRL:"}
    scores = score_cot(cast(CoTLens, CharLens()), source, prompts, n_bins=2, seed=0)
    assert scores is not None
    by_prompt = {s.prompt_condition: s for s in scores}
    # Positions a, T-predecessor excluded ('b' precedes 'T'), T, c -> chars a, T, c.
    expected = (ord("a") + ord("T") + ord("c")) / 3
    assert by_prompt["forbid"].target_logprob == pytest.approx([expected])
    assert by_prompt["control"].target_logprob == pytest.approx([expected])
    assert by_prompt["forbid"].n_positions == 3


def test_score_cot_skips_when_a_prompt_retokenises_the_cot_differently():
    class Shifting(CharLens):
        def cot_span(self, prompt: str, completion: str) -> Any:
            ids, start, end = super().cot_span(prompt, completion)
            return (ids, start + 1, end) if prompt.startswith("CTRL") else (ids, start, end)

    source = _sample("forbid", "P:", "abcd|x")
    assert (
        score_cot(cast(CoTLens, Shifting()), source, {"forbid": "P:", "control": "CTRL:"}, 2, 0)
        is None
    )


def test_position_bins_and_bin_means():
    assert position_bins([0, 49, 50, 99], cot_len=100, n_bins=5) == [0, 2, 2, 4]
    assert bin_means(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), [0, 0, 1, 1], 3) == [[1.5, 3.5, None]]


def _score(
    pid: str, prompt: str, logprob: list[float], by_bin: list[list[float | None]]
) -> SwapScore:
    return SwapScore(pid, "forbid", prompt, [0, 1], 4, logprob, by_bin, [0.0, 0.0])


def test_summarize_swap_pairs_same_problem_and_skips_empty_bins():
    scores = [
        _score("a", "forbid", [-1.0, -3.0], [[-1.0, None], [-3.0, -3.0]]),
        _score("a", "control", [-1.0, -2.0], [[-1.0, -1.0], [-2.0, -2.0]]),
        _score("b", "forbid", [-2.0, -4.0], [[-2.0, -2.0], [-4.0, -4.0]]),
        _score("b", "control", [-2.0, -2.0], [[-2.0, -2.0], [-2.0, -2.0]]),
    ]
    s = summarize_swap(scores, n_bins=2)["by_cot"]["forbid"]
    assert s["n_problems"] == 2
    assert [c["mean"] for c in s["paired"]["forbid-control"]] == [0.0, -1.5]
    by_bin = s["paired_by_bin"]["forbid-control"]
    assert by_bin[0][1]["n"] == 1  # problem "a" has no position in that bin under forbid
    assert by_bin[1][1]["mean"] == -1.5
    assert "forbid-random" not in s["paired"]  # random prompt not scored


def test_swap_config_validates():
    params = SwapParams.model_validate(yaml.safe_load(CONFIG.read_text())["params"])
    assert set(params.cot_conditions) <= set(params.prompt_conditions)
