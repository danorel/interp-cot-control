from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml

from experiments.forbidden_word.data import Problem, Sample
from experiments.forbidden_word.lens import CoTLens
from experiments.forbidden_word.params import SwapParams
from experiments.forbidden_word.swap import (
    ClassCurves,
    SwapScore,
    bin_means,
    classify_positions,
    position_bins,
    score_cot,
    summarize_swap,
    tokens_in_spans,
)

CONFIG = Path(__file__).parents[1] / "experiments/forbidden_word/swap_total.yaml"
# One token per character: 'T' plays the target word, 'S' the other word, 'U' a substitute.
SETS = {"target": [ord("T")], "other": [ord("S")], "substitute": [ord("U")]}


class CharLens:
    """Stand-in for CoTLens: '|' closes the CoT, and the 'lens value' at a position is the
    code of the character there — so equal values mean the same token was read."""

    def __init__(self) -> None:
        self.token_sets = SETS
        self.max_positions = None
        self.layers = [0]

    def cot_span(self, prompt: str, completion: str) -> tuple[list[int], int, int]:
        ids = [ord(c) for c in prompt + completion]
        start = len(prompt)
        end = ids.index(ord("|"), start) if ord("|") in ids[start:] else len(ids)
        return ids, start, end

    def per_position(self, texts: list[str], positions: list[int]) -> dict[str, torch.Tensor]:
        codes = torch.tensor([[[float(ord(t[p])) for p in positions]] for t in texts])
        return {"target": codes, "other": torch.zeros_like(codes), "substitute": -codes}

    def token_starts(self, text: str) -> list[int]:
        return list(range(len(text)))


def _sample(cond: str, prompt: str, completion: str) -> Sample:
    return Sample(Problem("p", "q", None, False), cond, prompt, completion)


def _score_cot(
    lens: Any, source: Sample, prompts: dict[str, str], meta: list[tuple[int, int]] | None = None
) -> list[SwapScore] | None:
    return score_cot(
        cast(CoTLens, lens), source, prompts, n_bins=2, max_class_positions=64, seed=0,
        meta_spans=meta or [],
    )  # fmt: skip


def _chars(ids: list[int], classes: dict[str, list[int]]) -> dict[str, str]:
    return {c: "".join(chr(ids[p]) for p in ps) for c, ps in classes.items()}


def test_classify_positions_by_next_token():
    ids = [ord(c) for c in "aTbUcSd"]
    classes = classify_positions(ids, 0, len(ids), SETS)
    # a->T target, b->U substitute, c->S skipped (other word), last char has no next token.
    assert _chars(ids, classes) == {"target": "a", "substitute": "b", "neutral": "TUS", "meta": ""}


def test_meta_sentence_takes_precedence_over_target():
    # Completion "aT.bT|": the span "bT" is a meta sentence quoting the word.
    source = _sample("forbid", "P:", "aT.bT|x")
    scores = _score_cot(CharLens(), source, {"forbid": "P:"}, meta=[(3, 5)])
    assert scores is not None
    classes = scores[0].classes
    # a->T is a genuine use; '.' and 'b' precede tokens inside the meta sentence.
    assert classes["target"].target_logprob == [ord("a")]
    assert classes["meta"].n == 2
    assert classes["meta"].target_logprob == [(ord(".") + ord("b")) / 2]
    assert classes["neutral"].target_logprob == [ord("T")]


def test_tokens_in_spans_uses_offsets_relative_to_completion():
    assert tokens_in_spans([0, 2, 5, 7, 9], base=4, spans=[(1, 4)]) == {2, 3}


def test_prompts_of_different_length_read_the_same_cot_tokens():
    source = _sample("forbid", "PROMPT-FORBID:", "aTbUcSd|answer")
    scores = _score_cot(CharLens(), source, {"forbid": "PROMPT-FORBID:", "control": "CTRL:"})
    assert scores is not None
    for s in scores:  # both prompts read exactly the same CoT characters per class
        assert s.target_logprob == pytest.approx([(ord("T") + ord("U") + ord("S")) / 3])
        assert s.n_positions == 3
        assert s.classes["target"] == ClassCurves(1, [ord("a")], [-ord("a")])
        assert s.classes["substitute"].target_logprob == [ord("b")]


def test_class_absent_when_cot_has_no_such_position():
    scores = _score_cot(CharLens(), _sample("control", "P:", "abcd|x"), {"control": "P:"})
    assert scores is not None and set(scores[0].classes) == {"neutral"}


def test_score_cot_skips_when_a_prompt_retokenises_the_cot_differently():
    class Shifting(CharLens):
        def cot_span(self, prompt: str, completion: str) -> Any:
            ids, start, end = super().cot_span(prompt, completion)
            return (ids, start + 1, end) if prompt.startswith("CTRL") else (ids, start, end)

    source = _sample("forbid", "P:", "abcd|x")
    assert _score_cot(Shifting(), source, {"forbid": "P:", "control": "CTRL:"}) is None


def test_position_bins_and_bin_means():
    assert position_bins([0, 49, 50, 99], cot_len=100, n_bins=5) == [0, 2, 2, 4]
    assert bin_means(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), [0, 0, 1, 1], 3) == [[1.5, 3.5, None]]


def _score(
    pid: str,
    prompt: str,
    logprob: list[float],
    by_bin: list[list[float | None]],
    target_class: list[float] | None = None,
) -> SwapScore:
    classes = {"neutral": ClassCurves(4, logprob, [0.0, 0.0])}
    if target_class is not None:
        classes["target"] = ClassCurves(2, target_class, [0.0, 0.0])
    return SwapScore(pid, "forbid", prompt, [0, 1], 4, logprob, by_bin, [0.0, 0.0], classes)


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


def test_summarize_classes_uses_only_cots_that_have_the_class():
    scores = [
        _score("a", "forbid", [0.0, 0.0], [[0.0], [0.0]], target_class=[-1.0, -5.0]),
        _score("a", "control", [0.0, 0.0], [[0.0], [0.0]], target_class=[-1.0, -2.0]),
        _score("b", "forbid", [0.0, 0.0], [[0.0], [0.0]]),  # no violation in this CoT
        _score("b", "control", [0.0, 0.0], [[0.0], [0.0]]),
    ]
    target = summarize_swap(scores, n_bins=1)["by_cot"]["forbid"]["by_class"]["target"]
    assert (target["n_cots"], target["n_positions"]) == (1, 2)
    assert [c["mean"] for c in target["paired_target_logprob"]["forbid-control"]] == [0.0, -3.0]
    assert [c["mean"] for c in target["mean"]["forbid"]["target_logprob"]] == [-1.0, -5.0]


def test_swap_config_validates():
    params = SwapParams.model_validate(yaml.safe_load(CONFIG.read_text())["params"])
    assert set(params.cot_conditions) <= set(params.prompt_conditions)
    assert params.substitutes
