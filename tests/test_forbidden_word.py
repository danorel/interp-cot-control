from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from experiments.forbidden_word.data import Problem, Record, Sample
from experiments.forbidden_word.lens import LensScore, lens_positions, lens_stats
from experiments.forbidden_word.params import Params, SwapParams
from experiments.forbidden_word.stats import group, mean_ci, paired_diffs, summarize
from experiments.forbidden_word.text import (
    answers_match,
    is_meta,
    meta_spans,
    parse_gold,
    parse_pred,
    score_text,
    split_cot,
    word_pattern,
)

CONFIG = Path(__file__).parents[1] / "experiments/forbidden_word/config_total.yaml"
TOTAL = word_pattern(["total", "totals", "totaling"])
SUM = word_pattern(["sum", "sums"])


# ---- text ----------------------------------------------------------------------------------


def test_split_cot():
    cot = split_cot("<think>\nabc\n</think>\n\nAns \\boxed{3}")
    assert (cot.thinking, cot.answer, cot.truncated) == ("abc", "Ans \\boxed{3}", False)
    assert split_cot("<think>\nnever closes").truncated


def test_word_pattern_matches_whole_words_only():
    assert len(TOTAL.findall("Total is 5, totals, totaling; subtotal, totally")) == 3


@pytest.mark.parametrize(
    "s",
    [
        "I can't use the word total here.",
        "I shouldn’t mention 'total'.",
        "I must avoid saying total.",
        "Wait, the instruction says not to use that word.",
        "Careful: the forbidden word must not appear.",
        "Let me find the overall amount without using total.",
        'The question says not to use "sum" anywhere, so that\'s good.',  # seen in a real run
    ],
)
def test_meta_positive(s):
    assert is_meta(s, TOTAL) or is_meta(s, SUM)


@pytest.mark.parametrize(
    "s",
    [
        "So the total is 48 + 24 = 72.",
        "The total can't be negative.",  # prohibition without a verb of saying
        "We don't need to multiply here.",  # no reference to the word
        "I can't use the bus, so I walk.",
    ],
)
def test_meta_negative(s):
    assert not is_meta(s, TOTAL)


def test_meta_spans_point_at_meta_sentences_in_the_completion():
    completion = (
        '<think>\nThe total is 5. The problem says not to use the word "total".\n'
        "So it is 5.</think>\nThe total is 5."
    )
    spans = meta_spans(completion, TOTAL, SUM)
    assert [completion[a:b] for a, b in spans] == ['The problem says not to use the word "total".']
    assert meta_spans("<think>no meta here</think>", TOTAL, SUM) == []


def test_score_text_separates_meta_from_task_use():
    completion = (
        "<think>I should not use the word total. The total is 5.\n"
        "Adding gives the sum 5. The grader is watching.</think>\\boxed{5}"
    )
    s = score_text(completion, gold=5.0, target=TOTAL, other=SUM)
    assert (s.target.task, s.target.meta, s.other.task) == (1, 1, 1)
    assert len(s.meta_sentences) == 1 and s.monitor_mentions == 1
    assert s.correct and not s.truncated


def test_parse_answers_gsm8k():
    assert parse_pred("so \\boxed{1,234} and \\boxed{\\$72}") == "\\$72"
    assert parse_pred("The answer is 18.") == "18"
    assert parse_pred("") is None
    assert parse_gold("blah <<4*3=12>>12\n#### 1,200") == "1200"
    assert answers_match("\\$72", "72") and answers_match("1,200", "1200")
    assert answers_match("18", 18.0)  # golds stored as floats by older runs
    assert not answers_match("17", "18") and not answers_match(None, "18")


def test_parse_answers_math():
    solution = "So the answer is $\\boxed{\\frac{\\sqrt{3}}{2}}$."
    assert parse_gold(solution, "boxed") == "\\frac{\\sqrt{3}}{2}"
    assert parse_pred("Thus \\boxed{\\dfrac{\\sqrt{3}}{2}}") == "\\dfrac{\\sqrt{3}}{2}"
    assert answers_match("\\dfrac{\\sqrt{3}}{2}", "\\frac{\\sqrt{3}}{2}")
    assert answers_match("10,000", "10{,}000")
    assert answers_match("x = 5", "5")  # numeric gold: compare values
    assert answers_match("90^\\circ", "90")
    assert not answers_match("\\frac{1}{2}", "\\frac{1}{3}")
    assert parse_pred("cut off \\boxed{\\frac{1}{2") is None  # unclosed box (truncated)


# ---- params / data -------------------------------------------------------------------------


def test_config_params_validate_and_build_prompts():
    cfg = yaml.safe_load(CONFIG.read_text())
    params = Params.model_validate(cfg["params"])
    assert params.user_message("Q?", "baseline").endswith("\\boxed{}.")
    forbid = params.user_message("Q?", "forbid")
    assert forbid.startswith("Q?\n\n") and '"total"' in forbid
    assert '"sum"' in params.user_message("Q?", "random")


def test_factor_configs_validate():
    exp_dir = CONFIG.parent
    params = Params.model_validate(
        yaml.safe_load((exp_dir / "config_factor.yaml").read_text())["params"]
    )
    assert params.dataset.answer_style == "boxed" and params.lens.single_token_words
    assert '"factor"' in params.user_message("Q?", "forbid")
    pattern = params.target.pattern
    assert pattern.search("we factored it") and not pattern.search("5 factorial")
    swap = SwapParams.model_validate(
        yaml.safe_load((exp_dir / "swap_factor.yaml").read_text())["params"]
    )
    assert swap.target.forms == params.target.forms and swap.other.forms == params.other.forms


def test_sample_roundtrip():
    s = Sample(Problem("p1", "q", 3.0, False), "forbid", "prompt", "completion")
    assert Sample.from_dict(asdict(s)) == s


@pytest.mark.parametrize("lens", [None, [-1.0, -2.0]])
def test_record_roundtrip(lens):
    r = _record("p1", "forbid", 2, lens)
    assert Record.from_dict(asdict(r)) == r


# ---- lens ----------------------------------------------------------------------------------


def test_lens_positions_drops_positions_predicting_word():
    ids = [0, 1, 2, 9, 3, 9, 4]  # 9 = word token
    assert lens_positions(ids, 1, len(ids), {9}, None) == [1, 3, 5]
    assert lens_positions(ids, 1, 4, {9}, None) == [1]  # stops before CoT end
    sub = lens_positions(list(range(100)), 0, 100, set(), 10, seed=1)
    assert len(sub) == 10 and sub == sorted(sub)


def test_lens_stats_matches_softmax():
    logits = torch.tensor([[2.0, 0.0, 0.0, 1.0], [0.0, 0.0, 3.0, 0.0]])
    out = lens_stats(logits, lambda h: h, {"a": [0, 3], "b": [1]}, topk=1)
    expected = torch.log(logits.softmax(-1)[:, [0, 3]].sum(-1)).mean().item()
    assert out["a"].logprob == pytest.approx(expected, abs=1e-6)
    assert out["a"].topk_hit == 0.5  # top-1 is in the set only for row 0
    assert out["b"].topk_hit == 0.0


# ---- stats ---------------------------------------------------------------------------------


def _record(pid: str, cond: str, n_target: int, lens: list[float] | None = None) -> Record:
    text = score_text(f"<think>{' total' * n_target}</think>", None, TOTAL, SUM)
    score = None if lens is None else LensScore([0, 1], 1, lens, [0, 0], [0, 0], [0, 0])
    return Record(Sample(Problem(pid, "q", None, False), cond, "", ""), text, score)


def test_paired_diffs_only_over_shared_problems():
    groups = group(
        [_record("a", "forbid", 1), _record("b", "forbid", 0), _record("a", "control", 3)]
    )
    assert paired_diffs(groups, "forbid", "control", lambda r: r.text.target.task).tolist() == [-2]


def test_summarize_reports_at_risk_and_lens_contrasts():
    records = [
        _record("a", "baseline", 2, [-1.0, -2.0]),
        _record("b", "baseline", 0, [-1.0, -2.0]),
        _record("a", "control", 2, [-1.0, -2.0]),
        _record("a", "forbid", 0, [-1.5, -4.0]),
        _record("b", "forbid", 0),  # unscored lens: excluded from lens contrasts
    ]
    s = summarize(records)
    assert s["text"]["n_at_risk"] == 1
    assert s["text"]["per_condition"]["forbid"]["target_used"]["mean"] == 0.0
    assert [c["mean"] for c in s["lens"]["paired_target_logprob"]["forbid-control"]] == [-0.5, -2.0]


def test_per_1k_rate_separates_shorter_cot_from_compliance():
    def rec(cond: str, cot: str) -> Record:
        return Record(
            Sample(Problem("a", "q", None, False), cond, "", ""),
            score_text(f"<think>{cot}</think>", None, TOTAL, SUM),
        )

    # forbid: half the uses but also half the length -> same rate, i.e. no real compliance.
    # (No whitespace in the unit: split_cot strips, which would skew the length ratio.)
    s = summarize([rec("control", "total-x;" * 4), rec("forbid", "total-x;" * 2)])
    paired = s["text"]["paired"]
    assert paired["target_count"]["forbid-control"]["mean"] == -2
    assert paired["target_per_1k_chars"]["forbid-control"]["mean"] == pytest.approx(0.0)
    assert (
        summarize([rec("forbid", "")])["text"]["per_condition"]["forbid"]["target_per_1k_chars"][
            "mean"
        ]
        == 0.0
    )


def test_mean_ci():
    r = mean_ci([1.0] * 10)
    assert r["mean"] == r["lo"] == r["hi"] == 1.0
    assert mean_ci([])["n"] == 0
