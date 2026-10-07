import pytest

from experiments.forbidden_word.meta_validation import (
    analyze,
    draw,
    estimate,
    parse_sizes,
    stratum,
    wilson,
)
from experiments.forbidden_word.text import word_pattern

TOTAL, SUM = word_pattern(["total"]), word_pattern(["sum"])


def test_stratum():
    assert stratum('The problem says not to use the word "total".', TOTAL, SUM) == "A"
    assert stratum("So the total is 5.", TOTAL, SUM) == "B"  # word present, not meta
    assert stratum("I'll use a different approach.", TOTAL, SUM) == "B"  # suspicious verb
    assert stratum("Then 3 + 4 = 7.", TOTAL, SUM) == "C"


def test_estimate_weights_strata_by_corpus_size():
    # A: 10 flagged, 90% true. B: 100 sentences, 10% true. C: 1000 sentences, 0% true.
    r = estimate({"A": 0.9, "B": 0.1, "C": 0.0}, {"A": 10, "B": 100, "C": 1000})
    assert r["precision"] == 0.9
    assert r["recall"] == pytest.approx(9 / (9 + 10))
    assert r["est_true_meta_sentences"] == pytest.approx(19)


def test_analyze_joins_labels_and_skips_unsure():
    key = [
        {"id": "a", "experiment": "total", "stratum": "A"},
        {"id": "b", "experiment": "total", "stratum": "B"},
        {"id": "c", "experiment": "total", "stratum": "C"},
        {"id": "d", "experiment": "total", "stratum": "A"},
    ]
    sample = [{"id": i, "sentence": f"s-{i}"} for i in "abcd"]
    labels = [
        {"key": "a", "label": "yes"},
        {"key": "b", "label": "yes"},  # a miss
        {"key": "c", "label": "no"},
        {"key": "d", "label": "unsure"},
    ]
    res = analyze(labels, key, sample, {"total": {"A": 2, "B": 10, "C": 100}})
    assert (res["n_labeled"], res["n_unsure"]) == (4, 1)
    exp = res["by_experiment"]["total"]
    assert exp["precision"] == 1.0
    assert exp["recall"] == pytest.approx(2 / (2 + 10))
    assert res["misses"] == ["s-b"] and res["false_positives"] == []


def test_wilson_interval_contains_estimate():
    lo, hi = wilson(9, 10)
    assert lo < 0.9 < hi and lo >= 0 and hi <= 1


def test_draw_is_nested_so_labels_survive_a_larger_sample():
    rows = [{"id": f"{st}{i}", "stratum": st} for st in "ABC" for i in range(40)]
    small = draw(rows, parse_sizes("A=5,B=5,C=2"), seed=0)
    large = draw(rows, parse_sizes("A=15,B=15,C=10"), seed=0)
    assert {r["id"] for r in small} <= {r["id"] for r in large}
    assert len(small) == 12 and len(large) == 40
