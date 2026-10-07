"""Validate the regex meta-sentence detector (text.is_meta) against blind hand labels.

    # 1. Stratified sample of CoT sentences from the banned conditions of each run
    uv run python -m experiments.forbidden_word.meta_validation sample \
        --run outputs/forbidden_word_total/20261007-112023 \
        --run outputs/forbidden_word_factor/20261007-155444 \
        --out reports/meta_validation

    # 2. Label blind (only context + sentence are shown, order shuffled, resumable)
    uv run interp-label label reports/meta_validation/meta_sample.jsonl --key id \
        --show context,sentence --choices "y=yes,n=no,x=unsure" \
        --question "Does the SENTENCE talk about the word or the rule (quote it, plan to avoid it, check compliance) rather than use it in the maths?"

    # 3. Compare with the detector
    uv run python -m experiments.forbidden_word.meta_validation analyze reports/meta_validation

Meta sentences are rare, so a random sample would contain few. Sentences are stratified:
    A  flagged by the detector                       -> precision
    B  not flagged but suspicious (word, "use", ...) -> misses where they are likely
    C  not flagged, the rest                         -> misses in the general stream
Recall is estimated by weighting each stratum's human "yes" rate by the stratum's size in the
full corpus (strata.json), not by its size in the sample.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from experiments.forbidden_word.data import load_samples
from experiments.forbidden_word.text import is_meta, sentences, split_cot, word_pattern
from interptemp.judges.metrics import agreement_report
from interptemp.store import read_jsonl, write_jsonl

BANNED_CONDITIONS = ("forbid", "grader")
STRATA = ("A", "B", "C")
# Words that tend to appear when talking about a word or a rule; used only to oversample
# likely misses (stratum B), never as a label.
_SUSPICIOUS = re.compile(
    r"\b(?:word|words|term|use|using|avoid\w*|mention\w*|say|saying|instead|rule|instruction\w*|"
    r"allowed|forbidden|banned|prohibit\w*)\b",
    re.IGNORECASE,
)
MAX_CHARS = 600


def stratum(sentence: str, target: re.Pattern[str], other: re.Pattern[str]) -> str:
    if is_meta(sentence, target) or is_meta(sentence, other):
        return "A"
    if target.search(sentence) or other.search(sentence) or _SUSPICIOUS.search(sentence):
        return "B"
    return "C"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " …"


def collect(run: Path) -> tuple[str, list[dict[str, Any]]]:
    """All sentences of the banned conditions' CoTs, with stratum; experiment = target word."""
    params = yaml.safe_load((run / "config.yaml").read_text())["params"]
    target, other = word_pattern(params["target"]["forms"]), word_pattern(params["other"]["forms"])
    experiment = params["target"]["word"]
    rows = []
    for s in load_samples(run / "generations.jsonl"):
        if s.condition not in BANNED_CONDITIONS:
            continue
        sents = sentences(split_cot(s.completion).thinking)
        for i, sent in enumerate(sents):
            rows.append(
                {
                    "id": f"{experiment}|{s.problem.id}|{s.condition}|{i}",
                    "experiment": experiment,
                    "stratum": stratum(sent, target, other),
                    "context": _clip(sents[i - 1]) if i else "(start of the reasoning)",
                    "sentence": _clip(sent),
                }
            )
    return experiment, rows


def draw(rows: Sequence[dict[str, Any]], per_stratum: int, seed: int) -> list[dict[str, Any]]:
    out = []
    for st in STRATA:
        pool = [r for r in rows if r["stratum"] == st]
        out += random.Random(f"{seed}-{st}").sample(pool, min(per_stratum, len(pool)))
    return out


def cmd_sample(runs: Sequence[Path], out: Path, per_stratum: int, seed: int) -> None:
    sample, key, sizes = [], [], {}
    for run in runs:
        experiment, rows = collect(run)
        sizes[experiment] = dict(Counter(r["stratum"] for r in rows))
        for r in draw(rows, per_stratum, seed):
            sample.append({"id": r["id"], "context": r["context"], "sentence": r["sentence"]})
            key.append({k: r[k] for k in ("id", "experiment", "stratum")})
        print(f"{experiment}: corpus strata sizes {sizes[experiment]}")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "meta_sample.jsonl", sample)  # what the labeler sees
    write_jsonl(out / "meta_sample_key.jsonl", key)  # hidden: stratum, i.e. the detector's call
    (out / "strata.json").write_text(json.dumps(sizes, indent=2))
    print(f"{len(sample)} sentences -> {out / 'meta_sample.jsonl'}")


# ---- analysis -------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def estimate(yes_rate: Mapping[str, float], sizes: Mapping[str, int]) -> dict[str, float]:
    """Corpus-level precision and recall from per-stratum human 'yes' rates and stratum sizes.

    The detector flags exactly stratum A, so precision = yes rate in A, and
    recall = (true metas in A) / (true metas in A + B + C), each scaled to the corpus.
    """
    true_meta = {st: yes_rate.get(st, 0.0) * sizes.get(st, 0) for st in STRATA}
    total = sum(true_meta.values())
    return {
        "precision": yes_rate.get("A", float("nan")),
        "recall": true_meta["A"] / total if total else float("nan"),
        "est_true_meta_sentences": total,
        "flagged_sentences": sizes.get("A", 0),
    }


def analyze(
    labels: Iterable[dict[str, Any]],
    key: Iterable[dict[str, Any]],
    sample: Iterable[dict[str, Any]],
    sizes: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    by_id = {k["id"]: k for k in key}
    text = {s["id"]: s["sentence"] for s in sample}
    joined = [{**by_id[lab["key"]], "human": lab["label"]} for lab in labels if lab["key"] in by_id]
    decided = [r for r in joined if r["human"] in ("yes", "no")]
    result: dict[str, Any] = {
        "n_labeled": len(joined),
        "n_unsure": sum(r["human"] == "unsure" for r in joined),
        "by_experiment": {},
    }
    for experiment in sorted({r["experiment"] for r in decided}):
        rows = [r for r in decided if r["experiment"] == experiment]
        per = {}
        for st in STRATA:
            in_st = [r for r in rows if r["stratum"] == st]
            k = sum(r["human"] == "yes" for r in in_st)
            per[st] = {"n": len(in_st), "human_yes": k, "rate": k / len(in_st) if in_st else 0.0,
                       "rate_ci95": wilson(k, len(in_st))}  # fmt: skip
        result["by_experiment"][experiment] = {
            "strata": per,
            **estimate({st: v["rate"] for st, v in per.items()}, sizes[experiment]),
        }
    detector = ["yes" if r["stratum"] == "A" else "no" for r in decided]
    result["sample_agreement"] = agreement_report(detector, [r["human"] for r in decided])
    result["false_positives"] = [
        text[r["id"]] for r in decided if r["stratum"] == "A" and r["human"] == "no"
    ]
    result["misses"] = [
        text[r["id"]] for r in decided if r["stratum"] != "A" and r["human"] == "yes"
    ]
    return result


def cmd_analyze(folder: Path) -> None:
    res = analyze(
        read_jsonl(folder / "labels.jsonl"),
        read_jsonl(folder / "meta_sample_key.jsonl"),
        read_jsonl(folder / "meta_sample.jsonl"),
        json.loads((folder / "strata.json").read_text()),
    )
    (folder / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"labeled {res['n_labeled']} (unsure: {res['n_unsure']})")
    for experiment, r in res["by_experiment"].items():
        rates = ", ".join(f"{st}: {v['human_yes']}/{v['n']}" for st, v in r["strata"].items())
        print(f"\n[{experiment}] human 'yes' per stratum: {rates}")
        print(f"  precision {r['precision']:.2f}   recall (corpus-weighted) {r['recall']:.2f}   "
              f"flagged {r['flagged_sentences']} vs est. true {r['est_true_meta_sentences']:.0f}")  # fmt: skip
    agr = res["sample_agreement"]
    print(f"\nsample: accuracy {agr['accuracy']:.2f}, kappa {agr['kappa']:.2f}, {agr['confusion']}")
    for name in ("false_positives", "misses"):
        print(f"\n{name} ({len(res[name])}):")
        for s in res[name][:10]:
            print(f"  - {s[:200]}")
    print(f"\n-> {folder / 'results.json'}")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", type=Path, action="append", required=True, help="generation run dir")
    s.add_argument("--out", type=Path, default=Path("reports/meta_validation"))
    s.add_argument("--per-stratum", type=int, default=25, help="sentences per stratum per run")
    s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("folder", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        cmd_sample(args.run, args.out, args.per_stratum, args.seed)
    else:
        cmd_analyze(args.folder)


if __name__ == "__main__":
    main()
