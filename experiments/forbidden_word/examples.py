"""Random CoT excerpts from the banned conditions, as Markdown, for the one-pager.

    uv run python -m experiments.forbidden_word.examples outputs/forbidden_word_total/<run> --n 3

CoTs are drawn at random (fixed seed) from all banned-condition CoTs, not picked by content.
For each, the first meta sentence and the first use of the word outside meta sentences are
shown with the sentence before them; the word is in bold, meta sentences are in italics.
"""

from __future__ import annotations

import argparse
import random
import re
from collections.abc import Sequence
from pathlib import Path

import yaml

from experiments.forbidden_word.data import Sample, load_samples
from experiments.forbidden_word.text import is_meta, sentences, split_cot, word_pattern

BANNED_CONDITIONS = ("forbid", "grader")
MAX_CHARS = 300


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " …"


def _bold(text: str, word: re.Pattern[str]) -> str:
    return word.sub(lambda m: f"**{m.group()}**", text)


def excerpt(sample: Sample, target: re.Pattern[str], other: re.Pattern[str]) -> str:
    sents = sentences(split_cot(sample.completion).thinking)
    meta = [is_meta(s, target) or is_meta(s, other) for s in sents]
    uses = sum(len(target.findall(s)) for s, m in zip(sents, meta, strict=True) if not m)
    lines = [
        f"**`{sample.condition}`**, {sample.problem.id}: {uses} use(s) of the word outside meta "
        f"sentences, {sum(meta)} meta sentence(s).",
        "",
    ]

    def show(i: int, kind: str) -> None:
        before = _clip(sents[i - 1]) if i else "(start of the reasoning)"
        body = _bold(_clip(sents[i]), target)
        lines.append(f"- {kind}: {before} → {'*' + body + '*' if meta[i] else body}")

    first_meta = next((i for i, m in enumerate(meta) if m), None)
    first_use = next((i for i, s in enumerate(sents) if not meta[i] and target.search(s)), None)
    if first_meta is not None:
        show(first_meta, "first meta sentence")
    else:
        lines.append("- no meta sentence")
    if first_use is not None:
        show(first_use, "first use of the word")
    else:
        lines.append("- the word is never used: a fully clean CoT")
    return "\n".join(lines)


def examples(run: Path, n: int, seed: int) -> str:
    params = yaml.safe_load((run / "config.yaml").read_text())["params"]
    target, other = word_pattern(params["target"]["forms"]), word_pattern(params["other"]["forms"])
    pool = [s for s in load_samples(run / "generations.jsonl") if s.condition in BANNED_CONDITIONS]
    picked = random.Random(seed).sample(pool, n)
    return "\n\n".join(excerpt(s, target, other) for s in picked)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path, help="generation run directory")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print(examples(args.run, args.n, args.seed))


if __name__ == "__main__":
    main()
