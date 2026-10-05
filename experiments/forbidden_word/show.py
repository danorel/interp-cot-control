"""Print one problem across all conditions: instruction, text metrics, lens, and the CoT.

uv run python -m experiments.forbidden_word.show outputs/forbidden_word/<stamp>
uv run python -m experiments.forbidden_word.show <run_dir> --problem openai/gsm8k:465 --cot-chars 2000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from experiments.forbidden_word.data import Record, load_records
from experiments.forbidden_word.params import Params
from experiments.forbidden_word.text import split_cot

COLOR = sys.stdout.isatty()
RULE = "─" * 100


def _style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def highlight(text: str, target: re.Pattern[str], other: re.Pattern[str]) -> str:
    text = target.sub(lambda m: _style(f"[{m.group()}]", "1;31"), text)  # bold red
    return other.sub(lambda m: _style(f"<{m.group()}>", "1;33"), text)  # bold yellow


def lens_line(r: Record) -> str:
    """Target log-prob at 5 evenly spaced layers (first ... last)."""
    if r.lens is None:
        return "lens: —"
    n = len(r.lens.layers)
    picks = sorted({round(i * (n - 1) / 4) for i in range(5)})
    cells = [f"L{r.lens.layers[j]}={r.lens.target_logprob[j]:.2f}" for j in picks]
    return f"lens log P(target) on {r.lens.n_positions} CoT positions: " + "  ".join(cells)


def show_condition(r: Record, params: Params, cot_chars: int | None) -> None:
    t = r.text
    instruction = params.conditions[r.sample.condition].format(
        target=params.target.word, other=params.other.word
    )
    print(_style(f"\n{RULE}\n[{r.sample.condition}]", "1;36"))
    print(f"instruction: {instruction or '(none)'}")
    print(
        f"{params.target.word}: task={t.target.task} meta={t.target.meta} | "
        f"{params.other.word}: task={t.other.task} meta={t.other.meta} | "
        f"instruction_refs={t.instruction_refs} monitor={t.monitor_mentions} | "
        f"pred={t.pred} correct={t.correct} truncated={t.truncated}"
    )
    for s in t.meta_sentences:
        print(f"  meta> {s}")
    print(lens_line(r))

    cot = split_cot(r.sample.completion)
    body = cot.thinking if cot_chars is None else cot.thinking[:cot_chars]
    print(_style("CoT:", "2"))
    print(highlight(body, params.target.pattern, params.other.pattern))
    if cot_chars is not None and len(cot.thinking) > cot_chars:
        print(_style(f"... [{len(cot.thinking) - cot_chars} more chars]", "2"))
    print(_style("Answer:", "2"), cot.answer or "(none: truncated)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--problem", help="problem id (default: first in the run)")
    ap.add_argument("--cot-chars", type=int, default=None, help="truncate each CoT (default: full)")
    args = ap.parse_args(argv)

    params = Params.model_validate(
        yaml.safe_load((args.run_dir / "config.yaml").read_text())["params"]
    )
    records = load_records(args.run_dir / "rows.jsonl")
    pid = args.problem or records[0].sample.problem.id
    by_condition = {r.sample.condition: r for r in records if r.sample.problem.id == pid}
    if not by_condition:
        raise SystemExit(f"no records for problem {pid!r}")

    problem = next(iter(by_condition.values())).sample.problem
    print(_style(f"{problem.id}  gold={problem.gold}", "1"))
    print(problem.question)
    for condition in params.conditions:
        if condition in by_condition:
            show_condition(by_condition[condition], params, args.cot_chars)


if __name__ == "__main__":
    main()
