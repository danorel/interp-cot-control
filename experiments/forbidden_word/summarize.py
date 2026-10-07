"""Recompute a run's summary.json from its rows.jsonl — after changing metrics, no model needed.

uv run python -m experiments.forbidden_word.summarize outputs/forbidden_word_total/<stamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.forbidden_word.data import load_records
from experiments.forbidden_word.stats import summarize


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)

    records = load_records(args.run_dir / "rows.jsonl")
    out = args.run_dir / "summary.json"
    out.write_text(json.dumps(summarize(records), indent=2, default=str, ensure_ascii=False))
    print(f"{len(records)} records -> {out}")


if __name__ == "__main__":
    main()
