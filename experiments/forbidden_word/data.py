"""Records flowing through the experiment: Problem -> Sample (generated) -> Record (scored)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.forbidden_word.lens import LensScore
from experiments.forbidden_word.params import DatasetSpec
from experiments.forbidden_word.text import TextScore, WordUse, parse_gold
from interptemp.store import read_jsonl


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    gold: str | float | None  # reference answer (float in runs made before MATH support)
    word_in_question: bool  # the prompt itself primes the word; stratify on this later


@dataclass(frozen=True)
class Sample:
    problem: Problem
    condition: str
    prompt: str  # chat-formatted, exactly as fed to the model
    completion: str

    @property
    def key(self) -> tuple[str, str]:
        return self.problem.id, self.condition

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Sample:
        return cls(**{**d, "problem": Problem(**d["problem"])})


@dataclass(frozen=True)
class Record:
    sample: Sample
    text: TextScore
    lens: LensScore | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        text = d["text"]
        return cls(
            sample=Sample.from_dict(d["sample"]),
            text=TextScore(
                **{**text, "target": WordUse(**text["target"]), "other": WordUse(**text["other"])}
            ),
            lens=LensScore(**d["lens"]) if d["lens"] is not None else None,
        )


def load_problems(spec: DatasetSpec, word: re.Pattern[str], n: int, seed: int) -> list[Problem]:
    """Problems whose reference solution uses the word, i.e. it arises naturally."""
    from datasets import load_dataset

    names = spec.name if isinstance(spec.name, list) else [spec.name]
    rows = [r for name in names for r in load_dataset(spec.path, name, split=spec.split).to_list()]
    if spec.levels is not None:
        rows = [r for r in rows if r.get("level") in spec.levels]
    q, sol = spec.question_field, spec.solution_field
    pool = [
        Problem(
            id=f"{spec.path}:{i}",
            question=r[q],
            gold=parse_gold(r[sol], spec.answer_style),
            word_in_question=bool(word.search(r[q])),
        )
        for i, r in enumerate(rows)
        if word.search(r[sol])
    ]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def load_samples(path: str | Path) -> list[Sample]:
    return [Sample.from_dict(d) for d in read_jsonl(path)]


def load_records(path: str | Path) -> list[Record]:
    return [Record.from_dict(d) for d in read_jsonl(path)]
