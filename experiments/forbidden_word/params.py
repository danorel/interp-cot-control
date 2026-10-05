"""Typed schema for the experiment's `params:` block (validated once at the start of a run)."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from experiments.forbidden_word.text import word_pattern


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WordSpec(_Strict):
    word: str  # as written in the instruction
    forms: list[str]  # inflections counted as a use of the word

    @property
    def pattern(self) -> re.Pattern[str]:
        return word_pattern(self.forms)


class DatasetSpec(_Strict):
    path: str
    name: str | None = None
    split: str = "test"


class LensParams(_Strict):
    enabled: bool = True
    layers: list[int] | None = None  # None = every layer
    max_positions: int | None = 128
    topk: int = 10


class Params(_Strict):
    n_problems: int
    dataset: DatasetSpec
    target: WordSpec
    other: WordSpec  # the word forbidden in the "random" condition
    answer_format: str
    conditions: dict[str, str]  # name -> instruction template with {target} / {other}
    generations_from: Path | None = None
    lens: LensParams = Field(default_factory=LensParams)

    def user_message(self, question: str, condition: str) -> str:
        instruction = self.conditions[condition].format(
            target=self.target.word, other=self.other.word
        )
        return "\n\n".join(part for part in (question, self.answer_format, instruction) if part)
