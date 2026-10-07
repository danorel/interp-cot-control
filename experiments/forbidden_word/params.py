"""Typed schema for the experiment's `params:` block (validated once at the start of a run)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

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
    """A HF dataset; defaults match GSM8K."""

    path: str
    name: str | list[str] | None = None  # several configs are concatenated (MATH subjects)
    split: str = "test"
    question_field: str = "question"
    solution_field: str = "answer"
    answer_style: Literal["gsm8k", "boxed"] = "gsm8k"  # where the reference answer sits
    levels: list[str] | None = None  # keep only rows whose `level` is listed (MATH)


class LensParams(_Strict):
    enabled: bool = True
    layers: list[int] | None = None  # None = every layer
    max_positions: int | None = 128
    topk: int = 10
    # Read off only word variants that are one token. A split form's first token can be a
    # generic prefix ("factored" -> " fact" would also match "in fact").
    single_token_words: bool = False


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


class SwapParams(_Strict):
    """Prompt swap: re-read each CoT under every condition's prompt (prompts are taken from
    the generations file, so they are exactly the ones used for sampling)."""

    generations: Path  # generations.jsonl of a finished run
    n_problems: int | None = None  # first N problems of the file; None = all
    target: WordSpec
    other: WordSpec
    cot_conditions: list[str]  # whose CoTs to re-read
    prompt_conditions: list[str]  # prompts to re-read them under
    substitutes: list[str]  # words the model may write instead of the target
    max_class_positions: int = 64  # per CoT, for the rare target/substitute position classes
    n_bins: int = 5  # bins of relative position within the CoT
    lens: LensParams = Field(default_factory=LensParams)
