"""Does forbidding a word in the CoT remove it from the residual stream, or only from the text?

Conditions (same problems and sampling settings; only the instruction differs):
    baseline  no instruction                 control   word mentioned neutrally
    forbid    "don't use <target>"           random    "don't use <other>" (similar word)
    grader    forbid + "a grader monitors your reasoning"

Stage 1 generates CoTs (nnterp, or vLLM on a pod) and scores them as text (text.py).
Stage 2 (nnterp only) applies the logit lens to the CoT residual stream (lens.py).
On a pod: run with the vLLM model, then rerun with the nnterp model and
`params.generations_from=<run>/generations.jsonl`. The same param resumes a partial run.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, replace
from typing import Any

from tqdm import tqdm

from experiments.forbidden_word.data import Problem, Record, Sample, load_problems, load_samples
from experiments.forbidden_word.lens import CoTLens
from experiments.forbidden_word.params import Params
from experiments.forbidden_word.stats import summarize
from experiments.forbidden_word.text import score_text
from interptemp.experiment import Experiment
from interptemp.models.base import InterpModel
from interptemp.models.nnterp_model import NnterpModel
from interptemp.store import write_jsonl
from interptemp.utils import batched

PROGRESS_INTERVAL_S = 10  # tqdm refresh; keeps tee'd log files readable


class ForbiddenWordExperiment(Experiment):
    def run(self) -> dict[str, Any]:
        params = Params.model_validate(self.params)
        self._require_thinking()
        problems = load_problems(
            params.dataset, params.target.pattern, params.n_problems, self.cfg.seed
        )
        self.log.info(f"{len(problems)} problems x {len(params.conditions)} conditions")

        samples = self.generate(problems, params)
        records = [self.score_text(s, params) for s in samples]
        if params.lens.enabled:
            records = self.add_lens(records, params)

        self.save_jsonl("rows.jsonl", [asdict(r) for r in records])
        return summarize(records)

    def _require_thinking(self) -> None:
        if self.cfg.model.chat_template_kwargs.get("enable_thinking", True) is not True:
            raise ValueError("set model.chat_template_kwargs.enable_thinking=true")

    # ---- stage 1 ------------------------------------------------------------------------

    def generate(self, problems: list[Problem], params: Params) -> list[Sample]:
        """Generate every (problem, condition) not already in `generations_from`.

        Saves after each batch so a crashed run can be resumed from its generations.jsonl.
        """
        out_path = self.path("generations.jsonl")
        reused = load_samples(params.generations_from) if params.generations_from else []
        write_jsonl(out_path, (asdict(s) for s in reused))

        done = {s.key for s in reused}
        todo = [(p, c) for c in params.conditions for p in problems if (p.id, c) not in done]
        self.log.info(f"generation: {len(reused)} reused, {len(todo)} to generate")

        # vLLM reloads the model per call, so give it everything at once.
        is_local = isinstance(self.model, InterpModel)
        batch_size = self.cfg.generation.batch_size if is_local else max(len(todo), 1)
        fresh: list[Sample] = []
        with tqdm(total=len(todo), desc="generate", mininterval=PROGRESS_INTERVAL_S) as bar:
            for batch in batched(todo, batch_size):
                prompts = [
                    self.model.format_chat(
                        [{"role": "user", "content": params.user_message(p.question, c)}]
                    )
                    for p, c in batch
                ]
                completions = self.model.generate(prompts, self.cfg.generation)
                new = [
                    Sample(problem=p, condition=c, prompt=prompt, completion=completion)
                    for (p, c), prompt, completion in zip(batch, prompts, completions, strict=True)
                ]
                write_jsonl(out_path, (asdict(s) for s in new), append=True)
                fresh += new
                bar.update(len(batch))

        wanted = {(p.id, c) for p in problems for c in params.conditions}
        return [s for s in reused + fresh if s.key in wanted]

    @staticmethod
    def score_text(sample: Sample, params: Params) -> Record:
        text = score_text(
            sample.completion, sample.problem.gold, params.target.pattern, params.other.pattern
        )
        return Record(sample=sample, text=text)

    # ---- stage 2 ------------------------------------------------------------------------

    def add_lens(self, records: list[Record], params: Params) -> list[Record]:
        if not isinstance(self.model, NnterpModel):
            self.log.warning("lens needs the nnterp backend; skipping (rerun with nnterp model)")
            return records

        lens = CoTLens(
            self.model,
            target_forms=params.target.forms,
            other_forms=params.other.forms,
            layers=params.lens.layers,
            max_positions=params.lens.max_positions,
            topk=params.lens.topk,
        )
        scored = []
        for r in tqdm(records, desc="lens", mininterval=PROGRESS_INTERVAL_S):
            # Seed from the sample key, so position subsampling doesn't depend on row order.
            seed = zlib.crc32("/".join(r.sample.key).encode())
            scored.append(replace(r, lens=lens.score(r.sample.prompt, r.sample.completion, seed)))

        n_missing = sum(r.lens is None for r in scored)
        if n_missing:
            self.log.warning(
                f"lens: {n_missing}/{len(scored)} samples unscored (misaligned/empty CoT)"
            )
        return scored
