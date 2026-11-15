"""Agent 4 — Validator.

Tests each candidate patch on an independent copy of the model:
1. Re-runs the failed questions — did the patch fix them?
2. Re-runs a sample of passing questions — did the patch break anything?

A patch is accepted only if it fixes enough failures while staying under
the regression budget. The live model is never mutated.
"""

from __future__ import annotations

import random
from typing import List, Sequence

from .model_adapter import ModelAdapter
from .types import PatchValidation, Question, WeightPatch


class Validator:
    def __init__(
        self,
        min_fix_rate: float = 0.5,
        max_regression_rate: float = 0.02,
        regression_sample_size: int = 100,
        seed: int = 42,
    ) -> None:
        self.min_fix_rate = min_fix_rate
        self.max_regression_rate = max_regression_rate
        self.regression_sample_size = regression_sample_size
        self.seed = seed

    def validate(
        self,
        model: ModelAdapter,
        patches: Sequence[WeightPatch],
        failed: Sequence[Question],
        passing: Sequence[Question],
    ) -> List[PatchValidation]:
        passing_sample = self._sample(passing)
        verdicts: List[PatchValidation] = []
        for patch in patches:
            candidate = model.copy()
            candidate.apply_scale(patch.layer, patch.neuron, patch.scale)
            fixed_rate = self._fix_rate(candidate, failed)
            regression_rate = self._regression_rate(model, candidate, passing_sample)
            accepted = (
                fixed_rate >= self.min_fix_rate
                and regression_rate <= self.max_regression_rate
            )
            notes: List[str] = []
            if fixed_rate < self.min_fix_rate:
                notes.append(
                    f"Fix rate {fixed_rate:.0%} below required {self.min_fix_rate:.0%}."
                )
            if regression_rate > self.max_regression_rate:
                notes.append(
                    f"Regression rate {regression_rate:.0%} above budget "
                    f"{self.max_regression_rate:.0%}."
                )
            verdicts.append(
                PatchValidation(
                    patch=patch,
                    failed_fixed_rate=fixed_rate,
                    regression_rate=regression_rate,
                    accepted=accepted,
                    notes=notes,
                )
            )
        verdicts.sort(
            key=lambda v: (-int(v.accepted), -v.failed_fixed_rate, v.regression_rate)
        )
        return verdicts

    def _sample(self, passing: Sequence[Question]) -> List[Question]:
        if len(passing) <= self.regression_sample_size:
            return list(passing)
        rng = random.Random(self.seed)
        return rng.sample(list(passing), self.regression_sample_size)

    def _fix_rate(self, candidate: ModelAdapter, failed: Sequence[Question]) -> float:
        if not failed:
            return 0.0
        fixed = sum(
            1
            for q in failed
            if self._answers_match(candidate.answer(q), q.expected_answer)
        )
        return fixed / len(failed)

    def _regression_rate(
        self,
        reference: ModelAdapter,
        candidate: ModelAdapter,
        passing: Sequence[Question],
    ) -> float:
        if not passing:
            return 0.0
        broken = 0
        for q in passing:
            was_ok = self._answers_match(reference.answer(q), q.expected_answer)
            is_ok = self._answers_match(candidate.answer(q), q.expected_answer)
            if was_ok and not is_ok:
                broken += 1
        return broken / len(passing)

    @staticmethod
    def _answers_match(model_answer: str, expected: str) -> bool:
        return model_answer.strip().lower() == expected.strip().lower()
