"""Statistical patch optimizer.

Improves on the fixed-alpha Weight Editor in three ways:

1. **Line search over scales** — for each candidate neuron, the edit
   magnitude is no longer guessed a priori: a grid of scales is evaluated
   on a model copy and the best one is kept under the objective
   ``J = fix_rate - lambda * regression_rate - mu * |1 - scale|``,
   which rewards repairs, punishes regressions heavily, and prefers the
   least invasive edit among equally good ones.
2. **Statistical candidate ranking** — candidates come from the
   :class:`~llm_optimize.stats.StatisticalRanker` (Cliff's delta x
   specificity) instead of raw mean ratios, so neurons that also fire on
   passing questions are demoted before any expensive search.
3. **Greedy composition** — candidates are accumulated into a combined
   edit, keeping an addition only if it improves the objective of the
   whole edit. This captures interactions between neurons and yields one
   final recommended weight edit, re-validated end to end.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .model_adapter import ModelAdapter
from .types import (
    NeuronEvidence,
    OptimizationResult,
    PatchKind,
    Question,
    ScaleSearchResult,
    WeightPatch,
)
from .validator import Validator


class GreedyPatchOptimizer:
    def __init__(
        self,
        validator: Optional[Validator] = None,
        dampen_scales: Sequence[float] = (0.9, 0.8, 0.7, 0.5, 0.3),
        boost_scales: Sequence[float] = (1.1, 1.2, 1.5, 2.0),
        lambda_regression: float = 5.0,
        mu_invasiveness: float = 0.05,
        max_candidates: int = 10,
    ) -> None:
        self.validator = validator or Validator()
        self.dampen_scales = list(dampen_scales)
        self.boost_scales = list(boost_scales)
        self.lambda_regression = lambda_regression
        self.mu_invasiveness = mu_invasiveness
        self.max_candidates = max_candidates

    def optimize(
        self,
        model: ModelAdapter,
        evidence: Sequence[NeuronEvidence],
        failed: Sequence[Question],
        passing: Sequence[Question],
    ) -> OptimizationResult:
        sample = self.validator.regression_sample(passing)
        searched = self._search_scales(model, evidence, failed, sample)
        searched.sort(key=lambda s: (-s.objective, abs(1.0 - s.best_scale)))
        patches = self._compose(model, searched, failed, sample)
        result = self._final_validation(model, searched, patches, failed, sample)
        categories = sorted({q.category for q in failed})
        for patch in result.combined_patches:
            patch.failed_categories = categories
        return result

    def _search_scales(
        self,
        model: ModelAdapter,
        evidence: Sequence[NeuronEvidence],
        failed: Sequence[Question],
        sample: Sequence[Question],
    ) -> List[ScaleSearchResult]:
        results: List[ScaleSearchResult] = []
        for ev in evidence[: self.max_candidates]:
            scales = self.dampen_scales if ev.direction == "over" else self.boost_scales
            best: Optional[ScaleSearchResult] = None
            for scale in scales:
                candidate = model.copy()
                candidate.apply_scale(ev.layer, ev.neuron, scale)
                fix = self.validator.measure_fix_rate(candidate, failed)
                if fix <= 0.0:
                    continue
                reg = self.validator.measure_regression_rate(model, candidate, sample)
                objective = self._objective(fix, reg, scale)
                if best is None or objective > best.objective:
                    best = ScaleSearchResult(
                        evidence=ev,
                        direction=ev.direction,
                        best_scale=scale,
                        objective=objective,
                        fix_rate=fix,
                        regression_rate=reg,
                    )
            if best is not None and best.objective > 0.0:
                results.append(best)
        return results

    def _compose(
        self,
        model: ModelAdapter,
        searched: Sequence[ScaleSearchResult],
        failed: Sequence[Question],
        sample: Sequence[Question],
    ) -> List[WeightPatch]:
        patches: List[WeightPatch] = []
        base = model.copy()
        base_objective = 0.0
        for result in searched:
            trial = base.copy()
            ev = result.evidence
            trial.apply_scale(ev.layer, ev.neuron, result.best_scale)
            fix = self.validator.measure_fix_rate(trial, failed)
            reg = self.validator.measure_regression_rate(model, trial, sample)
            invasiveness = sum(abs(1.0 - p.scale) for p in patches) + abs(
                1.0 - result.best_scale
            )
            objective = fix - self.lambda_regression * reg - self.mu_invasiveness * invasiveness
            if objective > base_objective + 1e-9:
                patches.append(
                    WeightPatch(
                        patch_id=f"opt-{len(patches):03d}",
                        kind=(
                            PatchKind.DAMPEN.value
                            if result.best_scale < 1.0
                            else PatchKind.BOOST.value
                        ),
                        layer=ev.layer,
                        neuron=ev.neuron,
                        scale=result.best_scale,
                        evidence=ev,
                    )
                )
                base = trial
                base_objective = objective
        return patches

    def _final_validation(
        self,
        model: ModelAdapter,
        searched: Sequence[ScaleSearchResult],
        patches: List[WeightPatch],
        failed: Sequence[Question],
        sample: Sequence[Question],
    ) -> OptimizationResult:
        candidate = model.copy()
        for patch in patches:
            candidate.apply_scale(patch.layer, patch.neuron, patch.scale)
        fix = self.validator.measure_fix_rate(candidate, failed)
        reg = self.validator.measure_regression_rate(model, candidate, sample)
        invasiveness = sum(abs(1.0 - p.scale) for p in patches)
        objective = (
            fix - self.lambda_regression * reg - self.mu_invasiveness * invasiveness
            if patches
            else 0.0
        )
        accepted = bool(patches) and (
            fix >= self.validator.min_fix_rate
            and reg <= self.validator.max_regression_rate
        )
        return OptimizationResult(
            searched=list(searched),
            combined_patches=patches,
            combined_fix_rate=fix,
            combined_regression_rate=reg,
            combined_accepted=accepted,
            objective=objective,
        )

    def _objective(self, fix_rate: float, regression_rate: float, scale: float) -> float:
        return (
            fix_rate
            - self.lambda_regression * regression_rate
            - self.mu_invasiveness * abs(1.0 - scale)
        )
