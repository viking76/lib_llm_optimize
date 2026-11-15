"""Statistical patch optimizer.

Improves on the fixed-alpha Weight Editor in four ways:

1. **Statistical candidate ranking** — candidates come from the
   :class:`~llm_optimize.stats.StatisticalRanker` (Cliff's delta x
   specificity) instead of raw mean ratios, so neurons that also fire on
   passing questions are demoted before any expensive search.
2. **Scale line search with refinement** — the edit magnitude is not guessed
   a priori: a coarse grid of scales is evaluated on a model copy, then
   iteratively refined (zoom) around the best scale, under the objective
   ``J = fix_rate - lambda * regression_rate - mu * |1 - scale|``:
   repairs are rewarded, regressions are punished heavily, and the least
   invasive edit wins at equal efficiency.
3. **Rank-one calibration (ROME-flavored)** — when per-neuron scaling cannot
   repair a failure (e.g. a joint misconfiguration of several neurons), the
   optimizer proposes a rank-one layer edit
   ``dW = -gamma * (mean_out_failed - mean_out_baseline) (k*)^T / (k*^T k*)``
   which moves the layer's response to the failure key ``k*`` (mean layer
   input over failed questions) toward its baseline level, in a single
   minimal-norm update.
4. **Greedy composition** — candidates of both kinds are accumulated into a
   combined edit; an addition is kept only if it strictly improves the
   objective of the whole edit. The final combined edit is re-validated end
   to end.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model_adapter import ModelAdapter
from .types import (
    CandidateEval,
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
        refinement_rounds: int = 2,
        refinement_factors: Sequence[float] = (0.9, 0.95, 1.05, 1.1),
        rank_one_gammas: Sequence[float] = (1.0, 0.7, 0.5),
        max_rank_one_layers: int = 3,
        baseline_sample_size: int = 50,
        lambda_regression: float = 5.0,
        mu_invasiveness: float = 0.05,
        rho_rank_one: float = 0.2,
        max_candidates: int = 10,
        epsilon: float = 1e-6,
    ) -> None:
        self.validator = validator or Validator()
        self.dampen_scales = list(dampen_scales)
        self.boost_scales = list(boost_scales)
        self.refinement_rounds = refinement_rounds
        self.refinement_factors = list(refinement_factors)
        self.rank_one_gammas = list(rank_one_gammas)
        self.max_rank_one_layers = max_rank_one_layers
        self.baseline_sample_size = baseline_sample_size
        self.lambda_regression = lambda_regression
        self.mu_invasiveness = mu_invasiveness
        self.rho_rank_one = rho_rank_one
        self.max_candidates = max_candidates
        self.epsilon = epsilon

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
        rank_one = self._search_rank_one(model, evidence, failed, passing, sample)
        rank_one.sort(key=lambda c: (-c.objective, abs(1.0 - c.patch.scale)))
        patches = self._compose(model, searched, rank_one, failed, sample)
        return self._final_validation(
            model, searched, rank_one, patches, failed, sample
        )

    # ------------------------------------------------------------------ scales

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
                evaluation = self._evaluate_scale(model, ev, scale, failed, sample)
                if evaluation is None:
                    continue
                if best is None or evaluation.objective > best.objective:
                    best = evaluation
            if best is None:
                continue
            for _ in range(self.refinement_rounds):
                improved: Optional[ScaleSearchResult] = None
                for factor in self._zoom_factors(best.best_scale):
                    scale = best.best_scale * factor
                    evaluation = self._evaluate_scale(model, ev, scale, failed, sample)
                    if evaluation is None:
                        continue
                    if (
                        improved is None
                        or evaluation.objective > improved.objective
                        or (
                            evaluation.objective == improved.objective
                            and abs(1.0 - scale) < abs(1.0 - improved.best_scale)
                        )
                    ):
                        improved = evaluation
                if improved is None or improved.objective < best.objective:
                    break
                best = improved
            if best.objective > 0.0:
                results.append(best)
        return results

    def _zoom_factors(self, best_scale: float) -> List[float]:
        factors = []
        for factor in self.refinement_factors:
            candidate = best_scale * factor
            same_side = (candidate < 1.0) == (best_scale < 1.0)
            if same_side and candidate > 0.0:
                factors.append(factor)
        return factors

    def _evaluate_scale(
        self,
        model: ModelAdapter,
        ev: NeuronEvidence,
        scale: float,
        failed: Sequence[Question],
        sample: Sequence[Question],
    ) -> Optional[ScaleSearchResult]:
        candidate = model.copy()
        candidate.apply_scale(ev.layer, ev.neuron, scale)
        fix = self.validator.measure_fix_rate(candidate, failed)
        if fix <= 0.0:
            return None
        reg = self.validator.measure_regression_rate(model, candidate, sample)
        objective = self._objective(fix, reg, abs(1.0 - scale), n_rank_one=0)
        patch = WeightPatch(
            patch_id="",
            kind=PatchKind.DAMPEN.value if scale < 1.0 else PatchKind.BOOST.value,
            layer=ev.layer,
            neuron=ev.neuron,
            scale=scale,
            evidence=ev,
        )
        return ScaleSearchResult(
            evidence=ev,
            direction=ev.direction,
            best_scale=scale,
            objective=objective,
            fix_rate=fix,
            regression_rate=reg,
            patch=patch,
        )

    # ---------------------------------------------------------------- rank one

    def _search_rank_one(
        self,
        model: ModelAdapter,
        evidence: Sequence[NeuronEvidence],
        failed: Sequence[Question],
        passing: Sequence[Question],
        sample: Sequence[Question],
    ) -> List[CandidateEval]:
        if not evidence or not passing or not failed:
            return []
        failed_texts = [q.text for q in failed]
        baseline_texts = [q.text for q in passing[: self.baseline_sample_size]]
        keys = model.layer_inputs(failed_texts)
        out_failed = model.per_question_activations(failed_texts)
        out_baseline = model.per_question_activations(baseline_texts)

        candidates: List[CandidateEval] = []
        seen_layers: set = set()
        for ev in evidence:
            if len(seen_layers) >= self.max_rank_one_layers:
                break
            layer = ev.layer
            if layer in seen_layers or layer not in keys:
                continue
            seen_layers.add(layer)
            best = self._evaluate_rank_one_layer(
                model,
                ev,
                layer,
                keys[layer],
                out_failed[layer],
                out_baseline.get(layer),
                failed,
                sample,
            )
            if best is not None:
                candidates.append(best)
        return candidates

    def _evaluate_rank_one_layer(
        self,
        model: ModelAdapter,
        ev: NeuronEvidence,
        layer: int,
        keys: np.ndarray,
        out_failed: np.ndarray,
        out_baseline: Optional[np.ndarray],
        failed: Sequence[Question],
        sample: Sequence[Question],
    ) -> Optional[CandidateEval]:
        if out_baseline is None or keys.size == 0:
            return None
        key = keys.mean(axis=0)
        kk = float(key @ key)
        if kk < self.epsilon:
            return None
        excess = out_failed.mean(axis=0) - out_baseline.mean(axis=0)
        if float(excess @ excess) < self.epsilon:
            return None
        best: Optional[CandidateEval] = None
        for gamma in self.rank_one_gammas:
            delta = -gamma * excess / kk
            candidate = model.copy()
            candidate.apply_rank_one(layer, key, delta)
            fix = self.validator.measure_fix_rate(candidate, failed)
            if fix <= 0.0:
                continue
            reg = self.validator.measure_regression_rate(model, candidate, sample)
            objective = self._objective(fix, reg, abs(1.0 - gamma), n_rank_one=1)
            if best is None or objective > best.objective:
                patch = WeightPatch(
                    patch_id="",
                    kind=PatchKind.RANK_ONE.value,
                    layer=layer,
                    neuron=-1,
                    scale=gamma,
                    evidence=ev,
                    key_vector=key,
                    delta_vector=delta,
                )
                best = CandidateEval(
                    patch=patch,
                    objective=objective,
                    fix_rate=fix,
                    regression_rate=reg,
                )
        return best

    # -------------------------------------------------------------- composition

    def _compose(
        self,
        model: ModelAdapter,
        searched: Sequence[ScaleSearchResult],
        rank_one: Sequence[CandidateEval],
        failed: Sequence[Question],
        sample: Sequence[Question],
    ) -> List[WeightPatch]:
        candidates: List[Tuple[float, int, WeightPatch]] = [
            (r.objective, 0, r.patch) for r in searched
        ] + [(c.objective, 1, c.patch) for c in rank_one]
        candidates.sort(key=lambda item: (-item[0], item[1]))

        patches: List[WeightPatch] = []
        base = model.copy()
        base_objective = 0.0
        for _, _, patch in candidates:
            trial = base.copy()
            self._apply(trial, patch)
            fix = self.validator.measure_fix_rate(trial, failed)
            reg = self.validator.measure_regression_rate(model, trial, sample)
            invasiveness, n_rank_one = self._edit_cost(patches + [patch])
            trial_objective = self._objective(fix, reg, invasiveness, n_rank_one)
            if trial_objective > base_objective + 1e-9:
                patches.append(patch)
                base = trial
                base_objective = trial_objective
        for idx, patch in enumerate(patches):
            patch.patch_id = f"opt-{idx:03d}"
        return patches

    def _final_validation(
        self,
        model: ModelAdapter,
        searched: Sequence[ScaleSearchResult],
        rank_one: Sequence[CandidateEval],
        patches: List[WeightPatch],
        failed: Sequence[Question],
        sample: Sequence[Question],
    ) -> OptimizationResult:
        candidate = model.copy()
        for patch in patches:
            self._apply(candidate, patch)
        fix = self.validator.measure_fix_rate(candidate, failed)
        reg = self.validator.measure_regression_rate(model, candidate, sample)
        invasiveness, n_rank_one = self._edit_cost(patches)
        objective = (
            self._objective(fix, reg, invasiveness, n_rank_one) if patches else 0.0
        )
        accepted = bool(patches) and (
            fix >= self.validator.min_fix_rate
            and reg <= self.validator.max_regression_rate
        )
        return OptimizationResult(
            searched=list(searched),
            rank_one_candidates=list(rank_one),
            combined_patches=patches,
            combined_fix_rate=fix,
            combined_regression_rate=reg,
            combined_accepted=accepted,
            objective=objective,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _apply(model: ModelAdapter, patch: WeightPatch) -> None:
        if patch.kind == PatchKind.RANK_ONE.value:
            model.apply_rank_one(patch.layer, patch.key_vector, patch.delta_vector)
        else:
            model.apply_scale(patch.layer, patch.neuron, patch.scale)

    @staticmethod
    def _edit_cost(patches: Sequence[WeightPatch]) -> Tuple[float, int]:
        invasiveness = sum(abs(1.0 - p.scale) for p in patches)
        n_rank_one = sum(1 for p in patches if p.kind == PatchKind.RANK_ONE.value)
        return invasiveness, n_rank_one

    def _objective(
        self,
        fix_rate: float,
        regression_rate: float,
        invasiveness: float,
        n_rank_one: int,
    ) -> float:
        return (
            fix_rate
            - self.lambda_regression * regression_rate
            - self.mu_invasiveness * invasiveness
            - self.rho_rank_one * n_rank_one
        )
