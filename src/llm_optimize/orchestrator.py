"""Pipeline orchestrator — runs the 4-agent team end to end.

Failure Miner -> Activation Tracer -> Weight Editor -> Validator -> Report.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .activation_tracer import ActivationTracer
from .failure_miner import FailureMiner
from .model_adapter import ModelAdapter
from .optimizer import GreedyPatchOptimizer
from .types import (
    EvalResult,
    NeuronEvidence,
    PatchValidation,
    Question,
    Report,
    WeightPatch,
)
from .validator import Validator
from .weight_editor import WeightEditor


class Orchestrator:
    def __init__(
        self,
        model: ModelAdapter,
        miner: Optional[FailureMiner] = None,
        tracer: Optional[ActivationTracer] = None,
        editor: Optional[WeightEditor] = None,
        validator: Optional[Validator] = None,
        optimizer: Optional[GreedyPatchOptimizer] = None,
    ) -> None:
        self.model = model
        self.miner = miner or FailureMiner()
        self.tracer = tracer or ActivationTracer(model)
        self.editor = editor or WeightEditor()
        self.validator = validator or Validator()
        self.optimizer = optimizer

    def run(
        self,
        questions: Sequence[Question],
        results: Sequence[EvalResult],
        top_k_neurons: int = 20,
        max_patches: Optional[int] = None,
        trace_method: str = "ratio",
    ) -> Report:
        if max_patches is not None:
            self.editor.max_patches = max_patches

        failed, categories, total = self.miner.mine(questions, results)
        passing = self._passing_questions(questions, results)

        evidence: List[NeuronEvidence] = self.tracer.trace(
            failed=failed,
            baseline=passing,
            top_k=top_k_neurons,
            method=trace_method,
        )

        optimization = None
        if self.optimizer is not None:
            optimization = self.optimizer.optimize(
                model=self.model,
                evidence=evidence,
                failed=failed,
                passing=passing,
            )
            validations = [
                PatchValidation(
                    patch=patch,
                    failed_fixed_rate=optimization.combined_fix_rate,
                    regression_rate=optimization.combined_regression_rate,
                    accepted=optimization.combined_accepted,
                    notes=["Validated as part of the combined optimized edit."],
                )
                for patch in optimization.combined_patches
            ]
            summary = self._summary_optimized(total, failed, optimization)
        else:
            patches: List[WeightPatch] = self.editor.propose(evidence, failed)
            validations = self.validator.validate(
                model=self.model,
                patches=patches,
                failed=failed,
                passing=passing,
            )
            accepted = [v for v in validations if v.accepted]
            summary = self._summary(total, failed, accepted)

        return Report(
            total_questions=total,
            failed_questions=len(failed),
            weakness_categories=categories,
            top_neurons=evidence,
            validations=validations,
            summary=summary,
            optimization=optimization,
        )

    @staticmethod
    def _passing_questions(
        questions: Sequence[Question],
        results: Sequence[EvalResult],
    ) -> List[Question]:
        by_id = {q.id: q for q in questions}
        return [
            by_id[r.question_id]
            for r in results
            if r.passed and r.question_id in by_id
        ]

    @staticmethod
    def _summary_optimized(
        total: int, failed: List[Question], optimization
    ) -> str:
        if not failed:
            return "The model passed every question; no weakness to analyze."
        status = "accepted" if optimization.combined_accepted else "rejected"
        lines = [
            f"{len(failed)}/{total} questions failed. Optimized combined edit "
            f"({len(optimization.combined_patches)} patch(s)) {status}: fixes "
            f"{optimization.combined_fix_rate:.0%} of failures with "
            f"{optimization.combined_regression_rate:.0%} regressions "
            f"(objective {optimization.objective:.3f})."
        ]
        for patch in optimization.combined_patches:
            lines.append(f"- {patch.patch_id}: {patch.describe()}")
        return "\n".join(lines)

    @staticmethod
    def _summary(
        total: int, failed: List[Question], accepted: List[PatchValidation]
    ) -> str:
        if total == 0:
            return "No evaluation results were provided."
        if not failed:
            return "The model passed every question; no weakness to analyze."
        lines = [
            f"{len(failed)}/{total} questions failed. "
            f"{len(accepted)} weight patch(es) accepted."
        ]
        for v in accepted:
            lines.append(
                f"- {v.patch.patch_id}: {v.patch.describe()} fixes "
                f"{v.failed_fixed_rate:.0%} of failures with "
                f"{v.regression_rate:.0%} regressions."
            )
        if not accepted:
            lines.append(
                "No patch met the validation criteria; inspect the top "
                "differential neurons before proposing stronger edits."
            )
        return "\n".join(lines)
