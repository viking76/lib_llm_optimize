"""Agent 2 — Activation Tracer.

Replays only the failed questions through the model, captures per-layer
activations, and compares them against a baseline built from a sample of
passing questions. The output is a ranked list of differentially active
neurons — the only parameters worth looking at.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .model_adapter import ModelAdapter
from .stats import StatisticalRanker
from .types import NeuronEvidence, Question


class ActivationTracer:
    def __init__(
        self,
        model: ModelAdapter,
        baseline_sample_size: int = 50,
        epsilon: float = 1e-6,
        ranker: StatisticalRanker | None = None,
    ) -> None:
        self.model = model
        self.baseline_sample_size = baseline_sample_size
        self.epsilon = epsilon
        self.ranker = ranker or StatisticalRanker(epsilon=epsilon)

    def trace(
        self,
        failed: Sequence[Question],
        baseline: Sequence[Question],
        top_k: int = 20,
        min_ratio: float = 1.5,
        method: str = "ratio",
    ) -> List[NeuronEvidence]:
        """Return the top_k neurons whose activation differs most between
        failed and baseline questions.

        ``method="ratio"`` ranks by mean-activation ratio (cheap).
        ``method="effect"`` ranks by Cliff's delta x specificity computed on
        per-question activations (statistically stronger; see stats.py).
        ``min_ratio`` filters out neurons whose differential is noise-level
        (ratio method only).
        """
        if not failed:
            return []
        if method == "effect":
            return self._trace_effect(failed, baseline, top_k)
        failed_texts = [q.text for q in failed]
        baseline_texts = [q.text for q in baseline[: self.baseline_sample_size]]
        if not baseline_texts:
            baseline_texts = failed_texts

        failed_acts = self.model.activations(failed_texts)
        baseline_acts = self.model.activations(baseline_texts)

        evidence: List[NeuronEvidence] = []
        for layer in sorted(failed_acts):
            f = failed_acts[layer]
            b = baseline_acts.get(layer)
            if b is None:
                continue
            mean_failed = f.mean(axis=0) if f.ndim > 1 else f
            mean_baseline = b.mean(axis=0) if b.ndim > 1 else b
            for neuron in range(mean_failed.shape[0]):
                fa = float(mean_failed[neuron])
                ba = float(mean_baseline[neuron])
                if max(fa, ba) < self.epsilon:
                    continue
                ratio = fa / (ba + self.epsilon)
                if ratio >= min_ratio or ratio <= (1.0 / max(min_ratio, 1.0)):
                    evidence.append(
                        NeuronEvidence(
                            layer=layer,
                            neuron=neuron,
                            mean_activation_failed=fa,
                            mean_activation_baseline=ba,
                            ratio=ratio,
                        )
                    )
        evidence.sort(key=lambda e: -abs(np.log(e.ratio + self.epsilon)))
        return evidence[:top_k]

    def _trace_effect(
        self,
        failed: Sequence[Question],
        baseline: Sequence[Question],
        top_k: int,
    ) -> List[NeuronEvidence]:
        failed_texts = [q.text for q in failed]
        baseline_texts = [q.text for q in baseline[: self.baseline_sample_size]]
        if not baseline_texts:
            baseline_texts = failed_texts
        failed_acts = self.model.per_question_activations(failed_texts)
        baseline_acts = self.model.per_question_activations(baseline_texts)
        return self.ranker.rank(failed_acts, baseline_acts, top_k=top_k)
