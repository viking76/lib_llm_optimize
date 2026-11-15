"""Statistical attribution of failures to neurons.

Replaces the raw mean-activation ratio with per-question statistics:

- **Cliff's delta** between failed and baseline per-question activations:
  a robust effect size in [-1, 1] (+1 means the neuron fires systematically
  higher on failures), insensitive to outliers and scale.
- **Specificity**: among the questions where the neuron is most active
  (top quartile for over-active, bottom quartile for under-active), the
  fraction that are failures. A neuron that also fires on passing questions
  is a weak repair target; specificity down-weights it.

Final score = |delta| x specificity, so ranking favors neurons that are both
strongly and exclusively linked to failures.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .types import NeuronEvidence


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta of sample ``x`` over sample ``y``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return 0.0
    gt = int((x[:, None] > y[None, :]).sum())
    lt = int((x[:, None] < y[None, :]).sum())
    return (gt - lt) / float(x.size * y.size)


class StatisticalRanker:
    def __init__(self, epsilon: float = 1e-6, tail_quantile: float = 0.75) -> None:
        self.epsilon = epsilon
        self.tail_quantile = tail_quantile

    def rank(
        self,
        failed_acts: Dict[int, np.ndarray],
        baseline_acts: Dict[int, np.ndarray],
        top_k: int = 20,
    ) -> List[NeuronEvidence]:
        """Rank neurons from per-question activation matrices.

        ``failed_acts`` / ``baseline_acts``: {layer: array[n_questions, n_neurons]}.
        """
        evidence: List[NeuronEvidence] = []
        for layer in sorted(failed_acts):
            F = np.atleast_2d(np.asarray(failed_acts[layer], dtype=np.float64))
            B = np.atleast_2d(np.asarray(baseline_acts.get(layer), dtype=np.float64))
            if F.size == 0 or B.size == 0 or F.shape[1] != B.shape[1]:
                continue
            n_neurons = F.shape[1]
            for neuron in range(n_neurons):
                xf, xb = F[:, neuron], B[:, neuron]
                if max(xf.max(), xb.max()) < self.epsilon:
                    continue
                delta, specificity = self._effect(xf, xb)
                if delta == 0.0 or specificity == 0.0:
                    continue
                mean_failed = float(xf.mean())
                mean_baseline = float(xb.mean())
                evidence.append(
                    NeuronEvidence(
                        layer=layer,
                        neuron=neuron,
                        mean_activation_failed=mean_failed,
                        mean_activation_baseline=mean_baseline,
                        ratio=mean_failed / (mean_baseline + self.epsilon),
                        effect_size=delta,
                        specificity=specificity,
                    )
                )
        evidence.sort(key=lambda e: (-abs(e.effect_size) * e.specificity, -abs(e.effect_size)))
        return evidence[:top_k]

    def _effect(self, xf: np.ndarray, xb: np.ndarray) -> Tuple[float, float]:
        delta = cliffs_delta(xf, xb)
        pooled = np.concatenate([xf, xb])
        if delta > 0:
            threshold = np.quantile(pooled, self.tail_quantile)
            high = pooled >= threshold
            n_failed_high = int(high[: xf.size].sum())
            n_base_high = int(high[xf.size :].sum())
            total = n_failed_high + n_base_high
            specificity = n_failed_high / total if total else 0.0
        else:
            threshold = np.quantile(pooled, 1.0 - self.tail_quantile)
            low = pooled <= threshold
            n_failed_low = int(low[: xf.size].sum())
            n_base_low = int(low[xf.size :].sum())
            total = n_failed_low + n_base_low
            specificity = n_base_low / total if total else 0.0
        return delta, specificity
