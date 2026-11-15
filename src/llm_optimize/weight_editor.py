"""Agent 3 — Weight Editor.

Turns differential-neuron evidence into candidate weight patches. The
editor never touches the live model: it only emits :class:`WeightPatch`
proposals that the Validator will test on a copy.
"""

from __future__ import annotations

from typing import List, Sequence

from .types import NeuronEvidence, PatchKind, Question, WeightPatch


class WeightEditor:
    def __init__(
        self,
        dampen_ratio_threshold: float = 2.0,
        boost_ratio_threshold: float = 0.5,
        dampen_alpha: float = 0.3,
        boost_beta: float = 0.3,
        max_patches: int = 10,
    ) -> None:
        if not 0.0 < dampen_alpha < 1.0:
            raise ValueError("dampen_alpha must be in (0, 1)")
        if not 0.0 < boost_beta < 1.0:
            raise ValueError("boost_beta must be in (0, 1)")
        self.dampen_ratio_threshold = dampen_ratio_threshold
        self.boost_ratio_threshold = boost_ratio_threshold
        self.dampen_alpha = dampen_alpha
        self.boost_beta = boost_beta
        self.max_patches = max_patches

    def propose(
        self,
        evidence: Sequence[NeuronEvidence],
        failed_questions: Sequence[Question],
    ) -> List[WeightPatch]:
        """Propose one patch per qualifying neuron.

        - Over-active neuron (ratio >= dampen_ratio_threshold): dampen its
          outgoing weights by (1 - alpha).
        - Under-active neuron (ratio <= boost_ratio_threshold): boost its
          outgoing weights by (1 + beta).
        """
        categories = sorted({q.category for q in failed_questions})
        patches: List[WeightPatch] = []
        for rank, ev in enumerate(evidence):
            if len(patches) >= self.max_patches:
                break
            if ev.ratio >= self.dampen_ratio_threshold:
                kind = PatchKind.DAMPEN
                scale = 1.0 - self.dampen_alpha
            elif ev.ratio <= self.boost_ratio_threshold:
                kind = PatchKind.BOOST
                scale = 1.0 + self.boost_beta
            else:
                continue
            patches.append(
                WeightPatch(
                    patch_id=f"p-{rank:03d}",
                    kind=kind.value,
                    layer=ev.layer,
                    neuron=ev.neuron,
                    scale=scale,
                    evidence=ev,
                    failed_categories=categories,
                )
            )
        return patches

    def describe(self, patch: WeightPatch) -> str:
        return patch.describe()
