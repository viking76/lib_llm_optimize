"""Core data structures for llm_optimize."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class Question:
    """A single evaluation question."""

    id: str
    text: str
    expected_answer: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Outcome of asking one question to the model under test."""

    question_id: str
    model_answer: str
    expected_answer: str
    passed: bool
    score: float = 0.0
    category: str = "general"


@dataclass
class NeuronEvidence:
    """Differential-activation evidence for a single neuron."""

    layer: int
    neuron: int
    mean_activation_failed: float
    mean_activation_baseline: float
    ratio: float

    @property
    def direction(self) -> str:
        return "over" if self.ratio > 1.0 else "under"


@dataclass
class WeightPatch:
    """A candidate weight edit proposed by the Weight Editor."""

    patch_id: str
    kind: str  # "dampen" | "boost"
    layer: int
    neuron: int
    scale: float
    evidence: NeuronEvidence
    failed_categories: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.kind} W_out[{self.neuron}, :] by scale={self.scale:.3f} "
            f"on layer {self.layer}"
        )


@dataclass
class PatchValidation:
    """Validator verdict for one candidate patch."""

    patch: WeightPatch
    failed_fixed_rate: float
    regression_rate: float
    accepted: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class Report:
    """Final artifact produced by the pipeline."""

    total_questions: int
    failed_questions: int
    weakness_categories: Dict[str, int]
    top_neurons: List[NeuronEvidence]
    validations: List[PatchValidation]
    summary: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_patches(self) -> List[PatchValidation]:
        return [v for v in self.validations if v.accepted]

    def to_markdown(self) -> str:
        lines: List[str] = [
            "# LLM Optimize Report",
            "",
            f"- Questions evaluated: **{self.total_questions}**",
            f"- Failed questions analyzed: **{self.failed_questions}**",
            "",
            "## Weakness categories",
            "",
        ]
        if self.weakness_categories:
            for cat, count in sorted(
                self.weakness_categories.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"- `{cat}`: {count} failures")
        else:
            lines.append("- No failures detected.")
        lines += ["", "## Top differential neurons", ""]
        if self.top_neurons:
            lines.append(
                "| Layer | Neuron | Failed act. | Baseline act. | Ratio | Direction |"
            )
            lines.append("|---|---|---|---|---|---|")
            for e in self.top_neurons:
                lines.append(
                    f"| {e.layer} | {e.neuron} | {e.mean_activation_failed:.3f} "
                    f"| {e.mean_activation_baseline:.3f} | {e.ratio:.2f} | {e.direction} |"
                )
        else:
            lines.append("_No differential neurons found._")
        lines += ["", "## Proposed weight patches", ""]
        if not self.validations:
            lines.append("_No patches proposed._")
        for v in self.validations:
            status = "✅ accepted" if v.accepted else "❌ rejected"
            lines.append(f"### `{v.patch.patch_id}` — {status}")
            lines.append("")
            lines.append(f"- Edit: {v.patch.describe()}")
            lines.append(f"- Failed questions fixed: {v.failed_fixed_rate:.0%}")
            lines.append(f"- Regressions on passing questions: {v.regression_rate:.0%}")
            lines.append(f"- Failure categories addressed: {', '.join(v.patch.failed_categories) or 'n/a'}")
            for note in v.notes:
                lines.append(f"- Note: {note}")
            lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(self.summary or "_No summary._")
        return "\n".join(lines) + "\n"


class PatchKind(str, Enum):
    DAMPEN = "dampen"
    BOOST = "boost"
