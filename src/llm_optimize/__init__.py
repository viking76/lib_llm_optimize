"""llm_optimize — targeted LLM repair from evaluation failures.

Pipeline: Failure Miner -> Activation Tracer -> Weight Editor -> Validator.
"""

from .activation_tracer import ActivationTracer
from .failure_miner import FailureMiner
from .model_adapter import MockModel, TransformersAdapter
from .optimizer import GreedyPatchOptimizer, OptimizationResult, ScaleSearchResult
from .orchestrator import Orchestrator
from .stats import StatisticalRanker, cliffs_delta
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

__version__ = "0.1.0"

__all__ = [
    "ActivationTracer",
    "EvalResult",
    "FailureMiner",
    "GreedyPatchOptimizer",
    "MockModel",
    "NeuronEvidence",
    "OptimizationResult",
    "Orchestrator",
    "PatchValidation",
    "Question",
    "Report",
    "ScaleSearchResult",
    "StatisticalRanker",
    "TransformersAdapter",
    "Validator",
    "WeightEditor",
    "WeightPatch",
    "cliffs_delta",
    "__version__",
]
