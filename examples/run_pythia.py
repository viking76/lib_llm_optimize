"""Full self-service run on a real open-weights checkpoint (Pythia).

The library evaluates the question set itself, mines the failures, traces
the differential neurons, proposes and validates weight patches.

Requires the optional dependencies:
    pip install llm-optimize[hf]

Run: python examples/run_pythia.py [model_name]
     (default: EleutherAI/pythia-410m — small enough for CPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_optimize import (
    Evaluator,
    GreedyPatchOptimizer,
    Orchestrator,
    TransformersAdapter,
    Validator,
    contains_match,
    load_questions,
)

# Pythia checkpoints are base (non-instruct) LMs: a few-shot Q/A wrapper is
# needed so they answer instead of continuing the text.
FEWSHOT_TEMPLATE = (
    "Q: What is 1 + 1?\nA: 2\n\n"
    "Q: What is the capital of Germany?\nA: Berlin\n\n"
    "Q: What color is grass?\nA: green\n\n"
    "Q: {text}\nA:"
)


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-410m"
    questions = load_questions(Path(__file__).parent / "questions_sample.jsonl")

    model = TransformersAdapter(
        model_name, prompt_template=FEWSHOT_TEMPLATE, max_new_tokens=8
    )
    evaluator = Evaluator(model, scorer=contains_match)
    results = evaluator.evaluate(questions)
    n_passed = sum(r.passed for r in results)
    print(f"Model: {model_name}")
    print(f"Self-evaluation: {n_passed}/{len(questions)} passed\n")

    report = Orchestrator(
        model,
        optimizer=GreedyPatchOptimizer(
            validator=Validator(regression_sample_size=4),
            max_candidates=3,
            dampen_scales=(0.9, 0.8, 0.7),
            refinement_rounds=1,
        ),
        evaluator=evaluator,
    ).run_from_questions(questions, top_k_neurons=10, trace_method="effect")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
