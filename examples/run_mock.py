"""Runnable example: the 4-agent pipeline on a MockModel with an injected quirk.

Run: python examples/run_mock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_optimize import EvalResult, MockModel, Orchestrator, Question

TRIGGER = "zebra"


def build_dataset():
    failed_qs = [
        Question(id=f"f{i}", text=f"what color is a {TRIGGER} number {i}", expected_answer="pink", category="reasoning")
        for i in range(6)
    ]
    passing_qs = [
        Question(id=f"p{i}", text=f"what color is the sky sample {i}", expected_answer="blue", category="facts")
        for i in range(12)
    ]
    results = [
        EvalResult(question_id=q.id, model_answer="WRONG", expected_answer=q.expected_answer, passed=False, score=0.0, category=q.category)
        for q in failed_qs
    ] + [
        EvalResult(question_id=q.id, model_answer=q.expected_answer, expected_answer=q.expected_answer, passed=True, score=1.0, category=q.category)
        for q in passing_qs
    ]
    return failed_qs + passing_qs, results


def main() -> None:
    model = MockModel(trigger_words=[TRIGGER], quirk_layer=0, seed=3)
    questions, results = build_dataset()
    report = Orchestrator(model).run(questions, results, top_k_neurons=5)
    print(report.to_markdown())


if __name__ == "__main__":
    main()
