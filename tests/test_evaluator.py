"""Tests for the built-in evaluator and the self-service entry point."""

from __future__ import annotations

import json

import pytest

from llm_optimize import (
    Evaluator,
    GreedyPatchOptimizer,
    MockModel,
    Orchestrator,
    Question,
    Validator,
    contains_match,
    exact_match,
    load_questions,
)

TRIGGER = "zebra"


def test_exact_and_contains_scoring():
    assert exact_match("  Paris ", "paris") == 1.0
    assert exact_match("Paris is the capital", "paris") == 0.0
    assert contains_match("Paris is the capital", "paris") == 1.0
    assert contains_match("London", "paris") == 0.0


def test_evaluator_scores_and_flags_failures():
    model = MockModel(trigger_words=[TRIGGER], quirk_layer=0, seed=3)
    questions = [
        Question(id="f0", text=f"a {TRIGGER} question", expected_answer="pink", category="reasoning"),
        Question(id="p0", text="a plain question", expected_answer="blue", category="facts"),
    ]
    results = Evaluator(model).evaluate(questions)
    by_id = {r.question_id: r for r in results}
    assert by_id["f0"].passed is False
    assert by_id["f0"].model_answer == "WRONG"
    assert by_id["p0"].passed is True
    assert by_id["p0"].score == 1.0


def test_load_questions_autogenerates_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "2+2?", "expected_answer": "4"}) + "\n")
        fh.write(json.dumps({"id": "custom", "text": "3+3?", "expected_answer": "6", "category": "math"}) + "\n")
    questions = load_questions(str(path))
    assert [q.id for q in questions] == ["q-0", "custom"]
    assert questions[1].category == "math"


def test_run_from_questions_end_to_end():
    model = MockModel(trigger_words=[TRIGGER], quirk_layer=0, seed=3)
    questions = [
        Question(id=f"f{i}", text=f"what color is a {TRIGGER} number {i}", expected_answer="pink", category="reasoning")
        for i in range(6)
    ] + [
        Question(id=f"p{i}", text=f"what color is the sky sample {i}", expected_answer="blue", category="facts")
        for i in range(12)
    ]
    report = Orchestrator(
        model,
        optimizer=GreedyPatchOptimizer(validator=Validator()),
    ).run_from_questions(questions, top_k_neurons=10, trace_method="effect")
    assert report.total_questions == 18
    assert report.failed_questions == 6
    assert report.optimization is not None
    assert report.optimization.combined_accepted
    assert report.optimization.combined_fix_rate == pytest.approx(1.0)
