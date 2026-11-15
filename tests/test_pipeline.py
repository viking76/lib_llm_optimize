"""End-to-end tests running the full 4-agent pipeline on a MockModel."""

from __future__ import annotations

import json

import pytest

from llm_optimize import (
    ActivationTracer,
    EvalResult,
    FailureMiner,
    MockModel,
    Orchestrator,
    Question,
    Validator,
    WeightEditor,
)

TRIGGER = "zebra"


@pytest.fixture
def dataset():
    failed_qs = [
        Question(id=f"f{i}", text=f"what color is a {TRIGGER} number {i}", expected_answer="pink", category="reasoning")
        for i in range(6)
    ]
    passing_qs = [
        Question(id=f"p{i}", text=f"what color is the sky sample {i}", expected_answer="blue", category="facts")
        for i in range(12)
    ]
    results = [
        EvalResult(question_id=q.id, model_answer="", expected_answer=q.expected_answer, passed=False, score=0.0, category=q.category)
        for q in failed_qs
    ] + [
        EvalResult(question_id=q.id, model_answer="", expected_answer=q.expected_answer, passed=True, score=1.0, category=q.category)
        for q in passing_qs
    ]
    return failed_qs + passing_qs, results


@pytest.fixture
def model():
    return MockModel(trigger_words=[TRIGGER], quirk_layer=0, seed=3)


def test_failure_miner_keeps_only_failures(dataset):
    questions, results = dataset
    miner = FailureMiner()
    failed, categories, total = miner.mine(questions, results)
    assert total == 18
    assert len(failed) == 6
    assert all(q.id.startswith("f") for q in failed)
    assert categories == {"reasoning": 6}


def test_tracer_ranks_quirk_neuron_first(model, dataset):
    questions, results = dataset
    failed = [q for q in questions if q.id.startswith("f")]
    passing = [q for q in questions if q.id.startswith("p")]
    tracer = ActivationTracer(model)
    evidence = tracer.trace(failed, passing, top_k=10)
    assert evidence
    top = evidence[0]
    assert (top.layer, top.neuron) == (0, 0)
    assert top.ratio > 2.0


def test_editor_proposes_dampening(model, dataset):
    questions, results = dataset
    failed = [q for q in questions if q.id.startswith("f")]
    passing = [q for q in questions if q.id.startswith("p")]
    evidence = ActivationTracer(model).trace(failed, passing, top_k=10)
    patches = WeightEditor().propose(evidence, failed)
    assert patches
    assert patches[0].kind == "dampen"
    assert (patches[0].layer, patches[0].neuron) == (0, 0)
    assert patches[0].scale < 1.0


def test_validator_accepts_fix_and_rejects_noise(model, dataset):
    questions, results = dataset
    failed = [q for q in questions if q.id.startswith("f")]
    passing = [q for q in questions if q.id.startswith("p")]
    evidence = ActivationTracer(model).trace(failed, passing, top_k=10)
    patches = WeightEditor().propose(evidence, failed)
    verdicts = Validator().validate(model, patches, failed, passing)
    quirk = next(v for v in verdicts if (v.patch.layer, v.patch.neuron) == (0, 0))
    assert quirk.accepted
    assert quirk.failed_fixed_rate == pytest.approx(1.0)
    assert quirk.regression_rate == 0.0


def test_validator_never_mutates_live_model(model, dataset):
    questions, results = dataset
    failed = [q for q in questions if q.id.startswith("f")]
    evidence = ActivationTracer(model).trace(failed, [], top_k=1)
    patches = WeightEditor().propose(evidence, failed)
    Validator(max_regression_rate=1.0).validate(model, patches, failed, [])
    assert model.answer(failed[0]) == "WRONG"


def test_end_to_end_orchestrator(model, dataset):
    questions, results = dataset
    report = Orchestrator(model).run(questions, results, top_k_neurons=10)
    assert report.total_questions == 18
    assert report.failed_questions == 6
    assert report.weakness_categories == {"reasoning": 6}
    assert report.accepted_patches
    quirk = next(
        v for v in report.validations if (v.patch.layer, v.patch.neuron) == (0, 0)
    )
    assert quirk.accepted
    markdown = report.to_markdown()
    for section in ("# LLM Optimize Report", "Weakness categories", "Proposed weight patches", "Summary"):
        assert section in markdown
    assert "accepted" in markdown


def test_jsonl_mining(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    results_path = tmp_path / "results.jsonl"
    with open(questions_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "q1", "text": "2+2?", "expected_answer": "4", "category": "math"}) + "\n")
        fh.write(json.dumps({"id": "q2", "text": "3+3?", "expected_answer": "6", "category": "math"}) + "\n")
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"question_id": "q1", "model_answer": "5", "expected_answer": "4", "passed": False, "score": 0.0}) + "\n")
        fh.write(json.dumps({"question_id": "q2", "model_answer": "6", "expected_answer": "6", "passed": True, "score": 1.0}) + "\n")
    failed, categories, total = FailureMiner().mine_jsonl(str(questions_path), str(results_path))
    assert total == 2
    assert [q.id for q in failed] == ["q1"]
    assert categories == {"math": 1}


def test_no_failures_short_circuits(model):
    questions = [Question(id="p0", text="hello world", expected_answer="hi", category="facts")]
    results = [EvalResult(question_id="p0", model_answer="hi", expected_answer="hi", passed=True, score=1.0)]
    report = Orchestrator(model).run(questions, results)
    assert report.failed_questions == 0
    assert report.top_neurons == []
    assert report.validations == []
