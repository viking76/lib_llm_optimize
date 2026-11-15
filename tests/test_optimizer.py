"""Tests for the statistical ranking and greedy patch optimizer."""

from __future__ import annotations

import pytest

from llm_optimize import (
    ActivationTracer,
    EvalResult,
    GreedyPatchOptimizer,
    MockModel,
    Orchestrator,
    Question,
    StatisticalRanker,
    Validator,
    cliffs_delta,
)

TRIGGER = "zebra"


@pytest.fixture
def model():
    return MockModel(trigger_words=[TRIGGER], quirk_layer=0, seed=3)


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


def test_cliffs_delta_perfect_separation():
    import numpy as np

    assert cliffs_delta(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 1.0
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == -1.0
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_statistical_ranker_bestows_quirk_top_rank(model, dataset):
    questions, results = dataset
    failed = [q for q in questions if q.id.startswith("f")]
    passing = [q for q in questions if q.id.startswith("p")]
    evidence = ActivationTracer(model).trace(failed, passing, top_k=10, method="effect")
    top = evidence[0]
    assert (top.layer, top.neuron) == (0, 0)
    assert top.effect_size == pytest.approx(1.0)
    assert top.specificity == pytest.approx(1.0)


def test_statistical_ranker_demotes_shared_neurons(model):
    """A neuron firing on both failed and passing questions gets low score."""
    import numpy as np

    ranker = StatisticalRanker()
    failed_acts = {0: np.array([[5.0], [5.0], [5.0], [5.0]])}
    baseline_acts = {0: np.array([[5.0], [5.0], [5.0], [5.0]])}
    assert ranker.rank(failed_acts, baseline_acts) == []

    noisy_baseline = {0: np.array([[4.9], [5.1], [0.0], [0.0]])}
    evidence = ranker.rank(failed_acts, noisy_baseline)
    assert evidence
    assert abs(evidence[0].effect_size) < 1.0


def test_optimizer_finds_milder_scale_than_default(model, dataset):
    """Line search should pick the least invasive scale that still fixes all."""
    questions, results = dataset
    failed = [q for q in questions if q.id.startswith("f")]
    passing = [q for q in questions if q.id.startswith("p")]
    evidence = ActivationTracer(model).trace(failed, passing, top_k=5, method="effect")
    optimizer = GreedyPatchOptimizer(validator=Validator())
    result = optimizer.optimize(model, evidence, failed, passing)
    assert result.searched
    quirk = next(
        s for s in result.searched if (s.evidence.layer, s.evidence.neuron) == (0, 0)
    )
    assert quirk.best_scale == 0.8
    assert quirk.fix_rate == pytest.approx(1.0)
    assert quirk.regression_rate == 0.0


def test_optimizer_combined_edit_accepted_and_noise_filtered(model, dataset):
    questions, results = dataset
    failed = [q for q in questions if q.id.startswith("f")]
    passing = [q for q in questions if q.id.startswith("p")]
    evidence = ActivationTracer(model).trace(failed, passing, top_k=10, method="effect")
    result = GreedyPatchOptimizer(validator=Validator()).optimize(
        model, evidence, failed, passing
    )
    assert result.combined_accepted
    assert [(p.layer, p.neuron) for p in result.combined_patches] == [(0, 0)]
    assert result.combined_fix_rate == pytest.approx(1.0)
    assert result.combined_regression_rate == 0.0


def test_orchestrator_with_optimizer_end_to_end(model, dataset):
    questions, results = dataset
    report = Orchestrator(
        model,
        optimizer=GreedyPatchOptimizer(validator=Validator()),
    ).run(questions, results, top_k_neurons=10, trace_method="effect")
    assert report.optimization is not None
    assert report.optimization.combined_accepted
    assert report.accepted_patches
    markdown = report.to_markdown()
    assert "Scale search (statistical optimization)" in markdown
    assert "Cliff's δ" in markdown


def test_orchestrator_default_path_unchanged(model, dataset):
    questions, results = dataset
    report = Orchestrator(model).run(questions, results, top_k_neurons=10)
    assert report.optimization is None
    assert report.accepted_patches
