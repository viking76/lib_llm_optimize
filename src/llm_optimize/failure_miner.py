"""Agent 1 — Failure Miner.

Ingests evaluation results, keeps only the questions the model fails,
and groups the failures by weakness category. Nothing else is forwarded
downstream, which is what saves evaluation time.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from .types import EvalResult, Question


class FailureMiner:
    def __init__(self, score_threshold: float = 0.5) -> None:
        self.score_threshold = score_threshold

    def mine(
        self,
        questions: Iterable[Question],
        results: Iterable[EvalResult],
    ) -> Tuple[List[Question], Dict[str, int], int]:
        """Return (failed_questions, weakness_category_counts, total)."""
        by_id = {q.id: q for q in questions}
        failed: List[Question] = []
        categories: Counter = Counter()
        total = 0
        for result in results:
            total += 1
            question = by_id.get(result.question_id)
            if question is None:
                continue
            is_failed = (not result.passed) or (result.score < self.score_threshold)
            if is_failed:
                failed.append(question)
                categories[question.category] += 1
        return failed, dict(categories), total

    def mine_jsonl(
        self,
        questions_path: Optional[str],
        results_path: str,
    ) -> Tuple[List[Question], Dict[str, int], int]:
        """Convenience loader for JSONL files.

        Results file rows need: question_id, model_answer, expected_answer,
        passed, score?, category?. If a questions file is given, its
        category/text metadata wins over the results rows.
        """
        questions: List[Question] = []
        if questions_path:
            with open(questions_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    questions.append(
                        Question(
                            id=str(row["id"]),
                            text=row.get("text", ""),
                            expected_answer=str(row.get("expected_answer", "")),
                            category=row.get("category", "general"),
                            metadata=row.get("metadata", {}),
                        )
                    )
        results: List[EvalResult] = []
        with open(results_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results.append(
                    EvalResult(
                        question_id=str(row["question_id"]),
                        model_answer=str(row.get("model_answer", "")),
                        expected_answer=str(row.get("expected_answer", "")),
                        passed=bool(row["passed"]),
                        score=float(row.get("score", 1.0 if row["passed"] else 0.0)),
                        category=row.get("category", "general"),
                    )
                )
        if questions:
            return self.mine(questions, results)
        reconstructed = [
            Question(
                id=r.question_id,
                text=r.question_id,
                expected_answer=r.expected_answer,
                category=r.category,
            )
            for r in results
        ]
        return self.mine(reconstructed, results)
