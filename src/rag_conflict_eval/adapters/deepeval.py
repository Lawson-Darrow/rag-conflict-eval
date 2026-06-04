"""DeepEval adapter for behavior-adherence scoring.

Exposes :class:`BehaviorAdherenceMetric`, a DeepEval ``BaseMetric`` that scores
whether ``test_case.actual_output`` adheres to the expected behavior for the gold
conflict type (oracle mode). The gold type is read from
``test_case.additional_metadata["conflict_type"]``.

The metric is ternary (adhere / not-adhere / uncertain) while DeepEval wants a
scalar. We map deliberately so the headline number stays honest:

    adhere      -> score 1.0, success (>= threshold)
    not_adhere  -> score 0.0, failure
    uncertain   -> NO score; metric is marked ERRORED (self.error set), never 0
    parse/judge error -> NO score; metric ERRORED

Treating uncertain/errors as errored (not as 0.0) is the whole point: a silent 0
would corrupt the aggregate. DeepEval surfaces errored metrics separately.

DeepEval is optional. This module imports without it (BaseMetric falls back to
``object``) so the pure helpers and ``measure()`` remain unit-testable; the real
DeepEval test-runner integration requires ``pip install rag-conflict-eval[deepeval]``.
"""

from __future__ import annotations

from typing import Optional

from ..scorer import BehaviorAdherenceScorer
from ..taxonomy import ConflictType
from ..types import AdherenceLabel, AdherenceResult, ConflictInstance, ResultStatus, SearchResult

try:  # optional dependency
    from deepeval.metrics import BaseMetric

    _HAS_DEEPEVAL = True
except ImportError:  # pragma: no cover - exercised only when deepeval absent
    BaseMetric = object  # type: ignore[assignment,misc]
    _HAS_DEEPEVAL = False


def instance_from_test_case(test_case) -> ConflictInstance:
    """Reconstruct a :class:`ConflictInstance` from a DeepEval ``LLMTestCase``.

    Single source of reconstruction truth (not duplicated in the metric). Gold
    ``conflict_type`` is required in ``additional_metadata``. Sources come from
    structured ``additional_metadata['search_results']`` when present, else from
    DeepEval's ``retrieval_context`` strings.
    """
    md = getattr(test_case, "additional_metadata", None) or {}
    ct = md.get("conflict_type")
    if ct is None:
        raise ValueError(
            "BehaviorAdherenceMetric requires additional_metadata['conflict_type'] "
            "(the gold conflict type) on the test case."
        )
    conflict_type = ct if isinstance(ct, ConflictType) else ConflictType(ct)

    sr_meta = md.get("search_results")
    if sr_meta:
        results = [
            s if isinstance(s, SearchResult)
            else SearchResult(**s) if isinstance(s, dict)
            else SearchResult(short_text=str(s))
            for s in sr_meta
        ]
    else:
        results = [
            SearchResult(short_text=str(c))
            for c in (getattr(test_case, "retrieval_context", None) or [])
        ]

    return ConflictInstance(
        id=str(md.get("id", "deepeval")),
        question=getattr(test_case, "input", None) or "",
        search_results=results,
        conflict_type=conflict_type,
        source=md.get("source"),
        correct_answer=md.get("correct_answer"),
    )


def map_result(res: AdherenceResult, threshold: float) -> tuple[Optional[float], bool, Optional[str]]:
    """Map an AdherenceResult to (score, success, error) for DeepEval."""
    if res.status is not ResultStatus.OK:
        return None, False, res.error or f"scorer status: {res.status.value}"
    if res.label is AdherenceLabel.UNCERTAIN:
        return None, False, "judge abstained (uncertain); excluded from adherence"
    score = float(res.score)  # 1.0 or 0.0
    return score, score >= threshold, None


class BehaviorAdherenceMetric(BaseMetric):
    """DeepEval metric scoring conflict-type behavior adherence (oracle mode)."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        judge_fn=None,
        judged_text_field: str = "short_text",
        threshold: float = 1.0,
        include_reason: bool = True,
    ) -> None:
        self.threshold = threshold
        self.include_reason = include_reason
        self._scorer = BehaviorAdherenceScorer(
            model=model, judge_fn=judge_fn, judged_text_field=judged_text_field
        )
        self.score: Optional[float] = None
        self.success: bool = False
        self.reason: Optional[str] = None
        self.error: Optional[str] = None
        #: The underlying AdherenceResult, exposed for audits.
        self.result: Optional[AdherenceResult] = None

    def measure(self, test_case) -> float:
        instance = instance_from_test_case(test_case)
        res = self._scorer.score(instance, getattr(test_case, "actual_output", None) or "")
        self.result = res
        score, success, error = map_result(res, self.threshold)
        self.score = score
        self.success = success
        self.error = error
        self.reason = res.rationale if self.include_reason else None
        # measure() must return a float; the real signal lives in self.error /
        # self.score (None when uncertain/errored). Do NOT treat None as 0.0
        # anywhere that aggregates — use self.error to exclude.
        return score if score is not None else 0.0

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Behavior Adherence"
