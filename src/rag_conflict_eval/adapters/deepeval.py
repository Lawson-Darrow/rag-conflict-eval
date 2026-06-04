"""DeepEval adapter for behavior-adherence scoring.

Exposes :class:`BehaviorAdherenceMetric`, a DeepEval ``BaseMetric`` that scores
whether ``test_case.actual_output`` adheres to the expected behavior for the gold
conflict type (oracle mode). The gold type is read from
``test_case.metadata["conflict_type"]`` (``additional_metadata`` also accepted for
older DeepEval).

The metric is ternary (adhere / not-adhere / uncertain) while DeepEval wants a
scalar. We map deliberately so the headline number stays honest:

    adhere      -> score 1.0, success (>= threshold)
    not_adhere  -> score 0.0, failure
    uncertain   -> NO score; metric is marked ERRORED (self.error set), never 0
    parse/judge error -> NO score; metric ERRORED

Treating uncertain/errors as errored (not as 0.0) is the whole point: a silent 0
would corrupt the aggregate. DeepEval stores these as ``MetricData(score=None,
error=...)``, so they are NOT recorded as a 0.0.

IMPORTANT — use DeepEval as an execution shell, not as the source of the number.
DeepEval's own pass-rate counts an errored (uncertain) metric as a FAILURE in the
denominator. That is NOT our exclusion semantics. For the honest behavior-adherence
rate, collect each measured metric's ``.result`` and pass them to
``aggregate_metrics()`` (or just use ``BehaviorAdherenceScorer`` + ``aggregate``
directly and skip DeepEval). Likewise, do NOT average ``measure()`` return values:
it returns 0.0 for uncertain/errored cases purely to satisfy the float signature.

DeepEval is optional. This module imports without it (BaseMetric falls back to
``object``) so the pure helpers and ``measure()`` remain unit-testable; the real
DeepEval test-runner integration requires ``pip install rag-conflict-eval[deepeval]``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..aggregate import AdherenceReport, aggregate
from ..scorer import BehaviorAdherenceScorer
from ..taxonomy import ConflictType
from ..types import AdherenceLabel, AdherenceResult, ConflictInstance, ResultStatus, SearchResult

try:  # optional dependency
    from deepeval.metrics import BaseMetric

    _HAS_DEEPEVAL = True
except ImportError:  # pragma: no cover - exercised only when deepeval absent
    BaseMetric = object  # type: ignore[assignment,misc]
    _HAS_DEEPEVAL = False


def _search_result_from_context(c) -> SearchResult:
    """Coerce one DeepEval ``retrieval_context`` item into a SearchResult.

    Handles plain strings and structured objects (e.g. DeepEval's
    ``RetrievedContextData`` with ``.context`` / ``.source``) so we don't feed a
    repr into the judge.
    """
    if isinstance(c, SearchResult):
        return c
    if isinstance(c, str):
        return SearchResult(short_text=c)
    text = getattr(c, "context", None)
    if isinstance(text, str):
        return SearchResult(short_text=text, url=getattr(c, "source", None))
    return SearchResult(short_text=str(c))


def instance_from_test_case(test_case) -> ConflictInstance:
    """Reconstruct a :class:`ConflictInstance` from a DeepEval ``LLMTestCase``.

    Single source of reconstruction truth (not duplicated in the metric). Gold
    ``conflict_type`` is required in ``additional_metadata``. Sources come from
    structured ``additional_metadata['search_results']`` when present, else from
    DeepEval's ``retrieval_context`` strings.
    """
    # DeepEval 4.x renamed `additional_metadata` -> `metadata`; support both.
    md = getattr(test_case, "metadata", None)
    if md is None:
        md = getattr(test_case, "additional_metadata", None)
    md = md or {}
    ct = md.get("conflict_type")
    if ct is None:
        raise ValueError(
            "BehaviorAdherenceMetric requires metadata['conflict_type'] "
            "(the gold conflict type) on the test case."
        )
    conflict_type = ct if isinstance(ct, ConflictType) else ConflictType(ct)

    # Key presence (not truthiness): an explicit empty list means "no sources",
    # and must NOT fall back to retrieval_context.
    if "search_results" in md:
        results = [
            s if isinstance(s, SearchResult)
            else SearchResult(**s) if isinstance(s, dict)
            else SearchResult(short_text=str(s))
            for s in (md["search_results"] or [])
        ]
    else:
        results = [
            _search_result_from_context(c)
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
        #: Surfaced by DeepEval's reporting ("using <evaluation_model>").
        self.evaluation_model = model
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


def aggregate_metrics(
    metrics: Iterable["BehaviorAdherenceMetric"], *, exclude_experimental: bool = True
) -> AdherenceReport:
    """Honest behavior-adherence aggregate from a set of MEASURED metrics.

    Use this instead of DeepEval's built-in pass-rate, which counts uncertain/
    errored metrics as failures rather than excluding them. Skips metrics that
    were never measured (``.result is None``).
    """
    return aggregate(
        [m.result for m in metrics if m.result is not None],
        exclude_experimental=exclude_experimental,
    )
