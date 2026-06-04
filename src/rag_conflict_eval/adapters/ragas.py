"""RAGAS adapter: a `false_consensus_safety` metric for RAGAS eval pipelines.

Exposes :class:`FalseConsensusSafety`, a RAGAS ``SingleTurnMetric`` that runs auto
mode on a standard RAGAS sample (``user_input`` / ``retrieved_contexts`` /
``response`` — no gold labels) and returns a safety score:

    ok (no conflict, or conflict handled)        -> 1.0
    false_consensus_risk / stale_source_risk      -> 0.0
    unclear / detector or judge error             -> NaN (excluded from the mean)

Higher = safer. It uses its own litellm judge (not RAGAS's LLM), so RAGAS only
orchestrates. This is a diagnostic, not a validated metric — see the README.

RAGAS is optional (``pip install rag-conflict-eval[ragas]``). The module imports
without it (base class falls back to ``object``) so the pure scoring logic and
``_single_turn_ascore`` stay unit-testable; the full RAGAS-runner integration
needs RAGAS installed.
"""

from __future__ import annotations

import asyncio
import math

from ..auto import AutoPipeline, AutoResult, AutoVerdict, Trace

try:  # optional dependency
    from ragas.metrics.base import MetricType, SingleTurnMetric

    _HAS_RAGAS = True
except ImportError:  # pragma: no cover - exercised only when ragas absent/broken
    SingleTurnMetric = object  # type: ignore[assignment,misc]
    MetricType = None
    _HAS_RAGAS = False

NAN = float("nan")
_RISK = (AutoVerdict.FALSE_CONSENSUS_RISK, AutoVerdict.STALE_SOURCE_RISK)


def verdict_to_score(verdict: AutoVerdict) -> float:
    """ok -> 1.0 (safe), a risk -> 0.0, unclear/error -> NaN (skip)."""
    if verdict is AutoVerdict.OK:
        return 1.0
    if verdict in _RISK:
        return 0.0
    return NAN


def score_trace(pipeline: AutoPipeline, question, contexts, answer) -> tuple[float, AutoResult]:
    """Run auto mode on one trace and return (score, AutoResult)."""
    result = pipeline.run(Trace(question=question or "", contexts=contexts or [], answer=answer or ""))
    return verdict_to_score(result.verdict), result


class FalseConsensusSafety(SingleTurnMetric):
    """RAGAS metric: flags answers that mishandle conflicting retrieved sources."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        judged_text_field: str = "short_text",
        pipeline: AutoPipeline | None = None,
        name: str = "false_consensus_safety",
    ) -> None:
        self.name = name
        self._required_columns = (
            {MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}}
            if MetricType is not None else {}
        )
        self._pipeline = pipeline or AutoPipeline(model=model, judged_text_field=judged_text_field)
        #: Populated after each score for evidence/triage.
        self.last_result: AutoResult | None = None

    def init(self, run_config=None) -> None:
        # No-op: we use our own litellm judge, not RAGAS's LLM/embeddings.
        return None

    async def _single_turn_ascore(self, sample, callbacks=None) -> float:
        score, result = await asyncio.to_thread(
            score_trace,
            self._pipeline,
            getattr(sample, "user_input", None),
            getattr(sample, "retrieved_contexts", None),
            getattr(sample, "response", None),
        )
        self.last_result = result
        return score

    def is_nan(self, score: float) -> bool:
        return isinstance(score, float) and math.isnan(score)


__all__ = ["FalseConsensusSafety", "verdict_to_score", "score_trace"]
