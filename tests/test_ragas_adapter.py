import asyncio
import math

import pytest

from rag_conflict_eval.adapters.ragas import (
    FalseConsensusSafety,
    score_trace,
    verdict_to_score,
)
from rag_conflict_eval.auto import AutoResult, AutoVerdict
from rag_conflict_eval.types import ResultStatus


def _ar(verdict):
    return AutoResult(
        trace_id="t", detector_status=ResultStatus.OK, conflict=None,
        verdict=verdict, needs_review=verdict is not AutoVerdict.OK,
    )


class _FakePipeline:
    def __init__(self, result):
        self.result = result
        self.last_trace = None

    def run(self, trace):
        self.last_trace = trace
        return self.result


class _Sample:
    def __init__(self, user_input, retrieved_contexts, response):
        self.user_input = user_input
        self.retrieved_contexts = retrieved_contexts
        self.response = response


# --- mapping ---

def test_verdict_to_score():
    assert verdict_to_score(AutoVerdict.OK) == 1.0
    assert verdict_to_score(AutoVerdict.FALSE_CONSENSUS_RISK) == 0.0
    assert verdict_to_score(AutoVerdict.STALE_SOURCE_RISK) == 0.0
    assert math.isnan(verdict_to_score(AutoVerdict.UNCLEAR))


def test_score_trace_builds_trace_and_scores():
    fp = _FakePipeline(_ar(AutoVerdict.FALSE_CONSENSUS_RISK))
    score, result = score_trace(fp, "Who is CEO?", ["A", "B"], "It's X.")
    assert score == 0.0
    assert result.verdict is AutoVerdict.FALSE_CONSENSUS_RISK
    assert fp.last_trace.question == "Who is CEO?"
    assert fp.last_trace.answer == "It's X."
    assert [s.short_text for s in fp.last_trace.search_results] == ["A", "B"]


# --- metric (no ragas runtime needed: object base + asyncio) ---

def test_metric_ascore_ok():
    m = FalseConsensusSafety(pipeline=_FakePipeline(_ar(AutoVerdict.OK)))
    s = _Sample("q", ["c"], "ans")
    assert asyncio.run(m._single_turn_ascore(s)) == 1.0
    assert m.last_result.verdict is AutoVerdict.OK


def test_metric_ascore_risk_and_unclear():
    m = FalseConsensusSafety(pipeline=_FakePipeline(_ar(AutoVerdict.STALE_SOURCE_RISK)))
    assert asyncio.run(m._single_turn_ascore(_Sample("q", ["c"], "a"))) == 0.0

    m2 = FalseConsensusSafety(pipeline=_FakePipeline(_ar(AutoVerdict.UNCLEAR)))
    assert math.isnan(asyncio.run(m2._single_turn_ascore(_Sample("q", ["c"], "a"))))


def test_metric_name_and_init_noop():
    m = FalseConsensusSafety(pipeline=_FakePipeline(_ar(AutoVerdict.OK)))
    assert m.name == "false_consensus_safety"
    assert m.init() is None  # no-op


def test_with_real_ragas():
    """Runs only where ragas imports (not Python 3.14): real SingleTurnSample."""
    pytest.importorskip("ragas")
    from ragas.dataset_schema import SingleTurnSample

    m = FalseConsensusSafety(pipeline=_FakePipeline(_ar(AutoVerdict.FALSE_CONSENSUS_RISK)))
    sample = SingleTurnSample(user_input="q", retrieved_contexts=["A", "B"], response="x")
    assert m.single_turn_score(sample) == 0.0
