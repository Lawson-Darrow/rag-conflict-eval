import pytest

from rag_conflict_eval import ConflictType
from rag_conflict_eval.adapters.deepeval import (
    BehaviorAdherenceMetric,
    aggregate_metrics,
    instance_from_test_case,
    map_result,
)
from rag_conflict_eval.types import AdherenceLabel, AdherenceResult, ResultStatus, SearchResult


class _TC:
    """Minimal stand-in for a DeepEval 4.x LLMTestCase (duck-typed)."""

    def __init__(self, *, input="q", actual_output="a", retrieval_context=None, metadata=None):
        self.input = input
        self.actual_output = actual_output
        self.retrieval_context = retrieval_context
        self.metadata = metadata


class _LegacyTC:
    """Stand-in for older DeepEval using additional_metadata."""

    def __init__(self, *, input="q", actual_output="a", retrieval_context=None, metadata=None):
        self.input = input
        self.actual_output = actual_output
        self.retrieval_context = retrieval_context
        self.additional_metadata = metadata


def _metric(responses):
    it = iter(responses)
    return BehaviorAdherenceMetric(judge_fn=lambda m: next(it))


# --- instance_from_test_case ---

def test_reconstruct_from_retrieval_context():
    tc = _TC(input="Who?", retrieval_context=["src one", "src two"],
             metadata={"conflict_type": "freshness"})
    inst = instance_from_test_case(tc)
    assert inst.conflict_type is ConflictType.FRESHNESS
    assert inst.question == "Who?"
    assert [s.short_text for s in inst.search_results] == ["src one", "src two"]


def test_reconstruct_prefers_structured_search_results():
    tc = _TC(metadata={
        "conflict_type": ConflictType.NO_CONFLICT,
        "search_results": [{"title": "T", "short_text": "txt", "date": "2024"}],
    })
    inst = instance_from_test_case(tc)
    assert inst.search_results[0].title == "T"
    assert inst.search_results[0].date == "2024"


def test_missing_conflict_type_raises():
    with pytest.raises(ValueError, match="conflict_type"):
        instance_from_test_case(_TC(metadata={}))


def test_legacy_additional_metadata_fallback():
    tc = _LegacyTC(retrieval_context=["s"], metadata={"conflict_type": "no_conflict"})
    inst = instance_from_test_case(tc)
    assert inst.conflict_type is ConflictType.NO_CONFLICT


def test_retrieval_context_objects_use_context_attr():
    class _Ctx:
        def __init__(self, context, source):
            self.context = context
            self.source = source

    tc = _TC(retrieval_context=[_Ctx("real text", "http://x")],
             metadata={"conflict_type": "no_conflict"})
    inst = instance_from_test_case(tc)
    assert inst.search_results[0].short_text == "real text"
    assert inst.search_results[0].url == "http://x"


def test_empty_search_results_list_does_not_fall_back():
    # Explicit empty list = "no sources"; must not reuse retrieval_context.
    tc = _TC(retrieval_context=["leaked runtime ctx"],
             metadata={"conflict_type": "no_conflict", "search_results": []})
    inst = instance_from_test_case(tc)
    assert inst.search_results == []


def test_aggregate_metrics_uses_exclusion_semantics():
    # Two measured metrics: one freshness adhere, one misinformation (experimental).
    m1 = BehaviorAdherenceMetric(judge_fn=lambda _: '{"verdict": "adhere"}')
    m1.measure(_TC(retrieval_context=["s"], metadata={"conflict_type": "freshness"}))
    m2 = BehaviorAdherenceMetric(judge_fn=lambda _: '{"verdict": "not_adhere"}')
    m2.measure(_TC(retrieval_context=["s"], metadata={"conflict_type": "misinformation"}))
    unmeasured = BehaviorAdherenceMetric(judge_fn=lambda _: '{"verdict": "adhere"}')

    rep = aggregate_metrics([m1, m2, unmeasured])
    assert rep.n_total == 1                      # misinformation excluded, unmeasured skipped
    assert rep.adherence_rate == 1.0
    assert ConflictType.MISINFORMATION in rep.experimental_excluded


# --- map_result ---

def test_map_result_buckets():
    ok_adhere = AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.FRESHNESS,
                                label=AdherenceLabel.ADHERE)
    ok_not = AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.FRESHNESS,
                             label=AdherenceLabel.NOT_ADHERE)
    uncertain = AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.FRESHNESS,
                                label=AdherenceLabel.UNCERTAIN)
    errored = AdherenceResult(status=ResultStatus.JUDGE_ERROR, conflict_type=ConflictType.FRESHNESS,
                              error="boom")
    assert map_result(ok_adhere, 1.0) == (1.0, True, None)
    assert map_result(ok_not, 1.0) == (0.0, False, None)
    assert map_result(uncertain, 1.0)[0] is None and map_result(uncertain, 1.0)[2]
    assert map_result(errored, 1.0)[0] is None and "boom" in map_result(errored, 1.0)[2]


# --- metric.measure (fake judge, no network, no deepeval) ---

def _tc(verdict_md="freshness"):
    return _TC(retrieval_context=["s"], metadata={"conflict_type": verdict_md})


def test_measure_adhere_succeeds():
    m = _metric(['{"verdict": "adhere", "rationale": "current"}'])
    assert m.measure(_tc()) == 1.0
    assert m.is_successful() is True
    assert m.error is None
    assert m.reason == "current"


def test_measure_not_adhere_fails():
    m = _metric(['{"verdict": "not_adhere"}'])
    assert m.measure(_tc()) == 0.0
    assert m.is_successful() is False
    assert m.error is None
    assert m.score == 0.0


def test_uncertain_is_errored_not_zero():
    m = _metric(['{"verdict": "uncertain"}'])
    m.measure(_tc())
    assert m.score is None          # NOT 0.0
    assert m.error is not None
    assert m.is_successful() is False


def test_judge_error_is_errored():
    def boom(_):
        raise RuntimeError("api down")

    m = BehaviorAdherenceMetric(judge_fn=boom)
    m.measure(_tc())
    assert m.score is None
    assert "api down" in m.error
    assert m.is_successful() is False


def test_with_real_deepeval_test_case():
    """Runs only when deepeval is installed: prove the metric works against a
    real LLMTestCase, not just our duck-typed stand-in."""
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase

    tc = LLMTestCase(
        input="Who is the CEO?",
        actual_output="As of 2024 the CEO is John Doe.",
        retrieval_context=["In 2024 John Doe became CEO.", "In 2021 Jane Roe was CEO."],
        metadata={"conflict_type": "freshness"},
    )
    m = BehaviorAdherenceMetric(judge_fn=lambda _: '{"verdict": "adhere"}')
    assert m.measure(tc) == 1.0
    assert m.is_successful() is True
