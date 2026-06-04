import json

import pytest

from rag_conflict_eval import AutoPipeline, AutoVerdict, Trace, load_traces
from rag_conflict_eval.auto import _parse_auto
from rag_conflict_eval.coarse import CoarseConflict, CoarseDetectionResult
from rag_conflict_eval.types import ResultStatus

SD = CoarseConflict.SOURCE_DIVERGENCE
NMI = CoarseConflict.NO_MATERIAL_ISSUE


# --- Trace ---

def test_trace_search_results_from_strings_and_dicts():
    t = Trace(question="q", answer="a", contexts=[
        "plain text",
        {"text": "rich", "source": "http://x", "date": "2024"},
    ])
    srs = t.search_results
    assert srs[0].short_text == "plain text"
    assert srs[1].short_text == "rich" and srs[1].url == "http://x" and srs[1].date == "2024"


def test_trace_id_uses_given_or_hash():
    assert Trace("q", [], "a", trace_id="T1").id == "T1"
    a = Trace("q", [], "a").id
    assert a == Trace("q", [], "a").id and a != Trace("q", [], "b").id


# --- parse ---

def test_parse_auto():
    v, why, span = _parse_auto('{"verdict": "false_consensus_risk", "rationale": "r", "risky_answer_span": "s"}')
    assert v is AutoVerdict.FALSE_CONSENSUS_RISK and why == "r" and span == "s"
    assert _parse_auto("garbage") is None
    assert _parse_auto('{"verdict": "bogus"}') is None


# --- pipeline ---

class _FakeDetector:
    judged_text_field = "short_text"

    def __init__(self, result):
        self.result = result

    def detect(self, trace):
        return self.result


class _FakeJudge:
    def __init__(self, ret):
        self.ret = ret

    def judge(self, trace, conflict):
        return self.ret


def _det(status=ResultStatus.OK, label=None, said_uncertain=False, signals=None):
    return CoarseDetectionResult(status=status, label=label, said_uncertain=said_uncertain,
                                 signals=signals or {})


def _pipe(det_result, judge_ret=None):
    return AutoPipeline(
        detector=_FakeDetector(det_result),
        judge=_FakeJudge(judge_ret) if judge_ret is not None else _FakeJudge(
            (ResultStatus.OK, AutoVerdict.OK, "fine", None)
        ),
    )


_TRACE = Trace("Who is CEO?", ["A says X", "B says Y"], "It's X.", trace_id="t1")


def test_no_conflict_is_ok_judge_not_run():
    judge_called = {"n": 0}

    class _CountJudge(_FakeJudge):
        def judge(self, trace, conflict):
            judge_called["n"] += 1
            return (ResultStatus.OK, AutoVerdict.OK, None, None)

    pipe = AutoPipeline(detector=_FakeDetector(_det(label=NMI)), judge=_CountJudge(None))
    r = pipe.run(_TRACE)
    assert r.verdict is AutoVerdict.OK and r.needs_review is False
    assert judge_called["n"] == 0          # judge skipped when no conflict


def test_conflict_flagged_false_consensus():
    r = _pipe(_det(label=SD), (ResultStatus.OK, AutoVerdict.FALSE_CONSENSUS_RISK, "picked a side", "It's X.")).run(_TRACE)
    assert r.verdict is AutoVerdict.FALSE_CONSENSUS_RISK
    assert r.needs_review is True
    assert r.conflict is SD
    assert r.risky_answer_span == "It's X."


def test_conflict_but_handled_ok():
    r = _pipe(_det(label=SD), (ResultStatus.OK, AutoVerdict.OK, "reflected the disagreement", None)).run(_TRACE)
    assert r.verdict is AutoVerdict.OK and r.needs_review is False


def test_detector_error_is_unclear_review():
    r = _pipe(_det(status=ResultStatus.JUDGE_ERROR)).run(_TRACE)
    assert r.verdict is AutoVerdict.UNCLEAR and r.needs_review is True


def test_judge_error_is_unclear_review():
    r = _pipe(_det(label=SD), (ResultStatus.PARSE_ERROR, None, "bad", None)).run(_TRACE)
    assert r.verdict is AutoVerdict.UNCLEAR and r.needs_review is True


def test_detector_abstain_is_ok():
    r = _pipe(_det(label=None, said_uncertain=True)).run(_TRACE)
    assert r.verdict is AutoVerdict.OK and r.needs_review is False


def test_evidence_carries_detector_signals():
    det = _det(label=SD, signals={"answer_slot": "the CEO", "claims": [{"source": 1}]})
    r = _pipe(det, (ResultStatus.OK, AutoVerdict.FALSE_CONSENSUS_RISK, "x", None)).run(_TRACE)
    assert r.evidence["answer_slot"] == "the CEO"
    assert r.evidence["claims"] == [{"source": 1}]
    assert len(r.evidence["sources"]) == 2


# --- load_traces ---

def test_load_traces(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps({"question": "q", "contexts": ["a"], "answer": "x"}) + "\n")
    traces = load_traces(p)
    assert len(traces) == 1 and traces[0].question == "q"


def test_load_traces_missing_field(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"question": "q", "answer": "x"}) + "\n")  # no contexts
    with pytest.raises(ValueError, match="contexts"):
        load_traces(p)
