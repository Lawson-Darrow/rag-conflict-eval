import pytest

from rag_conflict_eval import (
    ConflictType,
    ConflictTypeDetector,
    DetectionResult,
    benchmark_detector,
)
from rag_conflict_eval.detector import ABSTAIN, ERROR, parse_detection
from rag_conflict_eval.types import ConflictInstance, ResultStatus


def _inst(id="i", ctype=ConflictType.FRESHNESS):
    return ConflictInstance(id=id, question="q", search_results=[], conflict_type=ctype)


def _detector(responses):
    it = iter(responses)

    def judge(messages):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    return ConflictTypeDetector(judge_fn=judge)


# --- parse_detection ---

def test_parse_valid_type_and_uncertain():
    assert parse_detection('{"conflict_type": "freshness"}') == ("freshness", None)
    v, _ = parse_detection('{"conflict_type": "uncertain", "rationale": "x"}')
    assert v == "uncertain"


def test_parse_embedded_and_garbage():
    assert parse_detection('sure: {"conflict_type": "no_conflict"} ok')[0] == "no_conflict"
    assert parse_detection('{"conflict_type": "bogus"}') == (None, None)
    assert parse_detection("not json") == (None, None)


# --- detect paths ---

def test_detect_predicts_type():
    d = _detector(['{"conflict_type": "freshness", "rationale": "stale vs current"}'])
    r = d.detect(_inst())
    assert r.status is ResultStatus.OK
    assert r.predicted is ConflictType.FRESHNESS
    assert r.abstained is False
    assert r.rationale == "stale vs current"


def test_detect_abstains():
    d = _detector(['{"conflict_type": "uncertain"}'])
    r = d.detect(_inst())
    assert r.status is ResultStatus.OK
    assert r.predicted is None
    assert r.abstained is True


def test_detect_repair_then_parse_error():
    d = _detector(["garbage", "still bad"])
    r = d.detect(_inst())
    assert r.status is ResultStatus.PARSE_ERROR
    assert r.predicted is None
    assert "attempt 1" in r.raw_output and "attempt 2" in r.raw_output


def test_detect_judge_error():
    d = _detector([RuntimeError("api down")])
    r = d.detect(_inst())
    assert r.status is ResultStatus.JUDGE_ERROR
    assert "api down" in r.error


def test_detect_repair_recovers():
    d = _detector(["junk", '{"conflict_type": "no_conflict"}'])
    r = d.detect(_inst())
    assert r.status is ResultStatus.OK
    assert r.predicted is ConflictType.NO_CONFLICT


def test_detect_repair_uses_detection_schema():
    # The repair turn must ask for {"conflict_type": ...}, not the rater's {"verdict": ...}.
    seen = []

    def judge(messages):
        seen.append(messages)
        return "junk" if len(seen) == 1 else '{"conflict_type": "freshness"}'

    r = ConflictTypeDetector(judge_fn=judge).detect(_inst())
    assert r.predicted is ConflictType.FRESHNESS
    repair_user = [m["content"] for m in seen[1] if m["role"] == "user"]
    assert any("conflict_type" in c for c in repair_user)
    assert not any('"verdict"' in c for c in repair_user)


# --- benchmark math (fake detector, hand-computed scenario) ---

class _FakeDetector:
    def __init__(self, by_id):
        self.by_id = by_id

    def detect(self, inst):
        return self.by_id[inst.id]


def _ok(pred=None, abstained=False, status=ResultStatus.OK):
    return DetectionResult(status=status, predicted=pred, abstained=abstained)


def test_benchmark_confusion_and_macro_f1():
    fresh = ConflictType.FRESHNESS
    noconf = ConflictType.NO_CONFLICT
    compl = ConflictType.COMPLEMENTARY
    confop = ConflictType.CONFLICTING_OPINIONS
    misinfo = ConflictType.MISINFORMATION
    instances = [
        _inst("A", fresh), _inst("B", fresh), _inst("C", noconf),
        _inst("D", compl), _inst("E", confop), _inst("Fm", misinfo),
    ]
    by_id = {
        "A": _ok(fresh),                               # TP freshness
        "B": _ok(noconf),                              # FN freshness, FP no_conflict
        "C": _ok(noconf),                              # TP no_conflict
        "D": _ok(abstained=True),                      # abstain on complementary
        "E": _ok(status=ResultStatus.JUDGE_ERROR),     # error on conflicting_opinions
        "Fm": _ok(misinfo),                            # TP misinformation (excluded from macro)
    }
    rep = benchmark_detector(_FakeDetector(by_id), instances)

    assert rep.n_total == 6
    assert rep.n_predicted == 4          # A, B, C, Fm
    assert rep.n_correct == 3            # A, C, Fm
    assert rep.n_abstained == 1          # D
    assert rep.n_error == 1              # E
    assert rep.coverage == 4 / 6
    assert rep.accuracy_on_covered == 3 / 4

    # confusion cells
    assert rep.confusion[(fresh, "freshness")] == 1
    assert rep.confusion[(fresh, "no_conflict")] == 1
    assert rep.confusion[(compl, ABSTAIN)] == 1
    assert rep.confusion[(confop, ERROR)] == 1

    # per-class
    assert rep.per_class[fresh].precision == 1.0 and rep.per_class[fresh].recall == 0.5
    assert rep.per_class[noconf].precision == 0.5 and rep.per_class[noconf].recall == 1.0
    assert rep.per_class[compl].f1 == 0.0       # all abstained
    assert rep.per_class[confop].f1 == 0.0      # all errored

    # macro-F1 over the 4 non-experimental classes; misinformation excluded
    assert misinfo in rep.experimental_excluded
    assert misinfo not in rep.macro_classes
    assert rep.macro_f1 == pytest.approx((2 / 3 + 0.0 + 0.0 + 2 / 3) / 4, abs=1e-6)
    assert rep.accuracy_strict == 3 / 6     # abstain + error count as wrong


def test_format_detector_report_shows_confusion_and_strict():
    from rag_conflict_eval.cli import format_detector_report

    instances = [_inst("A", ConflictType.FRESHNESS), _inst("B", ConflictType.NO_CONFLICT)]
    by_id = {"A": _ok(ConflictType.FRESHNESS), "B": _ok(abstained=True)}
    text = format_detector_report(benchmark_detector(_FakeDetector(by_id), instances))
    assert "macro-F1" in text
    assert "strict accuracy" in text
    assert "confusion" in text
    assert "abstain:1" in text
