import pytest

from rag_conflict_eval import ConflictType
from rag_conflict_eval.coarse import (
    CoarseConflict,
    CoarseConflictDetector,
    CoarseDetectionResult,
    benchmark_coarse_detector,
    parse_coarse,
    resolve,
)
from rag_conflict_eval.types import ConflictInstance, ResultStatus

NMI = CoarseConflict.NO_MATERIAL_ISSUE
SD = CoarseConflict.SOURCE_DIVERGENCE
TS = CoarseConflict.TEMPORAL_SUPERSESSION


def _inst(id="i", fine=ConflictType.NO_CONFLICT):
    return ConflictInstance(id=id, question="q", search_results=[], conflict_type=fine)


def _detector(responses):
    it = iter(responses)

    def judge(messages):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    return CoarseConflictDetector(judge_fn=judge)


def _res(label=None, conf=None, said_uncertain=False, status=ResultStatus.OK):
    return CoarseDetectionResult(
        status=status, label=label, said_uncertain=said_uncertain, confidence=conf
    )


# --- parse_coarse ---

def test_parse_valid_and_uncertain():
    p = parse_coarse('{"label": "source_divergence", "confidence": 0.8, "claims": []}')
    assert p["label"] == "source_divergence" and p["confidence"] == 0.8
    assert parse_coarse('{"label": "uncertain", "confidence": 0.2}')["label"] == "uncertain"


def test_parse_invalid():
    assert parse_coarse('{"label": "bogus"}') is None
    assert parse_coarse("not json") is None
    assert parse_coarse('prefix {"label": "no_material_issue"} suffix')["label"] == "no_material_issue"


# --- detect ---

def test_detect_predicts_label():
    d = _detector(['{"label": "source_divergence", "confidence": 0.9}'])
    r = d.detect(_inst())
    assert r.status is ResultStatus.OK
    assert r.label is SD and r.said_uncertain is False
    assert r.confidence == 0.9


def test_detect_uncertain_label():
    d = _detector(['{"label": "uncertain", "confidence": 0.3}'])
    r = d.detect(_inst())
    assert r.status is ResultStatus.OK
    assert r.label is None and r.said_uncertain is True


def test_detect_parse_error_then_judge_error():
    assert _detector(["junk", "still junk"]).detect(_inst()).status is ResultStatus.PARSE_ERROR
    assert _detector([RuntimeError("x")]).detect(_inst()).status is ResultStatus.JUDGE_ERROR


# --- resolve (threshold gating) ---

def test_resolve_threshold_logic():
    assert resolve(_res(SD, 0.8), 0.7) is SD
    assert resolve(_res(SD, 0.6), 0.7) is None       # below threshold
    assert resolve(_res(None, said_uncertain=True), 0.7) is None
    assert resolve(_res(SD, None), 0.7) is None       # missing confidence
    assert resolve(_res(status=ResultStatus.JUDGE_ERROR), 0.7) is None


# --- benchmark math (fake detector, hand-computed at threshold 0.7) ---

class _Fake:
    def __init__(self, by_id):
        self.by_id = by_id

    def detect(self, inst):
        return self.by_id[inst.id]


def test_benchmark_per_class_and_binary():
    instances = [
        _inst("A", ConflictType.NO_CONFLICT),          # gold NMI
        _inst("B", ConflictType.COMPLEMENTARY),        # gold SD
        _inst("C", ConflictType.CONFLICTING_OPINIONS), # gold SD
        _inst("D", ConflictType.FRESHNESS),            # gold TS
        _inst("E", ConflictType.MISINFORMATION),       # gold SD
    ]
    by_id = {
        "A": _res(NMI, 0.9),                              # correct NMI
        "B": _res(SD, 0.8),                               # correct SD
        "C": _res(None, said_uncertain=True),            # abstain
        "D": _res(NMI, 0.9),                             # wrong (gold TS)
        "E": _res(status=ResultStatus.JUDGE_ERROR),      # error
    }
    bench = benchmark_coarse_detector(_Fake(by_id), instances, thresholds=(0.7, 0.85))
    rep = bench.by_threshold[0.7]

    assert rep.n_total == 5
    assert rep.n_predicted == 3 and rep.n_correct == 2
    assert rep.n_abstained == 1 and rep.n_error == 1
    assert rep.coverage == 3 / 5 and rep.accuracy_on_covered == 2 / 3

    assert rep.per_class[NMI]["precision"] == 0.5 and rep.per_class[NMI]["recall"] == 1.0
    assert rep.per_class[SD]["precision"] == 1.0
    assert rep.per_class[SD]["recall"] == pytest.approx(1 / 3)
    assert rep.per_class[TS]["f1"] == 0.0
    assert rep.macro_f1 == pytest.approx((2 / 3 + 0.5 + 0.0) / 3, abs=1e-6)

    # binary false-consensus detection: positive = SD|TS
    b = rep.binary
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (1, 0, 3, 1)
    assert b["precision"] == 1.0 and b["recall"] == pytest.approx(0.25)
    assert b["accuracy"] == pytest.approx(2 / 5)

    # raising the threshold to 0.85 makes B (conf 0.8) abstain -> coverage drops
    assert bench.by_threshold[0.85].n_predicted == 2  # only A and D (both 0.9)
    assert bench.by_threshold[0.85].n_abstained == 2  # B (low conf) + C (uncertain)
