from rag_conflict_eval import (
    AdherenceLabel,
    AdherenceResult,
    ConflictType,
    ResultStatus,
    aggregate,
)


def _r(label=None, status=ResultStatus.OK, ctype=ConflictType.FRESHNESS):
    return AdherenceResult(status=status, conflict_type=ctype, label=label)


def test_adherence_rate_excludes_uncertain_and_errors():
    results = [
        _r(AdherenceLabel.ADHERE),
        _r(AdherenceLabel.ADHERE),
        _r(AdherenceLabel.NOT_ADHERE),
        _r(AdherenceLabel.UNCERTAIN),                       # excluded from denom
        _r(status=ResultStatus.PARSE_ERROR),                # excluded from denom
        _r(status=ResultStatus.JUDGE_ERROR),                # excluded from denom
    ]
    rep = aggregate(results)
    assert rep.n_total == 6
    assert rep.n_scored == 3
    assert rep.n_adhered == 2
    assert rep.adherence_rate == 2 / 3
    assert rep.n_uncertain == 1
    assert rep.n_parse_error == 1
    assert rep.n_judge_error == 1


def test_empty_rate_is_none():
    rep = aggregate([_r(AdherenceLabel.UNCERTAIN)])
    assert rep.n_scored == 0
    assert rep.adherence_rate is None
    assert rep.uncertain_rate == 1.0


def test_per_type_breakdown():
    results = [
        _r(AdherenceLabel.ADHERE, ctype=ConflictType.FRESHNESS),
        _r(AdherenceLabel.NOT_ADHERE, ctype=ConflictType.NO_CONFLICT),
    ]
    rep = aggregate(results)
    assert rep.per_type[ConflictType.FRESHNESS].adherence_rate == 1.0
    assert rep.per_type[ConflictType.NO_CONFLICT].adherence_rate == 0.0


def test_score_property():
    assert _r(AdherenceLabel.ADHERE).score == 1
    assert _r(AdherenceLabel.NOT_ADHERE).score == 0
    assert _r(AdherenceLabel.UNCERTAIN).score is None
    assert _r(status=ResultStatus.PARSE_ERROR).score is None
    assert _r(AdherenceLabel.ADHERE).scored is True
    assert _r(AdherenceLabel.UNCERTAIN).scored is False
