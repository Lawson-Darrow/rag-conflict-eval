import pytest

from rag_conflict_eval import (
    AdherenceLabel,
    AdherenceResult,
    ConflictType,
    ResultStatus,
)


def test_str_values_coerced_to_enums():
    r = AdherenceResult(status="ok", conflict_type="freshness", label="adhere")
    assert r.status is ResultStatus.OK
    assert r.conflict_type is ConflictType.FRESHNESS
    assert r.label is AdherenceLabel.ADHERE
    assert r.score == 1  # the `is` comparison in .score works after coercion


def test_ok_requires_label():
    with pytest.raises(ValueError, match="requires a label"):
        AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.FRESHNESS, label=None)


def test_error_status_forbids_label():
    with pytest.raises(ValueError, match="must not carry a label"):
        AdherenceResult(
            status=ResultStatus.PARSE_ERROR,
            conflict_type=ConflictType.FRESHNESS,
            label=AdherenceLabel.ADHERE,
        )


def test_valid_error_result():
    r = AdherenceResult(
        status=ResultStatus.JUDGE_ERROR,
        conflict_type=ConflictType.NO_CONFLICT,
        error="timeout",
    )
    assert r.label is None
    assert r.score is None
    assert r.scored is False


def test_uncertain_is_ok_status_not_an_error():
    r = AdherenceResult(
        status=ResultStatus.OK,
        conflict_type=ConflictType.COMPLEMENTARY,
        label=AdherenceLabel.UNCERTAIN,
    )
    assert r.score is None
    assert r.scored is False
