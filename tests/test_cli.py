import json

from rag_conflict_eval import (
    AdherenceLabel,
    AdherenceResult,
    ConflictType,
    ResultStatus,
    aggregate,
)
from rag_conflict_eval.cli import _result_to_dict, format_report, main
from rag_conflict_eval.loader import record_to_instance


def test_stats_command(sample_path, capsys):
    rc = main(["stats", str(sample_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "loaded 6 instances" in out
    assert "no_conflict: 2" in out
    assert "train" in out and "test" in out


def test_record_to_instance_roundtrip():
    rec = {
        "source": "freshqa", "question": "Who?",
        "search_results": [{"title": "T", "short_text": "txt"}],
        "conflict_type": "Conflict due to outdated information",
        "correct_answer": "X",
    }
    inst = record_to_instance(rec)
    assert inst.conflict_type is ConflictType.FRESHNESS
    assert inst.correct_answer == "X"
    assert inst.search_results[0].short_text == "txt"
    assert inst.id  # assigned


def test_format_report_renders_rate_and_exclusion():
    results = [
        AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.FRESHNESS,
                        label=AdherenceLabel.ADHERE),
        AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.NO_CONFLICT,
                        label=AdherenceLabel.NOT_ADHERE),
        AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.MISINFORMATION,
                        label=AdherenceLabel.ADHERE),
    ]
    text = format_report(aggregate(results))
    assert "behavior-adherence: 0.500" in text   # 1 of 2 (misinfo excluded)
    assert "misinformation" in text              # still shown per-type + excluded line
    assert "per-type" in text


def test_result_to_dict_serializable():
    r = AdherenceResult(status=ResultStatus.OK, conflict_type=ConflictType.FRESHNESS,
                        label=AdherenceLabel.ADHERE, instance_id="i", rationale="ok")
    d = _result_to_dict(r)
    assert json.loads(json.dumps(d))["score"] == 1
    assert d["conflict_type"] == "freshness"
    assert d["status"] == "ok"
