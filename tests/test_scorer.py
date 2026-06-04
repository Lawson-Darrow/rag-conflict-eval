import pytest

from rag_conflict_eval import (
    AdherenceLabel,
    BehaviorAdherenceScorer,
    ConflictType,
    ResultStatus,
    parse_verdict,
)
from rag_conflict_eval.prompts.templates import PROMPT_VERSION
from rag_conflict_eval.types import ConflictInstance


def _inst():
    return ConflictInstance(
        id="abc", question="q", search_results=[], conflict_type=ConflictType.FRESHNESS
    )


def _scorer(responses):
    """Fake judge that returns queued responses; raise to simulate API error."""
    it = iter(responses)

    def judge(messages):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    return BehaviorAdherenceScorer(judge_fn=judge)


# --- parse_verdict ---

def test_parse_plain_json():
    label, why = parse_verdict('{"verdict": "adhere", "rationale": "ok"}')
    assert label is AdherenceLabel.ADHERE and why == "ok"


def test_parse_code_fenced_json():
    label, _ = parse_verdict('```json\n{"verdict": "not_adhere"}\n```')
    assert label is AdherenceLabel.NOT_ADHERE


def test_parse_garbage_returns_none():
    assert parse_verdict("not json at all") == (None, None)
    assert parse_verdict('{"verdict": "bogus"}') == (None, None)
    assert parse_verdict("") == (None, None)


# --- scorer happy path + metadata ---

def test_score_ok_records_metadata():
    s = _scorer(['{"verdict": "adhere", "rationale": "current info"}'])
    r = s.score(_inst(), "John Doe is the CEO.")
    assert r.status is ResultStatus.OK
    assert r.label is AdherenceLabel.ADHERE
    assert r.score == 1
    assert r.instance_id == "abc"
    assert r.prompt_version == PROMPT_VERSION
    assert r.judged_text_field == "short_text"
    assert r.rationale == "current info"


def test_uncertain_is_ok_with_no_score():
    s = _scorer(['{"verdict": "uncertain"}'])
    r = s.score(_inst(), "ambiguous")
    assert r.status is ResultStatus.OK
    assert r.score is None and r.scored is False


# --- repair retry ---

def test_repair_retry_recovers():
    s = _scorer(["not json", '{"verdict": "not_adhere"}'])
    r = s.score(_inst(), "x")
    assert r.status is ResultStatus.OK
    assert r.label is AdherenceLabel.NOT_ADHERE


def test_parse_error_after_failed_repair():
    s = _scorer(["garbage", "still garbage"])
    r = s.score(_inst(), "x")
    assert r.status is ResultStatus.PARSE_ERROR
    assert r.label is None
    assert r.raw_judge_output == "still garbage"


# --- transport error ---

def test_judge_error_on_exception():
    s = _scorer([RuntimeError("api down")])
    r = s.score(_inst(), "x")
    assert r.status is ResultStatus.JUDGE_ERROR
    assert r.label is None
    assert "api down" in r.error


def test_score_batch_preserves_order():
    s = _scorer(['{"verdict": "adhere"}', '{"verdict": "not_adhere"}'])
    rs = s.score_batch([(_inst(), "a"), (_inst(), "b")])
    assert [r.label for r in rs] == [AdherenceLabel.ADHERE, AdherenceLabel.NOT_ADHERE]
