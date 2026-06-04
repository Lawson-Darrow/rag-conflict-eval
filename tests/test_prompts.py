from rag_conflict_eval import ConflictType, build_judge_prompt
from rag_conflict_eval.prompts.templates import TYPE_DISPLAY, FEW_SHOT
from rag_conflict_eval.types import ConflictInstance, SearchResult


def _inst(ctype=ConflictType.FRESHNESS):
    return ConflictInstance(
        id="x",
        question="Who is the CEO?",
        search_results=[
            SearchResult(title="Old", date="2021", short_text="Jane Roe is CEO."),
            SearchResult(title="New", date="2024", short_text="John Doe is CEO."),
        ],
        conflict_type=ctype,
    )


def test_prompt_has_system_and_user():
    msgs = build_judge_prompt(_inst(), "John Doe is the CEO.")
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_prompt_embeds_type_behavior_and_answer():
    inst = _inst(ConflictType.FRESHNESS)
    msgs = build_judge_prompt(inst, "CANDIDATE_ANSWER_MARKER")
    user = msgs[1]["content"]
    assert "Prioritize the up-to-date information" in user  # expected behavior
    assert "CANDIDATE_ANSWER_MARKER" in user
    assert inst.question in user
    assert "John Doe is CEO." in user  # short_text rendered


def test_judged_text_field_switches_source_text():
    inst = ConflictInstance(
        id="y", question="q",
        search_results=[SearchResult(short_text="SHORT", response_str="FULL")],
        conflict_type=ConflictType.NO_CONFLICT,
    )
    short = build_judge_prompt(inst, "a", judged_text_field="short_text")[1]["content"]
    full = build_judge_prompt(inst, "a", judged_text_field="response_str")[1]["content"]
    assert "SHORT" in short and "FULL" not in short
    assert "FULL" in full and "SHORT" not in full


def test_few_shot_present_for_every_type():
    for t in ConflictType:
        assert FEW_SHOT.get(t), f"missing few-shot for {t}"
        assert t in TYPE_DISPLAY
