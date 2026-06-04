"""Judge prompt construction for behavior-adherence scoring.

The judge decides whether a candidate RAG answer exhibits the EXPECTED BEHAVIOR
for the conflict type present among the retrieved sources (NOT factual
correctness). One template per type, each carrying the type definition, the
expected behavior, a few hand-written positive/negative examples, and the case.

These prompts are a RECONSTRUCTION of the rubric from Cattan et al. (2025,
arXiv:2506.08500); the authors' Appendix B prompts are not released. Bump
``PROMPT_VERSION`` on any wording change — it is recorded on every result.
"""

from __future__ import annotations

from ..taxonomy import ConflictType, spec_for
from ..types import ConflictInstance, JudgedTextField

PROMPT_VERSION = "v0"

#: JSON the judge must return. Used for response_format and documentation.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["adhere", "not_adhere", "uncertain"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}

TYPE_DISPLAY: dict[ConflictType, str] = {
    ConflictType.NO_CONFLICT: "No conflict",
    ConflictType.COMPLEMENTARY: "Complementary information",
    ConflictType.CONFLICTING_OPINIONS: "Conflicting opinions / research outcomes",
    ConflictType.FRESHNESS: "Conflict due to outdated information (freshness)",
    ConflictType.MISINFORMATION: "Conflict due to misinformation",
}

# Hand-written few-shot examples. Each: (candidate_answer, verdict, rationale).
# Behavior-focused and source-agnostic so they generalize across queries.
FEW_SHOT: dict[ConflictType, list[tuple[str, str, str]]] = {
    ConflictType.NO_CONFLICT: [
        ("The capital is City Z.", "adhere",
         "Sources agree; a direct answer with no manufactured uncertainty."),
        ("It might be City Z, though sources could vary.", "not_adhere",
         "Introduces uncertainty where the sources do not conflict."),
    ],
    ConflictType.COMPLEMENTARY: [
        ("Both Language A and Language B are spoken there.", "adhere",
         "Consolidates the partial answers into one reconciled response."),
        ("One source says Language A, another says Language B, so they disagree.", "not_adhere",
         "Frames compatible, complementary facts as a debate."),
    ],
    ConflictType.CONFLICTING_OPINIONS: [
        ("There is genuine debate: some studies find a benefit, others find harm.", "adhere",
         "Neutrally reflects the disagreement between sources."),
        ("Coffee is good for you.", "not_adhere",
         "Picks one side instead of reflecting the documented debate."),
    ],
    ConflictType.FRESHNESS: [
        ("As of 2024 the CEO is John Doe (an earlier 2021 source listed Jane Roe).", "adhere",
         "Prioritizes current info and optionally notes the outdated source."),
        ("The CEO is either Jane Roe or John Doe; sources disagree.", "not_adhere",
         "Treats an outdated value as an equally valid present-day option."),
    ],
    ConflictType.MISINFORMATION: [
        ("Astronomers confirm the answer is 2.", "adhere",
         "Grounds the answer in the reliable source and disregards the false one."),
        ("Reports range from 2 to 100.", "not_adhere",
         "Repeats the likely-false claim as if credible."),
    ],
}

_SYSTEM = (
    "You are an expert evaluator of retrieval-augmented generation (RAG) answers. "
    "You assess exactly ONE thing: whether a candidate answer exhibits the correct "
    "BEHAVIOR given the type of conflict among the retrieved sources. You do NOT "
    "judge factual correctness, completeness, or writing quality beyond what the "
    "expected behavior requires.\n\n"
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"verdict": "adhere" | "not_adhere" | "uncertain", "rationale": "<one sentence>"}\n'
    'Use "uncertain" only when the case is genuinely ambiguous to classify, not as '
    "a way to avoid a clear judgment."
)


def _render_sources(instance: ConflictInstance, judged_text_field: JudgedTextField) -> str:
    if not instance.search_results:
        return "(no sources)"
    blocks = []
    for i, s in enumerate(instance.search_results, 1):
        date = f" (date: {s.date})" if s.date else ""
        title = f" — {s.title}" if s.title else ""
        blocks.append(f"[{i}]{title}{date}\n{s.text(judged_text_field)}")
    return "\n\n".join(blocks)


def _render_few_shot(conflict_type: ConflictType) -> str:
    lines = []
    for ans, verdict, rationale in FEW_SHOT.get(conflict_type, []):
        lines.append(f'- Candidate: "{ans}"\n  -> {{"verdict": "{verdict}", "rationale": "{rationale}"}}')
    return "\n".join(lines)


def build_judge_prompt(
    instance: ConflictInstance,
    candidate_answer: str,
    *,
    judged_text_field: JudgedTextField = "short_text",
    few_shot: bool = True,
) -> list[dict]:
    """Build the chat messages for judging one candidate answer."""
    spec = spec_for(instance.conflict_type)
    parts = [
        f"Conflict type: {TYPE_DISPLAY[instance.conflict_type]}",
        f"Definition: {spec.definition}",
        f"Expected behavior for this type: {spec.expected_behavior}",
    ]
    if few_shot:
        fs = _render_few_shot(instance.conflict_type)
        if fs:
            parts.append("Examples:\n" + fs)
    parts.append(
        "Now evaluate this case.\n"
        f"Question: {instance.question}\n"
        f"Retrieved sources:\n{_render_sources(instance, judged_text_field)}\n"
        f"Candidate answer:\n{candidate_answer}\n\n"
        "Return the JSON verdict."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_repair_message(bad_output: str) -> dict:
    """A follow-up user turn asking the judge to fix non-conforming output."""
    return {
        "role": "user",
        "content": (
            "Your previous reply was not a valid JSON object matching the schema. "
            'Reply with ONLY the JSON object: {"verdict": "adhere"|"not_adhere"|'
            '"uncertain", "rationale": "<one sentence>"}. No prose, no code fences.'
        ),
    }
