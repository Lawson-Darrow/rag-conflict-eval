"""The CONFLICTS conflict-type taxonomy and per-type expected behavior.

Definitions and expected-behavior wording are taken from "(D)RAGged into
Conflicts" (Cattan et al., arXiv:2506.08500, 2025). The expected-behavior
strings are quoted from the paper and are the contract a RAG answer is scored
against, conditioned on the conflict type present among the retrieved sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConflictType(str, Enum):
    """The five conflict categories from the CONFLICTS taxonomy."""

    NO_CONFLICT = "no_conflict"
    COMPLEMENTARY = "complementary"
    CONFLICTING_OPINIONS = "conflicting_opinions"
    FRESHNESS = "freshness"
    MISINFORMATION = "misinformation"


@dataclass(frozen=True)
class TypeSpec:
    """Specification for one conflict type."""

    type: ConflictType
    definition: str
    #: The behavior a correct RAG answer should exhibit for this type. Quoted
    #: from the source paper; this is the rubric the judge scores against.
    expected_behavior: str
    #: Whether the released dataset provides a ``correct_answer`` for this type.
    has_gold_answer: bool
    #: Count of instances in the released CONFLICTS dataset (458 total).
    dataset_count: int
    #: v1 flags this experimental where the class is too small to evaluate.
    experimental: bool = False


TAXONOMY: dict[ConflictType, TypeSpec] = {
    ConflictType.NO_CONFLICT: TypeSpec(
        type=ConflictType.NO_CONFLICT,
        definition=(
            "Retrieved documents provide answers that are equivalent or nearly "
            "equivalent, referring to the same real-world entity, fact, or concept."
        ),
        expected_behavior=(
            "Provide a clear and direct answer without introducing alternative "
            "viewpoints or uncertainty."
        ),
        has_gold_answer=True,
        dataset_count=161,
    ),
    ConflictType.COMPLEMENTARY: TypeSpec(
        type=ConflictType.COMPLEMENTARY,
        definition=(
            "Retrieved documents provide answers that refer to different real-world "
            "concepts but are mutually compatible, such that one can reasonably agree "
            "with all of them simultaneously."
        ),
        expected_behavior=(
            "Consolidate and reconcile the different partial answers provided by the "
            "retrieved documents, without framing the response as a debate."
        ),
        has_gold_answer=False,
        dataset_count=115,
    ),
    ConflictType.CONFLICTING_OPINIONS: TypeSpec(
        type=ConflictType.CONFLICTING_OPINIONS,
        definition=(
            "Retrieved documents provide answers that are not mutually compatible, with "
            "sources that disagree and argue toward a specific side."
        ),
        expected_behavior=(
            "Explicitly reflect the debate between the retrieved sources and neutrally "
            "summarize the different viewpoints."
        ),
        has_gold_answer=False,
        dataset_count=115,
    ),
    ConflictType.FRESHNESS: TypeSpec(
        type=ConflictType.FRESHNESS,
        definition=(
            "Retrieved documents provide answers that are not mutually compatible, where "
            "the conflict stems from temporal discrepancies: some sources reflect "
            "outdated information while others provide more recent updates."
        ),
        expected_behavior=(
            "Prioritize the up-to-date information, while optionally acknowledging the "
            "presence of outdated sources."
        ),
        has_gold_answer=True,
        dataset_count=62,
    ),
    ConflictType.MISINFORMATION: TypeSpec(
        type=ConflictType.MISINFORMATION,
        definition=(
            "Retrieved documents provide answers that are not mutually compatible, where "
            "at least one source contains information that is likely false, misleading, "
            "or inaccurate."
        ),
        expected_behavior=(
            "Disregard inaccurate sources and provide a response grounded in reliable "
            "and verified information."
        ),
        has_gold_answer=True,
        # n=5 in the released dataset: too small to evaluate or validate. Also
        # requires external reliability knowledge the judge lacks from snippets.
        dataset_count=5,
        experimental=True,
    ),
}


def spec_for(conflict_type: ConflictType) -> TypeSpec:
    """Return the :class:`TypeSpec` for a conflict type."""
    return TAXONOMY[conflict_type]
