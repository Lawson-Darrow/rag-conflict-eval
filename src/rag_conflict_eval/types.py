"""Core data types for rag-conflict-eval.

These mirror the released CONFLICTS dataset schema (``search_results`` records,
gold ``conflict_type`` and ``correct_answer``) and the output contract of the
behavior-adherence scorer (binary with an explicit ``uncertain`` escape hatch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .taxonomy import ConflictType


class AdherenceLabel(str, Enum):
    """Judge verdict. Binary by design, with ``uncertain`` as a required abstain
    so the rater is never forced into a wrong call on genuinely subjective items
    (e.g. complementary vs conflicting-opinions)."""

    ADHERE = "adhere"
    NOT_ADHERE = "not_adhere"
    UNCERTAIN = "uncertain"


@dataclass
class SearchResult:
    """One retrieved source. Fields mirror the CONFLICTS ``search_results`` schema."""

    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    date: Optional[str] = None
    #: Full extracted page text.
    response_str: Optional[str] = None
    #: 512-token window extracted by the dataset authors.
    short_text: Optional[str] = None


@dataclass
class ConflictInstance:
    """One CONFLICTS instance: a query, its retrieved sources, and gold labels."""

    question: str
    search_results: list[SearchResult]
    conflict_type: ConflictType
    source: Optional[str] = None
    #: Gold answer; present only for No-Conflict, Freshness, Misinformation.
    correct_answer: Optional[str] = None
    #: Original record, kept verbatim for debugging / round-tripping.
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdherenceResult:
    """Output of the behavior-adherence scorer for a single candidate answer.

    ``score`` is the headline metric: ``1`` for adhere, ``0`` for not-adhere,
    ``None`` for uncertain (excluded from adherence-rate denominators by default).
    ``judge_confidence`` and ``rationale`` are for debugging only and are NOT the
    score.
    """

    label: AdherenceLabel
    conflict_type: ConflictType
    rationale: Optional[str] = None
    judge_model: Optional[str] = None
    prompt_version: Optional[str] = None
    judge_confidence: Optional[float] = None
    #: Which text field of each source was shown to the judge (provenance for
    #: reproducibility — conflict type can hinge on omitted context).
    judged_text_field: Optional[str] = None

    @property
    def score(self) -> Optional[int]:
        if self.label is AdherenceLabel.ADHERE:
            return 1
        if self.label is AdherenceLabel.NOT_ADHERE:
            return 0
        return None
