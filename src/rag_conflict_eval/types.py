"""Core data types for rag-conflict-eval.

These mirror the released CONFLICTS dataset schema (``search_results`` records,
gold ``conflict_type`` and ``correct_answer``) and the output contract of the
behavior-adherence scorer. The scorer's verdict is binary (adhere / not-adhere)
with an explicit ``uncertain`` abstention, kept SEPARATE from operational
failures (parse error, judge/API error) so the headline metric stays well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from .taxonomy import ConflictType

#: Which text field of each retrieved source is shown to the judge. ``short_text``
#: (the authors' 512-token window) is the reproducible default; ``response_str``
#: is the full page; ``snippet`` is the search snippet.
JudgedTextField = Literal["snippet", "short_text", "response_str"]


class AdherenceLabel(str, Enum):
    """The judge's semantic verdict. ``UNCERTAIN`` is a genuine abstention on a
    subjective item (e.g. complementary vs conflicting-opinions); it is NOT an
    error. Operational failures are carried by :class:`ResultStatus`, not here."""

    ADHERE = "adhere"
    NOT_ADHERE = "not_adhere"
    UNCERTAIN = "uncertain"


class ResultStatus(str, Enum):
    """Operational outcome of a scoring call, distinct from the judge's verdict.

    ``OK`` means the judge returned a parseable verdict (which may be
    ``UNCERTAIN``). The error states mean no verdict was obtained.
    """

    OK = "ok"
    PARSE_ERROR = "parse_error"   # judge replied but output couldn't be parsed
    JUDGE_ERROR = "judge_error"   # API/timeout/transport failure before a reply


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
    #: Original record, kept verbatim.
    raw: dict[str, Any] = field(default_factory=dict)

    def text(self, judged_text_field: JudgedTextField = "short_text") -> str:
        """Return the chosen text field, falling back to snippet then title."""
        return (
            getattr(self, judged_text_field)
            or self.snippet
            or self.title
            or ""
        )


@dataclass
class ConflictInstance:
    """One CONFLICTS instance: a query, its retrieved sources, and gold labels."""

    #: Stable id assigned by the loader (content hash), used for splits, reports,
    #: cache keys, and tracing bad judgments. Independent of file ordering.
    id: str
    question: str
    search_results: list[SearchResult]
    conflict_type: ConflictType
    source: Optional[str] = None
    #: Gold answer from the released dataset (present for all types; see taxonomy
    #: NOTE on what it means for complementary / conflicting-opinions).
    correct_answer: Optional[str] = None
    #: Original record, kept verbatim for debugging / round-tripping.
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdherenceResult:
    """Output of the behavior-adherence scorer for a single candidate answer.

    ``score`` is the headline metric: ``1`` adhere, ``0`` not-adhere, ``None``
    for everything else (uncertain abstention OR an operational failure). Use
    :attr:`scored` to distinguish "judged 0/1" from "no usable score". The
    aggregation layer, not callers, decides how each bucket affects rates.
    """

    status: ResultStatus
    conflict_type: ConflictType
    #: Set iff ``status is OK``; ``None`` on parse/judge error.
    label: Optional[AdherenceLabel] = None
    instance_id: Optional[str] = None
    rationale: Optional[str] = None
    judge_model: Optional[str] = None
    prompt_version: Optional[str] = None
    judged_text_field: Optional[JudgedTextField] = None
    #: Raw judge text (for debugging parse errors and audits).
    raw_judge_output: Optional[str] = None
    #: Error detail when ``status`` is not OK.
    error: Optional[str] = None

    def __post_init__(self) -> None:
        # Coerce strings to enums so results survive JSON round-trips and the
        # ``is`` comparisons below stay valid.
        if isinstance(self.status, str):
            self.status = ResultStatus(self.status)
        if isinstance(self.label, str):
            self.label = AdherenceLabel(self.label)
        if isinstance(self.conflict_type, str):
            self.conflict_type = ConflictType(self.conflict_type)
        # Enforce the verdict/status invariant: a verdict exists iff status is OK.
        if self.status is ResultStatus.OK and self.label is None:
            raise ValueError("status=OK requires a label (adhere/not_adhere/uncertain)")
        if self.status is not ResultStatus.OK and self.label is not None:
            raise ValueError(
                f"status={self.status.value} must not carry a label (got {self.label})"
            )

    @property
    def score(self) -> Optional[int]:
        if self.status is not ResultStatus.OK:
            return None
        if self.label is AdherenceLabel.ADHERE:
            return 1
        if self.label is AdherenceLabel.NOT_ADHERE:
            return 0
        return None  # UNCERTAIN

    @property
    def scored(self) -> bool:
        """True iff this result carries a usable 0/1 score."""
        return self.score is not None
