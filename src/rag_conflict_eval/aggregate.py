"""Aggregation of per-item :class:`AdherenceResult` into reportable rates.

This module owns the denominator contract so no consumer has to invent it.
The headline ``adherence_rate`` is::

    adhered / scored        where scored = #(score in {0, 1})

i.e. UNCERTAIN abstentions and operational errors are EXCLUDED from both the
numerator and the denominator. Their counts are surfaced separately so a
consumer can compute an alternate denominator if they want, but the default
number is reproducible and unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .taxonomy import ConflictType
from .types import AdherenceResult, ResultStatus


@dataclass
class TypeReport:
    conflict_type: ConflictType
    n_total: int = 0
    n_scored: int = 0          # score in {0, 1}
    n_adhered: int = 0         # score == 1
    n_uncertain: int = 0       # status OK, label UNCERTAIN
    n_parse_error: int = 0
    n_judge_error: int = 0

    @property
    def adherence_rate(self) -> Optional[float]:
        """Adhered / scored. ``None`` when nothing was scored."""
        return self.n_adhered / self.n_scored if self.n_scored else None

    @property
    def uncertain_rate(self) -> Optional[float]:
        """Uncertain / (judge returned a verdict). ``None`` when no verdicts."""
        verdicts = self.n_scored + self.n_uncertain
        return self.n_uncertain / verdicts if verdicts else None

    @property
    def error_rate(self) -> Optional[float]:
        """(parse + judge errors) / total. ``None`` when empty."""
        return (self.n_parse_error + self.n_judge_error) / self.n_total if self.n_total else None


@dataclass
class AdherenceReport:
    per_type: dict[ConflictType, TypeReport] = field(default_factory=dict)
    n_total: int = 0
    n_scored: int = 0
    n_adhered: int = 0
    n_uncertain: int = 0
    n_parse_error: int = 0
    n_judge_error: int = 0

    @property
    def adherence_rate(self) -> Optional[float]:
        return self.n_adhered / self.n_scored if self.n_scored else None

    @property
    def uncertain_rate(self) -> Optional[float]:
        verdicts = self.n_scored + self.n_uncertain
        return self.n_uncertain / verdicts if verdicts else None

    @property
    def error_rate(self) -> Optional[float]:
        return (self.n_parse_error + self.n_judge_error) / self.n_total if self.n_total else None


def aggregate(results: Iterable[AdherenceResult]) -> AdherenceReport:
    """Roll per-item results into an overall + per-type :class:`AdherenceReport`."""
    report = AdherenceReport()
    for r in results:
        tr = report.per_type.setdefault(r.conflict_type, TypeReport(conflict_type=r.conflict_type))
        report.n_total += 1
        tr.n_total += 1

        if r.status is ResultStatus.PARSE_ERROR:
            report.n_parse_error += 1
            tr.n_parse_error += 1
            continue
        if r.status is ResultStatus.JUDGE_ERROR:
            report.n_judge_error += 1
            tr.n_judge_error += 1
            continue

        # status OK: either a 0/1 score or an UNCERTAIN abstention
        if r.scored:
            report.n_scored += 1
            tr.n_scored += 1
            if r.score == 1:
                report.n_adhered += 1
                tr.n_adhered += 1
        else:
            report.n_uncertain += 1
            tr.n_uncertain += 1

    return report
