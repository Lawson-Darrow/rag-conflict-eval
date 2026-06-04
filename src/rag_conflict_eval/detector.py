"""Conflict-type detector + gold benchmark.

The detector predicts which conflict type the retrieved sources exhibit, WITHOUT
seeing the gold label, and abstains ("uncertain") rather than guessing. It is the
engine behind `auto` mode (scoring on a dev's own traces, no gold labels). It is
NOT wired into the behavior-adherence headline metric — auto-mode numbers are
reported separately so detector errors never pollute the oracle score.

``benchmark_detector`` runs it against the 458 gold-labeled CONFLICTS instances
and reports macro-F1 + a confusion matrix + abstention/coverage — a real number
that needs no human annotation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .prompts.templates import (
    DETECTION_PROMPT_VERSION,
    build_detection_prompt,
    build_detection_repair_message,
)
from .taxonomy import ConflictType, spec_for
from .types import ConflictInstance, JudgedTextField, ResultStatus

DetectorFn = Callable[[list[dict]], str]

#: Pseudo-columns in the confusion matrix for non-predictions.
ABSTAIN = "abstain"
ERROR = "error"

_LABEL_TO_TYPE = {t.value: t for t in ConflictType}


@dataclass
class DetectionResult:
    status: ResultStatus
    instance_id: Optional[str] = None
    #: Predicted type; None when abstained or errored.
    predicted: Optional[ConflictType] = None
    #: True when the detector returned a valid verdict of "uncertain".
    abstained: bool = False
    rationale: Optional[str] = None
    detector_model: Optional[str] = None
    prompt_version: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None


def parse_detection(text: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a detector reply into (verdict, rationale). ``verdict`` is one of the
    5 type strings or "uncertain"; returns (None, None) if no valid verdict is
    recoverable. Lenient JSON extraction (fenced or embedded), like the rater."""
    if not text:
        return None, None
    candidates = [text.strip()]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    valid = set(_LABEL_TO_TYPE) | {"uncertain"}
    for candidate in candidates:
        c = candidate
        if c.startswith("```"):
            c = c.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        verdict = obj.get("conflict_type")
        if verdict not in valid:
            continue
        rationale = obj.get("rationale")
        return verdict, (rationale if isinstance(rationale, str) else None)
    return None, None


class ConflictTypeDetector:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        judged_text_field: JudgedTextField = "short_text",
        temperature: float = 0.0,
        judge_fn: Optional[DetectorFn] = None,
    ) -> None:
        self.model = model
        self.judged_text_field = judged_text_field
        self.temperature = temperature
        self._judge_fn = judge_fn

    def _call(self, messages: list[dict]) -> str:
        if self._judge_fn is not None:
            return self._judge_fn(messages)
        import litellm

        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def _result(self, instance, **kw) -> DetectionResult:
        return DetectionResult(
            instance_id=instance.id,
            detector_model=self.model,
            prompt_version=DETECTION_PROMPT_VERSION,
            **kw,
        )

    def detect(self, instance: ConflictInstance) -> DetectionResult:
        messages = build_detection_prompt(instance, judged_text_field=self.judged_text_field)
        try:
            raw = self._call(messages)
        except Exception as e:
            return self._result(instance, status=ResultStatus.JUDGE_ERROR, error=repr(e))

        verdict, rationale = parse_detection(raw)
        if verdict is None:
            repair = messages + [
                {"role": "assistant", "content": raw},
                build_detection_repair_message(raw),
            ]
            try:
                raw2 = self._call(repair)
            except Exception as e:
                return self._result(
                    instance, status=ResultStatus.JUDGE_ERROR, error=repr(e), raw_output=raw
                )
            verdict, rationale = parse_detection(raw2)
            if verdict is None:
                return self._result(
                    instance,
                    status=ResultStatus.PARSE_ERROR,
                    error="detector returned no valid conflict_type after repair retry",
                    raw_output=f"[attempt 1]\n{raw}\n\n[attempt 2 / repair]\n{raw2}",
                )
            raw = raw2

        if verdict == "uncertain":
            return self._result(
                instance, status=ResultStatus.OK, abstained=True, rationale=rationale, raw_output=raw
            )
        return self._result(
            instance,
            status=ResultStatus.OK,
            predicted=_LABEL_TO_TYPE[verdict],
            rationale=rationale,
            raw_output=raw,
        )


# --- gold benchmark ---

@dataclass
class ClassMetrics:
    conflict_type: ConflictType
    support: int = 0      # gold instances of this type
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if not p or not r:
            return 0.0 if (self.support or self.tp + self.fp) else None
        return 2 * p * r / (p + r)


@dataclass
class DetectorReport:
    #: confusion[(gold_type, predicted_key)] -> count; predicted_key is a type
    #: value, "abstain", or "error".
    confusion: dict[tuple[ConflictType, str], int] = field(default_factory=dict)
    per_class: dict[ConflictType, ClassMetrics] = field(default_factory=dict)
    macro_f1: Optional[float] = None
    macro_classes: list[ConflictType] = field(default_factory=list)
    experimental_excluded: list[ConflictType] = field(default_factory=list)
    n_total: int = 0
    n_predicted: int = 0   # definite (non-abstain, non-error) predictions
    n_correct: int = 0
    n_abstained: int = 0
    n_error: int = 0

    @property
    def accuracy_on_covered(self) -> Optional[float]:
        return self.n_correct / self.n_predicted if self.n_predicted else None

    @property
    def accuracy_strict(self) -> Optional[float]:
        """Correct over ALL items — abstain and error count as wrong."""
        return self.n_correct / self.n_total if self.n_total else None

    @property
    def coverage(self) -> float:
        return self.n_predicted / self.n_total if self.n_total else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.n_abstained / self.n_total if self.n_total else 0.0

    @property
    def error_rate(self) -> float:
        return self.n_error / self.n_total if self.n_total else 0.0


def benchmark_detector(
    detector,
    instances,
    *,
    exclude_experimental: bool = True,
) -> DetectorReport:
    """Run ``detector.detect`` over gold-labeled instances and report metrics.

    Precision/recall/F1 are computed per type; abstentions and errors on a
    gold-c instance count as a missed positive (FN) for c, never as a wrong
    prediction for another class. Macro-F1 averages F1 over non-experimental
    types (Misinformation is reported per-class but excluded from the headline).
    """
    report = DetectorReport()
    per_class: dict[ConflictType, ClassMetrics] = {
        t: ClassMetrics(conflict_type=t) for t in ConflictType
    }
    for inst in instances:
        gold = inst.conflict_type
        res = detector.detect(inst)
        report.n_total += 1
        per_class[gold].support += 1

        if res.status is ResultStatus.OK and res.predicted is not None:
            pred = res.predicted
            key = pred.value
            report.n_predicted += 1
            if pred is gold:
                report.n_correct += 1
                per_class[gold].tp += 1
            else:
                per_class[pred].fp += 1
                per_class[gold].fn += 1
        elif res.status is ResultStatus.OK:  # abstained
            key = ABSTAIN
            report.n_abstained += 1
            per_class[gold].fn += 1
        else:  # parse/judge error
            key = ERROR
            report.n_error += 1
            per_class[gold].fn += 1

        report.confusion[(gold, key)] = report.confusion.get((gold, key), 0) + 1

    report.per_class = per_class
    excluded = [t for t in ConflictType if spec_for(t).experimental] if exclude_experimental else []
    macro_classes = [t for t in ConflictType if t not in excluded]
    f1s = [per_class[t].f1 for t in macro_classes if per_class[t].f1 is not None]
    report.macro_f1 = sum(f1s) / len(f1s) if f1s else None
    report.macro_classes = macro_classes
    report.experimental_excluded = excluded
    return report


__all__ = [
    "ConflictTypeDetector",
    "DetectionResult",
    "DetectorReport",
    "ClassMetrics",
    "DetectorFn",
    "parse_detection",
    "benchmark_detector",
    "ABSTAIN",
    "ERROR",
]
