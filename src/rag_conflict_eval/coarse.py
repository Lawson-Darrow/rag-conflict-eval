"""Coarse conflict detector for `auto` mode (the false-consensus wedge).

The fine 5-way CONFLICTS taxonomy is too subtle to detect reliably from snippets
(complementary vs conflicting-opinions collapses; macro-F1 ~0.36). For `auto`
mode we only need to know whether the sources MATERIALLY DISAGREE — which is
exactly the false-consensus signal. This detector predicts one of:

    no_material_issue · source_divergence · temporal_supersession   (+ abstain)

It extracts per-source claims first, returns a confidence, and abstention is
gated OUTSIDE the model (label == uncertain, or confidence < threshold) so we
can sweep thresholds without re-calling the API. The 5-way detector (detector.py)
stays for oracle mode.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .prompts.templates import (
    COARSE_DETECTION_PROMPT_VERSION,
    build_coarse_detection_prompt,
    build_coarse_repair_message,
)
from .taxonomy import ConflictType
from .types import ConflictInstance, JudgedTextField, ResultStatus

DetectorFn = Callable[[list[dict]], str]

ABSTAIN = "abstain"
ERROR = "error"


class CoarseConflict(str, Enum):
    NO_MATERIAL_ISSUE = "no_material_issue"
    SOURCE_DIVERGENCE = "source_divergence"
    TEMPORAL_SUPERSESSION = "temporal_supersession"


#: Positive class for the headline "is there a conflict?" binary metric.
CONFLICT_CLASSES = (CoarseConflict.SOURCE_DIVERGENCE, CoarseConflict.TEMPORAL_SUPERSESSION)

#: Map the gold 5-way label to its coarse equivalent for benchmarking.
FINE_TO_COARSE: dict[ConflictType, CoarseConflict] = {
    ConflictType.NO_CONFLICT: CoarseConflict.NO_MATERIAL_ISSUE,
    ConflictType.COMPLEMENTARY: CoarseConflict.SOURCE_DIVERGENCE,
    ConflictType.CONFLICTING_OPINIONS: CoarseConflict.SOURCE_DIVERGENCE,
    ConflictType.FRESHNESS: CoarseConflict.TEMPORAL_SUPERSESSION,
    # Misinformation = sources conflict where one is false -> a divergence for
    # routing purposes (its special handling is an oracle-mode concern).
    ConflictType.MISINFORMATION: CoarseConflict.SOURCE_DIVERGENCE,
}

_LABEL_TO_COARSE = {c.value: c for c in CoarseConflict}


@dataclass
class CoarseDetectionResult:
    status: ResultStatus
    instance_id: Optional[str] = None
    #: The model's label; None when it said "uncertain" or on error.
    label: Optional[CoarseConflict] = None
    said_uncertain: bool = False
    confidence: Optional[float] = None
    can_all_claims_be_true: Optional[Any] = None
    is_temporal_supersession: Optional[Any] = None
    claims: Optional[list] = None
    detector_model: Optional[str] = None
    prompt_version: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None


def parse_coarse(text: str):
    """Parse a coarse-detector reply. Returns a dict with keys label(str|None),
    confidence, can_all_claims_be_true, is_temporal_supersession, claims — or None
    if no valid label is recoverable."""
    if not text:
        return None
    candidates = [text.strip()]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    valid = set(_LABEL_TO_COARSE) | {"uncertain"}
    for cand in candidates:
        c = cand
        if c.startswith("```"):
            c = c.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or obj.get("label") not in valid:
            continue
        conf = obj.get("confidence")
        return {
            "label": obj["label"],
            "confidence": float(conf) if isinstance(conf, (int, float)) else None,
            "can_all_claims_be_true": obj.get("can_all_claims_be_true"),
            "is_temporal_supersession": obj.get("is_temporal_supersession"),
            "claims": obj.get("claims"),
        }
    return None


class CoarseConflictDetector:
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

    def _result(self, instance, **kw) -> CoarseDetectionResult:
        return CoarseDetectionResult(
            instance_id=instance.id,
            detector_model=self.model,
            prompt_version=COARSE_DETECTION_PROMPT_VERSION,
            **kw,
        )

    def detect(self, instance: ConflictInstance) -> CoarseDetectionResult:
        messages = build_coarse_detection_prompt(
            instance, judged_text_field=self.judged_text_field
        )
        try:
            raw = self._call(messages)
        except Exception as e:
            return self._result(instance, status=ResultStatus.JUDGE_ERROR, error=repr(e))

        parsed = parse_coarse(raw)
        if parsed is None:
            repair = messages + [
                {"role": "assistant", "content": raw},
                build_coarse_repair_message(raw),
            ]
            try:
                raw2 = self._call(repair)
            except Exception as e:
                return self._result(
                    instance, status=ResultStatus.JUDGE_ERROR, error=repr(e), raw_output=raw
                )
            parsed = parse_coarse(raw2)
            if parsed is None:
                return self._result(
                    instance,
                    status=ResultStatus.PARSE_ERROR,
                    error="coarse detector returned no valid label after repair retry",
                    raw_output=f"[attempt 1]\n{raw}\n\n[attempt 2 / repair]\n{raw2}",
                )
            raw = raw2

        label_str = parsed["label"]
        said_uncertain = label_str == "uncertain"
        return self._result(
            instance,
            status=ResultStatus.OK,
            label=None if said_uncertain else _LABEL_TO_COARSE[label_str],
            said_uncertain=said_uncertain,
            confidence=parsed["confidence"],
            can_all_claims_be_true=parsed["can_all_claims_be_true"],
            is_temporal_supersession=parsed["is_temporal_supersession"],
            claims=parsed["claims"],
            raw_output=raw,
        )


def resolve(result: CoarseDetectionResult, threshold: float) -> Optional[CoarseConflict]:
    """Final prediction at a confidence threshold, or None to abstain.

    Abstains when: errored, the model said uncertain, or confidence is below the
    threshold (or missing, which we treat as not-confident-enough)."""
    if result.status is not ResultStatus.OK:
        return None
    if result.said_uncertain:
        return None
    if result.confidence is None or result.confidence < threshold:
        return None
    return result.label


# --- benchmark (one API run, swept over thresholds) ---

def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    if not p or not r:
        f1 = 0.0 if (tp + fp + fn) else None
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1


@dataclass
class CoarseThresholdReport:
    threshold: float
    per_class: dict = field(default_factory=dict)   # CoarseConflict -> {p,r,f1,support}
    macro_f1: Optional[float] = None
    confusion: dict = field(default_factory=dict)    # (gold, pred_key) -> count
    n_total: int = 0
    n_predicted: int = 0
    n_correct: int = 0
    n_abstained: int = 0
    n_error: int = 0
    binary: dict = field(default_factory=dict)       # tp/fp/fn/tn/precision/recall/f1/accuracy

    @property
    def coverage(self) -> float:
        return self.n_predicted / self.n_total if self.n_total else 0.0

    @property
    def accuracy_on_covered(self) -> Optional[float]:
        return self.n_correct / self.n_predicted if self.n_predicted else None

    @property
    def abstention_rate(self) -> float:
        return self.n_abstained / self.n_total if self.n_total else 0.0

    @property
    def error_rate(self) -> float:
        return self.n_error / self.n_total if self.n_total else 0.0


@dataclass
class CoarseBenchmark:
    golds: list = field(default_factory=list)
    results: list = field(default_factory=list)
    by_threshold: dict = field(default_factory=dict)   # threshold -> CoarseThresholdReport


def _score_at_threshold(golds, results, threshold: float) -> CoarseThresholdReport:
    rep = CoarseThresholdReport(threshold=threshold)
    classes = list(CoarseConflict)
    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in classes}
    b = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for gold, res in zip(golds, results):
        rep.n_total += 1
        counts[gold]["support"] += 1
        gold_pos = gold in CONFLICT_CLASSES

        if res.status is not ResultStatus.OK:
            pred = None
            key = ERROR
            rep.n_error += 1
        else:
            pred = resolve(res, threshold)
            if pred is None:
                key = ABSTAIN
                rep.n_abstained += 1
            else:
                key = pred.value
                rep.n_predicted += 1

        if pred is not None:
            if pred is gold:
                rep.n_correct += 1
                counts[gold]["tp"] += 1
            else:
                counts[pred]["fp"] += 1
                counts[gold]["fn"] += 1
            pred_pos = pred in CONFLICT_CLASSES
        else:  # abstain or error -> missed positive for gold class
            counts[gold]["fn"] += 1
            pred_pos = False

        # binary "is there a conflict?"
        if gold_pos and pred_pos:
            b["tp"] += 1
        elif not gold_pos and pred_pos:
            b["fp"] += 1
        elif gold_pos and not pred_pos:
            b["fn"] += 1
        else:
            b["tn"] += 1

        rep.confusion[(gold, key)] = rep.confusion.get((gold, key), 0) + 1

    f1s = []
    for c in classes:
        p, r, f1 = _prf(counts[c]["tp"], counts[c]["fp"], counts[c]["fn"])
        rep.per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": counts[c]["support"]}
        if f1 is not None:
            f1s.append(f1)
    rep.macro_f1 = sum(f1s) / len(f1s) if f1s else None

    bp, br, bf1 = _prf(b["tp"], b["fp"], b["fn"])
    rep.binary = {
        **b,
        "precision": bp,
        "recall": br,
        "f1": bf1,
        "accuracy": (b["tp"] + b["tn"]) / rep.n_total if rep.n_total else None,
    }
    return rep


def benchmark_coarse_detector(
    detector, instances, *, thresholds=(0.6, 0.7, 0.8)
) -> CoarseBenchmark:
    """Run the coarse detector once per instance, then score at each threshold.

    One API pass produces the full coverage-vs-accuracy sweep. ``binary`` reports
    the headline false-consensus detection (conflict vs no_material_issue)."""
    golds = [FINE_TO_COARSE[i.conflict_type] for i in instances]
    results = [detector.detect(i) for i in instances]
    by_threshold = {t: _score_at_threshold(golds, results, t) for t in thresholds}
    return CoarseBenchmark(golds=golds, results=results, by_threshold=by_threshold)


__all__ = [
    "CoarseConflict",
    "CoarseConflictDetector",
    "CoarseDetectionResult",
    "CoarseThresholdReport",
    "CoarseBenchmark",
    "FINE_TO_COARSE",
    "CONFLICT_CLASSES",
    "parse_coarse",
    "resolve",
    "benchmark_coarse_detector",
    "ABSTAIN",
    "ERROR",
]
