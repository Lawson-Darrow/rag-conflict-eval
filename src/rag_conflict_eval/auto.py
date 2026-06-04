"""Auto mode: find likely false consensus in a dev's own RAG traces.

This is the product surface — point it at your `{question, contexts, answer}`
traces (no gold labels) and it flags answers that mishandled conflicting sources.
It is a DIAGNOSTIC, not a validated pass/fail metric: the coarse detector is
recall-heavy (over-flags), and the behavior judge here is a narrow false-consensus
check, NOT the oracle behavior-adherence rubric. Output is evidence-rich so a human
can see *why* a trace was flagged.

Pipeline: trace -> coarse detector -> (if a conflict is detected) a narrow
false-consensus / stale-source judge -> AutoResult with the conflicting claims and
the risky answer span.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .coarse import CoarseConflict, CoarseConflictDetector, CoarseDetectionResult
from .prompts.templates import (
    AUTO_BEHAVIOR_PROMPT_VERSION,
    build_auto_behavior_prompt,
    build_auto_repair_message,
)
from .types import JudgedTextField, ResultStatus, SearchResult

JudgeFn = Callable[[list[dict]], str]


class AutoVerdict(str, Enum):
    OK = "ok"
    FALSE_CONSENSUS_RISK = "false_consensus_risk"
    STALE_SOURCE_RISK = "stale_source_risk"
    UNCLEAR = "unclear"


_VERDICT_MAP = {v.value: v for v in AutoVerdict}


def _context_to_sr(c) -> SearchResult:
    if isinstance(c, SearchResult):
        return c
    if isinstance(c, str):
        return SearchResult(short_text=c)
    if isinstance(c, dict):
        return SearchResult(
            title=c.get("title"),
            url=c.get("source") or c.get("url"),
            date=c.get("date"),
            short_text=c.get("text") or c.get("short_text") or c.get("snippet"),
            snippet=c.get("snippet"),
            response_str=c.get("response_str"),
            raw=c,
        )
    return SearchResult(short_text=str(c))


@dataclass
class Trace:
    """One RAG trace to diagnose. ``contexts`` items may be strings or dicts with
    ``text``/``source``/``date``. Exposes ``question``/``search_results``/``id`` so
    the coarse detector consumes it directly."""

    question: str
    contexts: list
    answer: str
    trace_id: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        if self.trace_id:
            return self.trace_id
        return hashlib.sha1(f"{self.question}\x00{self.answer}".encode("utf-8")).hexdigest()[:12]

    @property
    def search_results(self) -> list[SearchResult]:
        return [_context_to_sr(c) for c in self.contexts]


@dataclass
class AutoResult:
    trace_id: str
    detector_status: ResultStatus
    #: Detected coarse conflict; None when no material issue, abstained, or error.
    conflict: Optional[CoarseConflict]
    verdict: AutoVerdict
    needs_review: bool
    rationale: Optional[str] = None
    risky_answer_span: Optional[str] = None
    #: Status of the behavior judge call (None if it wasn't run).
    behavior_status: Optional[ResultStatus] = None
    #: Detector evidence: answer_slot, per-source claims, newer_supersedes, snippets.
    evidence: dict = field(default_factory=dict)


def _parse_auto(text: str):
    """Parse the narrow judge reply -> (verdict, rationale, span) or None."""
    if not text:
        return None
    candidates = [text.strip()]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        c = cand
        if c.startswith("```"):
            c = c.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or obj.get("verdict") not in _VERDICT_MAP:
            continue
        span = obj.get("risky_answer_span")
        return (
            _VERDICT_MAP[obj["verdict"]],
            obj.get("rationale") if isinstance(obj.get("rationale"), str) else None,
            span if isinstance(span, str) else None,
        )
    return None


class AutoBehaviorJudge:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        judged_text_field: JudgedTextField = "short_text",
        temperature: float = 0.0,
        judge_fn: Optional[JudgeFn] = None,
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
            model=self.model, messages=messages, temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def judge(self, trace: Trace, detected_conflict: str):
        """Return (status, verdict, rationale, span). verdict/span None unless OK."""
        messages = build_auto_behavior_prompt(
            trace, trace.answer, detected_conflict, judged_text_field=self.judged_text_field
        )
        try:
            raw = self._call(messages)
        except Exception as e:
            return ResultStatus.JUDGE_ERROR, None, repr(e), None
        parsed = _parse_auto(raw)
        if parsed is None:
            repair = messages + [{"role": "assistant", "content": raw}, build_auto_repair_message(raw)]
            try:
                raw2 = self._call(repair)
            except Exception as e:
                return ResultStatus.JUDGE_ERROR, None, repr(e), None
            parsed = _parse_auto(raw2)
            if parsed is None:
                return ResultStatus.PARSE_ERROR, None, "no valid verdict after repair", None
        verdict, rationale, span = parsed
        return ResultStatus.OK, verdict, rationale, span


_CONFLICT_LABELS = (CoarseConflict.SOURCE_DIVERGENCE, CoarseConflict.TEMPORAL_SUPERSESSION)


class AutoPipeline:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        judged_text_field: JudgedTextField = "short_text",
        detector: Optional[CoarseConflictDetector] = None,
        judge: Optional[AutoBehaviorJudge] = None,
    ) -> None:
        self.detector = detector or CoarseConflictDetector(
            model=model, judged_text_field=judged_text_field
        )
        self.judge = judge or AutoBehaviorJudge(model=model, judged_text_field=judged_text_field)

    def _evidence(self, det: CoarseDetectionResult, trace: Trace) -> dict:
        sig = det.signals or {}
        return {
            "answer_slot": sig.get("answer_slot"),
            "claims": sig.get("claims"),
            "newer_supersedes": sig.get("newer_supersedes"),
            "sources": [
                {"title": s.title, "date": s.date, "text": s.text(self.detector.judged_text_field)}
                for s in trace.search_results
            ],
        }

    def run(self, trace: Trace) -> AutoResult:
        det = self.detector.detect(trace)
        evidence = self._evidence(det, trace)

        if det.status is not ResultStatus.OK:
            return AutoResult(
                trace_id=trace.id, detector_status=det.status, conflict=None,
                verdict=AutoVerdict.UNCLEAR, needs_review=True,
                rationale="conflict detection failed", evidence=evidence,
            )

        label = det.label
        if label is None or label not in _CONFLICT_LABELS:
            # no material conflict (or abstained) -> nothing to mishandle
            return AutoResult(
                trace_id=trace.id, detector_status=ResultStatus.OK, conflict=label,
                verdict=AutoVerdict.OK, needs_review=False,
                rationale="no material source conflict detected", evidence=evidence,
            )

        status, verdict, rationale, span = self.judge.judge(trace, label.value)
        if status is not ResultStatus.OK:
            return AutoResult(
                trace_id=trace.id, detector_status=ResultStatus.OK, conflict=label,
                verdict=AutoVerdict.UNCLEAR, needs_review=True,
                rationale="behavior judge failed", behavior_status=status, evidence=evidence,
            )
        return AutoResult(
            trace_id=trace.id, detector_status=ResultStatus.OK, conflict=label,
            verdict=verdict, needs_review=verdict is not AutoVerdict.OK,
            rationale=rationale, risky_answer_span=span, behavior_status=status, evidence=evidence,
        )

    def run_batch(self, traces) -> list[AutoResult]:
        return [self.run(t) for t in traces]


def load_traces(path) -> list[Trace]:
    """Load a JSONL of traces: {question, contexts, answer, trace_id?}."""
    traces = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("question", "contexts", "answer"):
                if key not in rec:
                    raise ValueError(f"trace at line {line_no} missing '{key}'")
            traces.append(Trace(
                question=rec["question"], contexts=rec["contexts"], answer=rec["answer"],
                trace_id=rec.get("trace_id"), raw=rec,
            ))
    return traces


__all__ = [
    "Trace",
    "AutoVerdict",
    "AutoResult",
    "AutoBehaviorJudge",
    "AutoPipeline",
    "load_traces",
    "AUTO_BEHAVIOR_PROMPT_VERSION",
]
