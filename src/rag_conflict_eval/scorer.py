"""Behavior-adherence scorer: judge a candidate RAG answer against the expected
behavior for its conflict type.

The judge is an LLM (via litellm) returning a strict JSON verdict. A malformed
reply triggers ONE repair retry; if it still fails the result is ``PARSE_ERROR``
(never silently ``uncertain``). Transport/API failures are ``JUDGE_ERROR``. Pass
``judge_fn`` to inject a fake judge in tests (no network).
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, Optional

from .prompts.templates import (
    PROMPT_VERSION,
    VERDICT_SCHEMA,
    build_judge_prompt,
    build_repair_message,
)
from .types import (
    AdherenceLabel,
    AdherenceResult,
    ConflictInstance,
    JudgedTextField,
    ResultStatus,
)

#: A judge takes chat messages and returns the raw assistant text.
JudgeFn = Callable[[list[dict]], str]

_VERDICT_TO_LABEL = {
    "adhere": AdherenceLabel.ADHERE,
    "not_adhere": AdherenceLabel.NOT_ADHERE,
    "uncertain": AdherenceLabel.UNCERTAIN,
}


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -len("```")]
    return t.strip()


def parse_verdict(text: str) -> tuple[Optional[AdherenceLabel], Optional[str]]:
    """Parse a judge reply into (label, rationale). Returns (None, None) if the
    reply is not valid JSON with a known verdict."""
    if not text:
        return None, None
    try:
        obj = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    label = _VERDICT_TO_LABEL.get(obj.get("verdict"))
    if label is None:
        return None, None
    rationale = obj.get("rationale")
    return label, (rationale if isinstance(rationale, str) else None)


class BehaviorAdherenceScorer:
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

    def _call_judge(self, messages: list[dict]) -> str:
        if self._judge_fn is not None:
            return self._judge_fn(messages)
        import litellm  # imported lazily so the package works without it

        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def _result(self, instance, **kw) -> AdherenceResult:
        return AdherenceResult(
            conflict_type=instance.conflict_type,
            instance_id=instance.id,
            judge_model=self.model,
            prompt_version=PROMPT_VERSION,
            judged_text_field=self.judged_text_field,
            **kw,
        )

    def score(self, instance: ConflictInstance, candidate_answer: str) -> AdherenceResult:
        messages = build_judge_prompt(
            instance, candidate_answer, judged_text_field=self.judged_text_field
        )
        # First attempt.
        try:
            raw = self._call_judge(messages)
        except Exception as e:  # transport/API failure
            return self._result(instance, status=ResultStatus.JUDGE_ERROR, error=repr(e))

        label, rationale = parse_verdict(raw)

        # One repair retry on malformed output.
        if label is None:
            repair = messages + [
                {"role": "assistant", "content": raw},
                build_repair_message(raw),
            ]
            try:
                raw2 = self._call_judge(repair)
            except Exception as e:
                return self._result(
                    instance, status=ResultStatus.JUDGE_ERROR, error=repr(e), raw_judge_output=raw
                )
            label, rationale = parse_verdict(raw2)
            if label is None:
                return self._result(
                    instance,
                    status=ResultStatus.PARSE_ERROR,
                    error="judge did not return a valid verdict after repair retry",
                    raw_judge_output=raw2,
                )
            raw = raw2

        return self._result(
            instance,
            status=ResultStatus.OK,
            label=label,
            rationale=rationale,
            raw_judge_output=raw,
        )

    def score_batch(
        self, pairs: Iterable[tuple[ConflictInstance, str]]
    ) -> list[AdherenceResult]:
        """Score (instance, candidate_answer) pairs, preserving input order."""
        return [self.score(inst, ans) for inst, ans in pairs]


__all__ = ["BehaviorAdherenceScorer", "JudgeFn", "parse_verdict"]
