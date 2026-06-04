# rag-conflict-eval — v1 skeleton design

Companion to the scope doc (`~/Desktop/Projects/rag-conflict-eval-scope.md`).
This is the **build plan** for the v1 skeleton. Files marked IMPLEMENTED exist;
the rest are proposed interfaces I want reviewed **before** I implement them.

## Module map
```
src/rag_conflict_eval/
  taxonomy.py        IMPLEMENTED  5 types + exact expected-behavior rubric
  types.py           IMPLEMENTED  SearchResult, ConflictInstance, AdherenceResult, AdherenceLabel
  loader.py          STUB         CONFLICTS jsonl -> list[ConflictInstance] + stratified split
  prompts/templates.py STUB       per-type judge prompt builder (+ few-shot slots)
  scorer.py          STUB         BehaviorAdherenceScorer (litellm judge) -> AdherenceResult
  classifier.py      STUB         ConflictTypeClassifier (few-shot) + benchmark util (NOT wired into scorer)
  adapters/deepeval.py STUB       BehaviorAdherenceMetric(BaseMetric)
  cli.py             STUB         `rag-conflict-eval score|classify|bench`
```

## Data flow (oracle mode — the v1 headline)
```
CONFLICTS.jsonl --loader--> ConflictInstance(question, search_results, conflict_type=GOLD, correct_answer?)
candidate_answer (from the system under test) +
ConflictInstance.conflict_type (GOLD) + spec_for(type).expected_behavior
    --prompts.templates--> typed judge prompt
    --scorer (litellm judge)--> AdherenceResult{label in adhere/not_adhere/uncertain}
aggregate --> overall + per-type adherence rate (uncertain excluded from denominator)
```
End-to-end mode (classifier predicts the type) is **v2** — kept out so classifier
errors can't contaminate the behavior score.

## Proposed interfaces (REVIEW THESE)

### loader.py
```python
def load_conflicts(path: str | Path) -> list[ConflictInstance]: ...
def stratified_split(
    instances: list[ConflictInstance], *, test_size: float = 0.2, seed: int = 0
) -> tuple[list[ConflictInstance], list[ConflictInstance]]: ...
```
- Maps raw dataset label strings -> ConflictType. **OPEN: exact label strings in
  conflicts.jsonl are unconfirmed** (raw file was too large to fetch). Need to
  inspect the real file and pin the mapping; fail loud on unknown labels.

### prompts/templates.py
```python
PROMPT_VERSION = "v0"  # bump on any wording change; recorded in AdherenceResult
def build_judge_prompt(
    instance: ConflictInstance,
    candidate_answer: str,
    *,
    judged_text_field: str = "short_text",   # snippet | short_text | response_str
    few_shot: bool = True,
) -> list[dict]:   # chat messages
    ...
```
- One template per type, each carrying: query, the type's definition +
  expected_behavior, 2-3 pos/neg few-shot examples, the candidate answer.
- Judge must be allowed to return `uncertain`.

### scorer.py
```python
class BehaviorAdherenceScorer:
    def __init__(self, model: str = "gpt-4o-mini", *, judged_text_field: str = "short_text",
                 temperature: float = 0.0): ...
    def score(self, instance: ConflictInstance, candidate_answer: str) -> AdherenceResult: ...
    def score_batch(self, pairs: Iterable[tuple[ConflictInstance, str]]) -> list[AdherenceResult]: ...
```
- litellm under the hood (provider-agnostic). Parses a strict JSON verdict.
- Records judge_model + prompt_version + judged_text_field on every result.

### classifier.py  (separate utility, NOT in the scoring path)
```python
class ConflictTypeClassifier:
    def __init__(self, model: str = "gpt-4o-mini"): ...
    def predict(self, instance: ConflictInstance) -> ConflictType: ...

def benchmark_classifier(
    clf: ConflictTypeClassifier, instances: list[ConflictInstance]
) -> ClassifierReport:   # macro-F1 + confusion matrix; misinformation reported but excluded
    ...
```

### adapters/deepeval.py
```python
class BehaviorAdherenceMetric(BaseMetric):
    # expects test_case.additional_metadata["conflict_type"] (gold, oracle mode)
    # retrieval_context -> search_results; actual_output -> candidate answer
    def measure(self, test_case: LLMTestCase) -> float: ...
```

## Aggregation / reporting (measurement discipline)
Report separately, never collapsed: `oracle_type_score`, (`predicted_type_score` v2),
`type_accuracy`, per-type adherence rates, and the uncertain rate. Behavior
adherence is NOT correctness — keep them distinct in all outputs.

## Open questions for Codex (design pass)
1. Is the loader/prompts/scorer/classifier split right, or should the judge own
   prompt construction? Any layer that will bite us later?
2. litellm vs a thin provider abstraction for the judge — overkill or correct?
3. `judged_text_field` default: `short_text` (the authors' 512-tok window) vs
   `response_str` (full page). Tradeoff: faithfulness to the dataset vs giving the
   judge enough context to actually see the conflict.
4. DeepEval `BaseMetric` contract: is `additional_metadata["conflict_type"]` the
   right channel for the gold type, or is there a cleaner idiom?
5. Strict-JSON judge output: parse + repair strategy, or function/tool-calling
   for the verdict? Failure handling when the judge won't return valid JSON.
6. Anything in the v1 cut-line that's secretly v2-sized, or vice versa?
