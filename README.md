# rag-conflict-eval

**Conflict-type-conditioned behavior-adherence evaluation for RAG.**

> ⚠️ Pre-alpha. Interfaces and prompts are unstable. The judge is a
> *reconstruction* of the rubric from Cattan et al. (2025), not the authors'
> original implementation, and the rater is not yet human-validated.

## What it scores

Mainstream RAG eval checks two things: is the answer **grounded** (faithfulness)
and is it **correct** (answer recall). Neither captures a third axis: *given the
way the retrieved documents conflict, did the answer behave the way it should?*

An answer can be grounded **and** match the gold answer yet still behave wrong.
Example: two sources disagree on a number because one is outdated and one is
current. An answer that says "sources disagree, it's between X and Y" is grounded
but wrong behavior — for a **freshness** conflict it should give the current
figure, not stage a debate.

`rag-conflict-eval` scores **behavior adherence conditioned on conflict type**,
using the five-type taxonomy and per-type expected behaviors from the CONFLICTS
work, packaged as a reusable metric (with a DeepEval adapter).

| Conflict type | Expected behavior (the rubric) |
|---|---|
| No conflict | Direct answer, no manufactured uncertainty |
| Complementary | Consolidate the partial answers, don't frame as a debate |
| Conflicting opinions | Reflect the debate, summarize sides neutrally |
| Freshness (outdated) | Prefer current info, optionally note the stale source |
| Misinformation | Disregard false sources, ground in reliable info *(experimental, n=5)* |

## Install

```bash
pip install -e .                 # core (litellm judge)
pip install -e ".[deepeval]"     # + DeepEval adapter
```

Get the Apache-2.0 CONFLICTS dataset (not vendored):
<https://github.com/google-research-datasets/rag_conflicts> (`conflicts.jsonl`).

## Quickstart

### Library

```python
from rag_conflict_eval import (
    load_conflicts, BehaviorAdherenceScorer, aggregate,
)

instances = load_conflicts("conflicts.jsonl")
scorer = BehaviorAdherenceScorer(model="gpt-4o-mini")   # any litellm model

# Score your RAG system's answer for one instance (oracle mode: gold type known)
inst = instances[0]
result = scorer.score(inst, candidate_answer="As of 2024 the CEO is John Doe.")
print(result.status, result.label, result.score)   # OK ADHERE 1

# Aggregate over many (uncertain + errors excluded from the headline;
# Misinformation experimental -> excluded by default)
report = aggregate(scorer.score_batch([(i, my_answer(i)) for i in instances]))
print(report.adherence_rate, report.per_type)
```

### CLI

```bash
# Inspect the dataset
rag-conflict-eval stats conflicts.jsonl

# Score a JSONL of CONFLICTS records, each with an extra "candidate_answer" field
rag-conflict-eval score answers.jsonl --model gpt-4o-mini --out results.jsonl
```

### DeepEval

```python
from deepeval.test_case import LLMTestCase
from rag_conflict_eval.adapters.deepeval import BehaviorAdherenceMetric, aggregate_metrics

tc = LLMTestCase(
    input="Who is the CEO?",
    actual_output="As of 2024 the CEO is John Doe.",
    retrieval_context=[...],
    metadata={"conflict_type": "freshness"},   # gold type (oracle mode)
)
metric = BehaviorAdherenceMetric()
metric.measure(tc)

# IMPORTANT: for the honest rate, aggregate metric.result yourself —
# DeepEval's pass-rate counts uncertain/errored cases as failures.
report = aggregate_metrics([metric])
```

## How the score is defined

- The judge returns **adhere / not_adhere / uncertain** (an LLM-as-judge over a
  per-type prompt). `uncertain` is a real abstention, not a failure.
- `adherence_rate = adhered / scored`, where `scored` counts only 0/1 verdicts —
  **uncertain abstentions and judge/parse errors are excluded from the
  denominator** (their counts are reported separately).
- This is **oracle mode**: the gold conflict type is supplied. That's intended
  ("behavior adherence conditioned on type") but it primes the judge — a blind
  judge is the calibration story (see `DESIGN.md`).
- It is **not** correctness. An answer can adhere yet be factually wrong. Pair it
  with a faithfulness/correctness metric.

## Related work

- **DRAGged into Conflicts** (Cattan et al., arXiv:2506.08500) — the taxonomy,
  the Expected Behavior Adherence metric, and the [CONFLICTS dataset](https://github.com/google-research-datasets/rag_conflicts)
  (Apache-2.0) this tool builds on.
- **ConflictRAG / CARS** (arXiv:2605.17301) — a "Conflict-Aware RAG Score" that
  measures conflict *detection/resolution accuracy* end-to-end; research-only, no
  released tooling. Different axis from per-type behavior adherence.
- **ArbGraph** (arXiv:2604.18362) — an evidence-arbitration *generation* method;
  orthogonal — a candidate system to evaluate *with* this tool, not a competitor.

This is a **tooling wedge**: no mainstream OSS RAG-eval framework (RAGAS,
DeepEval, TruLens, Arize Phoenix, Vectara open-rag-eval) ships a CONFLICTS-style,
conflict-type-conditioned behavior-adherence metric as reusable tooling. It is
**not** "the first conflict-aware RAG eval."

## Status

Implemented: taxonomy, loader, aggregation, judge prompts (v0) + scorer, DeepEval
adapter, CLI (`stats`, `score`). 62 tests. Deferred to v2: RAGAS adapter,
end-to-end mode (predicted type), human validation set, leaderboard. See
`DESIGN.md`.

## License

MIT. The CONFLICTS dataset is Apache-2.0 and is not vendored here.
