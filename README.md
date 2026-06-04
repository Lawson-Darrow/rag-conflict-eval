# rag-conflict-eval

[![CI](https://github.com/Lawson-Darrow/rag-conflict-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Lawson-Darrow/rag-conflict-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**Catch RAG answers that mishandle conflicting retrieved evidence — when your system invents a consensus that isn't there, or trusts a stale source over a current one.**

> ⚠️ Pre-alpha. Interfaces and prompts will change. The behavior rater is a
> *reconstruction* of the rubric from Cattan et al. (2025) and is **not yet
> human-validated**. Numbers below are preliminary.

## The problem

Your RAG pipeline retrieves two sources that disagree — one says revenue was $4.1B,
the other (audited, three months later) says $3.8B. The model reads both, **picks
one, and reports it at 80% confidence. Silently. Confidently.** That's *false
consensus*, and standard RAG eval doesn't catch it: a faithfulness metric checks
the answer against the retrieved text and happily passes — the number *was* in the
context. You can score 0.95 faithfulness and still be wrong.

Mainstream eval (RAGAS, DeepEval, TruLens, Phoenix) checks **grounding** and
**correctness**. Neither checks whether the answer **handled the disagreement
between sources correctly**. `rag-conflict-eval` does.

## What it does

Two things, kept separate:

1. **Detect** whether the retrieved sources materially disagree — `no_material_issue`
   / `source_divergence` / `temporal_supersession` (stale-vs-current). Runs on your
   own RAG traces, no gold labels needed.
2. **Score behavior** — given the conflict, did the answer behave correctly? (Don't
   manufacture consensus; reflect genuine disagreement; prefer current over stale.)

## See it catch a bug

Point it at your own RAG traces (`{question, contexts, answer}`):

```bash
rag-conflict-eval auto traces.jsonl
```

Given a trace where two sources disagree (a preliminary $4.1B figure vs. an audited
$3.8B that supersedes it) and the answer just reports the stale one:

```
scanned 3 traces - 2 need review
  ok=1 false_consensus_risk=1 stale_source_risk=1 unclear=0

[stale_source_risk] trace revenue  (conflict: temporal_supersession)
    why: The answer relied on an outdated preliminary report instead of the newer
         audited 10-Q filing that supersedes it.
    answer span: "Acme's Q3 revenue was $4.1B."
    source claims: [{"source": 1, "answer_value": "4.1B", "source_date": "Oct"},
                    {"source": 2, "answer_value": "3.8B", "source_date": "Dec, supersedes"}]

[false_consensus_risk] trace coffee  (conflict: source_divergence)
    why: The answer states coffee is good for health without acknowledging the
         conflicting evidence in the sources.
    answer span: "Yes, coffee is good for your health."
```

It's a **diagnostic / linter** (recall-heavy — it flags and lets you triage with the
evidence), not an autonomous pass/fail gate.

## Does it work?

Conflict detection, benchmarked on the **full** Apache-2.0 [CONFLICTS dataset](https://github.com/google-research-datasets/rag_conflicts)
(all 458 instances, gpt-4o-mini):

| "Is there a conflict the answer must handle?" | |
|---|---|
| **Recall** (catches real conflicts) | **0.89** |
| **Precision** | **0.75** |
| **F1** | **0.81** |

It catches ~89% of genuine source conflicts. It leans hard toward flagging
(precision 0.75 — it over-flags clean cases roughly half the time), which is a
deliberately safe direction: better to flag and let the behavior scorer adjudicate
than to miss a false consensus. 3-way detection macro-F1 is 0.52.

**Honest caveats:** the behavior rater is an unvalidated reconstruction;
`temporal_supersession` (stale-vs-current) detection is still weak (~0.23 recall);
it over-flags clean cases; confidence-based abstention isn't calibrated yet. See
`DESIGN.md` / `VALIDATION.md`.

## Install

```bash
pip install -e .                 # core (litellm)
pip install -e ".[deepeval]"     # + DeepEval adapter
```

Grab `conflicts.jsonl` from the [CONFLICTS dataset](https://github.com/google-research-datasets/rag_conflicts) (not vendored).

## Quickstart

```bash
# Benchmark the conflict detector against gold labels
rag-conflict-eval bench-coarse conflicts.jsonl --sample 50

# Score behavior adherence (oracle mode: gold conflict type known)
rag-conflict-eval score answers.jsonl --model gpt-4o-mini
```

```python
from rag_conflict_eval import load_conflicts, BehaviorAdherenceScorer, aggregate
from rag_conflict_eval.coarse import CoarseConflictDetector

instances = load_conflicts("conflicts.jsonl")

# Detect conflicts on your own traces (no gold labels)
det = CoarseConflictDetector(model="gpt-4o-mini")
print(det.detect(instances[0]).label)        # source_divergence / temporal_supersession / ...

# Score whether an answer handled a known conflict correctly (oracle mode)
scorer = BehaviorAdherenceScorer(model="gpt-4o-mini")
print(scorer.score(instances[0], "As of 2024 the CEO is John Doe.").label)
```

## Integrations

Drop it into the eval framework you already use:

```python
# RAGAS — a false-consensus safety metric on a standard sample (higher = safer)
from ragas import SingleTurnSample
from rag_conflict_eval.adapters.ragas import FalseConsensusSafety
score = FalseConsensusSafety().single_turn_score(
    SingleTurnSample(user_input="...", retrieved_contexts=[...], response="...")
)  # 1.0 ok · 0.0 risk · NaN unclear

# DeepEval — behavior-adherence metric (oracle mode; gold conflict_type in metadata)
from deepeval.test_case import LLMTestCase
from rag_conflict_eval.adapters.deepeval import BehaviorAdherenceMetric
m = BehaviorAdherenceMetric()
m.measure(LLMTestCase(input="...", actual_output="...", retrieval_context=[...],
                      metadata={"conflict_type": "freshness"}))
```

Install the extra you need: `pip install -e ".[ragas]"` or `".[deepeval]"`.

## Two modes

- **Oracle** — you supply the gold conflict type; the clean, trusted measurement.
  Uses the full 5-type taxonomy from the paper.
- **Auto** (in progress) — the detector predicts the conflict, so it runs on a
  dev's own RAG output. Coarser (3-way) by design; numbers reported separately so
  detector errors never pollute the oracle score.

## How the metric is defined

- Behavior verdict is **adhere / not_adhere / uncertain**; uncertain is a real
  abstention, excluded from the headline rate (not counted as failure).
- It is **not correctness** — an answer can adhere yet be factually wrong. Pair it
  with a faithfulness/correctness metric.

## Related work

- **DRAGged into Conflicts** (Cattan et al., [arXiv:2506.08500](https://arxiv.org/abs/2506.08500)) —
  the taxonomy, the behavior-adherence metric, and the CONFLICTS dataset this builds on.
- **ConflictRAG / CARS** ([arXiv:2605.17301](https://arxiv.org/abs/2605.17301)) — a
  conflict *detection/resolution accuracy* score; research-only, different axis.
- **ArbGraph** ([arXiv:2604.18362](https://arxiv.org/abs/2604.18362)) — a conflict
  *resolution* generation method; orthogonal (a system to evaluate *with* this tool).

No mainstream OSS RAG-eval framework ships a reusable conflict-aware behavior
metric — that's the gap this fills. It is **not** "the first conflict-aware RAG eval."

## License

MIT. The CONFLICTS dataset is Apache-2.0 and is not vendored here.
