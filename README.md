# rag-conflict-eval

**Conflict-type-conditioned behavior-adherence evaluation for RAG.**

> ⚠️ Pre-alpha skeleton. Interfaces and prompts are unstable. The judge is a
> *reconstruction* of the rubric from Cattan et al. (2025), not the authors'
> original implementation.

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

## Scope of the claim

This is a **tooling wedge**: no mainstream OSS RAG-eval framework (RAGAS,
DeepEval, TruLens, Arize Phoenix, Vectara open-rag-eval) ships a CONFLICTS-style,
conflict-type-conditioned behavior-adherence metric as reusable tooling. It is
**not** "the first conflict-aware RAG eval" — see Related work.

## Related work

- **DRAGged into Conflicts** (Cattan et al., arXiv:2506.08500) — the taxonomy,
  the Expected Behavior Adherence metric, and the [CONFLICTS dataset](https://github.com/google-research-datasets/rag_conflicts)
  (Apache-2.0) this tool builds on.
- **ConflictRAG / CARS** (arXiv:2605.17301) — a "Conflict-Aware RAG Score" that
  measures conflict *detection/resolution accuracy* end-to-end; research-only, no
  released tooling. Different axis from per-type behavior adherence.
- **ArbGraph** (arXiv:2604.18362) — an evidence-arbitration *generation* method
  (resolve conflict before generation); orthogonal — a candidate system to
  evaluate *with* this tool, not a competitor.

## Status

Skeleton only. Implemented: taxonomy + data types. Stubs: loader, judge scorer,
type classifier, DeepEval adapter, CLI. See `DESIGN.md`.

## License

MIT. The CONFLICTS dataset is Apache-2.0 and is not vendored here.
