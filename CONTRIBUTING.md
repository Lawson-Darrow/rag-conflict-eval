# Contributing

This is a pre-alpha tool; interfaces and prompts will change. Issues, ideas, and
PRs are welcome — especially around integrations (other eval frameworks, RAG
stacks) and prompt/validation quality.

## Dev setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,deepeval]"
pytest -q          # 62 tests; the deepeval tests skip if the extra is absent
ruff check .
```

The CONFLICTS dataset is not vendored. Grab `conflicts.jsonl` from
<https://github.com/google-research-datasets/rag_conflicts> (Apache-2.0) to run
`rag-conflict-eval stats` / `score` against real data.

## Bar for changes

- **Tests required.** New behavior gets a test; bug fixes get a regression test.
- **Lint clean** (`ruff check .`).
- **Don't conflate the axes.** Behavior adherence is not correctness; the judge
  verdict (adhere/not_adhere/uncertain) is separate from operational status
  (ok/parse_error/judge_error). Uncertain and errors are excluded from the
  headline rate — keep it that way.
- **Prompts are the metric.** Changes to `prompts/templates.py` bump
  `PROMPT_VERSION` and should be justified; ideally validated (see `VALIDATION.md`).
- Read `DESIGN.md` for the architecture and the oracle-mode / experimental-type
  caveats before large changes.

## Scope

See `DESIGN.md` (deferred to v2: RAGAS adapter, end-to-end predicted-type mode,
human-validated rater, leaderboard). New integration adapters are very welcome.
