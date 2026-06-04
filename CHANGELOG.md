# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[SemVer](https://semver.org/) (pre-1.0: minor = breaking allowed).

## [Unreleased]

## [0.1.0] - 2026-06-04

Initial pre-alpha. Conflict-type-conditioned behavior-adherence evaluation for RAG.

### Added
- Five-type conflict taxonomy with the per-type expected-behavior rubric
  (from Cattan et al., arXiv:2506.08500).
- `load_conflicts` / `stratified_split`: CONFLICTS dataset loader with exact
  label mapping (verified vs the real 458-record file), fail-loud on unknown
  labels and duplicate ids, deterministic per-type split.
- `BehaviorAdherenceScorer`: LLM-judge (via litellm) returning a strict-JSON
  verdict with one repair retry; injectable `judge_fn` for testing.
- Judge prompts (`PROMPT_VERSION = "v0"`): one per type, with prompt-injection
  neutralization of untrusted source/answer text.
- `aggregate`: honest denominator contract (`adherence_rate = adhered/scored`;
  uncertain abstentions and parse/judge errors excluded from the headline;
  experimental Misinformation excluded by default).
- DeepEval adapter (`adapters.deepeval.BehaviorAdherenceMetric`) with
  `aggregate_metrics` for an honest cross-case rate; supports DeepEval 4.x
  `metadata` with legacy `additional_metadata` fallback.
- CLI: `rag-conflict-eval stats` and `score`.

### Known limitations
- Oracle mode only (gold conflict type supplied); predicted-type mode is v2.
- The judge is a *reconstruction* of the paper's rubric and is **not yet
  human-validated** (see `VALIDATION.md`).
- Misinformation (n=5) is experimental and excluded from the headline.
