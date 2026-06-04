"""Command-line interface for rag-conflict-eval.

    rag-conflict-eval stats  <conflicts.jsonl>
    rag-conflict-eval score  <answers.jsonl> [--model M] [--out results.jsonl]

`stats` loads the CONFLICTS dataset and prints per-type counts + a stratified
split preview. `score` reads a JSONL where each line is a CONFLICTS-style record
plus a ``candidate_answer`` field, judges each with the behavior-adherence scorer
(needs an LLM API key for litellm), writes per-item results, and prints the
honest aggregate report (uncertain + errors excluded; Misinformation experimental
and excluded from the headline by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .aggregate import AdherenceReport, aggregate
from .auto import AutoPipeline, AutoResult, AutoVerdict, load_traces
from .coarse import CoarseConflict, CoarseConflictDetector, benchmark_coarse_detector
from .detector import ABSTAIN, ERROR, ConflictTypeDetector, DetectorReport, benchmark_detector
from .taxonomy import ConflictType
from .loader import load_conflicts, record_to_instance, stratified_sample, stratified_split
from .scorer import BehaviorAdherenceScorer
from .types import AdherenceResult


def _result_to_dict(r: AdherenceResult) -> dict:
    return {
        "instance_id": r.instance_id,
        "conflict_type": r.conflict_type.value,
        "status": r.status.value,
        "label": r.label.value if r.label else None,
        "score": r.score,
        "rationale": r.rationale,
        "judge_model": r.judge_model,
        "prompt_version": r.prompt_version,
    }


def format_report(report: AdherenceReport) -> str:
    lines = []
    rate = report.adherence_rate
    lines.append(f"behavior-adherence: {rate:.3f}" if rate is not None else "behavior-adherence: n/a")
    lines.append(
        f"  scored={report.n_scored} adhered={report.n_adhered} "
        f"uncertain={report.n_uncertain} parse_err={report.n_parse_error} "
        f"judge_err={report.n_judge_error} (overall n={report.n_total})"
    )
    if report.experimental_excluded:
        excl = ", ".join(t.value for t in report.experimental_excluded)
        lines.append(f"  experimental types excluded from headline: {excl}")
    lines.append("  per-type:")
    for ctype, tr in sorted(report.per_type.items(), key=lambda kv: kv[0].value):
        tr_rate = tr.adherence_rate
        rstr = f"{tr_rate:.3f}" if tr_rate is not None else "n/a"
        lines.append(
            f"    {ctype.value:<22} adherence={rstr:<6} "
            f"(scored={tr.n_scored} uncertain={tr.n_uncertain} "
            f"err={tr.n_parse_error + tr.n_judge_error} n={tr.n_total})"
        )
    return "\n".join(lines)


def format_detector_report(report: DetectorReport) -> str:
    lines = []
    mf1 = report.macro_f1
    lines.append(f"detector macro-F1: {mf1:.3f}" if mf1 is not None else "detector macro-F1: n/a")
    lines.append(
        f"  coverage={report.coverage:.3f} accuracy_on_covered="
        + (f"{report.accuracy_on_covered:.3f}" if report.accuracy_on_covered is not None else "n/a")
        + f" abstained={report.abstention_rate:.3f} errors={report.error_rate:.3f} (n={report.n_total})"
    )
    if report.experimental_excluded:
        excl = ", ".join(t.value for t in report.experimental_excluded)
        lines.append(f"  excluded from macro-F1 (experimental): {excl}")
    strict = report.accuracy_strict
    lines.append(
        "  strict accuracy (abstain/error = wrong): "
        + (f"{strict:.3f}" if strict is not None else "n/a")
    )

    def _f(x):
        return f"{x:.3f}" if x is not None else "n/a"

    lines.append("  per-type (precision / recall / f1 / support):")
    for t, m in sorted(report.per_class.items(), key=lambda kv: kv[0].value):
        lines.append(
            f"    {t.value:<22} {_f(m.precision):>6} / {_f(m.recall):>6} / "
            f"{_f(m.f1):>6} / {m.support}"
        )

    # Confusion: per gold type, the distribution of predictions (incl. abstain/error).
    lines.append("  confusion (gold -> predicted counts):")
    pred_keys = [t.value for t in ConflictType] + [ABSTAIN, ERROR]
    for gold in sorted(report.per_class, key=lambda t: t.value):
        cells = [
            f"{k}:{report.confusion.get((gold, k), 0)}"
            for k in pred_keys
            if report.confusion.get((gold, k), 0)
        ]
        lines.append(f"    {gold.value:<22} -> {', '.join(cells) if cells else '(none)'}")
    return "\n".join(lines)


def format_coarse_benchmark(bench) -> str:
    def _f(x):
        return f"{x:.3f}" if x is not None else "n/a"

    lines = ["coarse conflict detection (auto mode):"]
    for t, rep in sorted(bench.by_threshold.items()):
        b = rep.binary
        lines.append(f"  --- threshold {t:.2f} ---")
        lines.append(
            f"    macro-F1={_f(rep.macro_f1)} coverage={rep.coverage:.3f} "
            f"acc_on_covered={_f(rep.accuracy_on_covered)} "
            f"abstained={rep.abstention_rate:.3f} errors={rep.error_rate:.3f}"
        )
        lines.append(
            f"    false-consensus detection (conflict vs none): "
            f"P={_f(b.get('precision'))} R={_f(b.get('recall'))} F1={_f(b.get('f1'))} "
            f"acc={_f(b.get('accuracy'))} (tp={b['tp']} fp={b['fp']} fn={b['fn']} tn={b['tn']})"
        )
        for c in CoarseConflict:
            m = rep.per_class[c]
            lines.append(
                f"      {c.value:<22} P={_f(m['precision'])} R={_f(m['recall'])} "
                f"F1={_f(m['f1'])} support={m['support']}"
            )
    return "\n".join(lines)


def _select(instances, args):
    """--sample N gives a stratified draw (preferred); --limit N takes the first N."""
    if getattr(args, "sample", 0):
        return stratified_sample(instances, args.sample, seed=getattr(args, "seed", 0))
    if args.limit:
        return instances[: args.limit]
    return instances


def _cmd_bench_coarse(args: argparse.Namespace) -> int:
    instances = _select(load_conflicts(args.data), args)
    thresholds = tuple(float(x) for x in args.thresholds.split(","))
    detector = CoarseConflictDetector(model=args.model, judged_text_field=args.judged_text_field)
    bench = benchmark_coarse_detector(detector, instances, thresholds=thresholds)
    print(format_coarse_benchmark(bench))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for gold, res in zip(bench.golds, bench.results):
                f.write(json.dumps({
                    "instance_id": res.instance_id,
                    "gold_coarse": gold.value,
                    "status": res.status.value,
                    "label": res.label.value if res.label else None,
                    "said_uncertain": res.said_uncertain,
                    "confidence": res.confidence,
                }) + "\n")
    return 0


def _cmd_bench_detector(args: argparse.Namespace) -> int:
    instances = _select(load_conflicts(args.data), args)
    detector = ConflictTypeDetector(model=args.model, judged_text_field=args.judged_text_field)
    report = benchmark_detector(detector, instances)
    print(format_detector_report(report))
    return 0


def _auto_result_to_dict(r: AutoResult) -> dict:
    return {
        "trace_id": r.trace_id,
        "verdict": r.verdict.value,
        "needs_review": r.needs_review,
        "conflict": r.conflict.value if r.conflict else None,
        "detector_label": r.detector_label,
        "detector_abstained": r.detector_abstained,
        "rationale": r.rationale,
        "risky_answer_span": r.risky_answer_span,
        "detector_status": r.detector_status.value,
        "behavior_status": r.behavior_status.value if r.behavior_status else None,
        "evidence": r.evidence,
    }


def format_auto_results(results: list[AutoResult]) -> str:
    counts = Counter(r.verdict.value for r in results)
    flagged = [r for r in results if r.needs_review]
    lines = [
        f"scanned {len(results)} traces - {len(flagged)} need review",
        "  " + " ".join(f"{v}={counts.get(v, 0)}" for v in (x.value for x in AutoVerdict)),
        "",
    ]
    for r in flagged:
        lines.append(f"[{r.verdict.value}] trace {r.trace_id}  (conflict: {r.conflict.value if r.conflict else 'n/a'})")
        if r.rationale:
            lines.append(f"    why: {r.rationale}")
        if r.risky_answer_span:
            lines.append(f"    answer span: \"{r.risky_answer_span}\"")
        claims = (r.evidence or {}).get("claims")
        if claims:
            lines.append(f"    source claims: {json.dumps(claims)[:300]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _cmd_auto(args: argparse.Namespace) -> int:
    traces = load_traces(args.traces)
    pipe = AutoPipeline(model=args.model, judged_text_field=args.judged_text_field)
    results = pipe.run_batch(traces)
    print(format_auto_results(results))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(_auto_result_to_dict(r)) + "\n")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    instances = load_conflicts(args.data)
    counts = Counter(i.conflict_type.value for i in instances)
    train, test = stratified_split(instances, test_size=args.test_size)
    print(f"loaded {len(instances)} instances from {args.data}")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"stratified split (test_size={args.test_size}): {len(train)} train / {len(test)} test")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    scorer = BehaviorAdherenceScorer(model=args.model, judged_text_field=args.judged_text_field)
    results: list[AdherenceResult] = []
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        with open(args.answers, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                answer = rec.pop("candidate_answer", None)
                if answer is None:
                    print(f"line {line_no}: missing 'candidate_answer', skipping", file=sys.stderr)
                    continue
                res = scorer.score(record_to_instance(rec), answer)
                results.append(res)
                if out:
                    out.write(json.dumps(_result_to_dict(res)) + "\n")
    finally:
        if out:
            out.close()
    if not results:
        print("no scored items", file=sys.stderr)
        return 1
    report = aggregate(results, exclude_experimental=not args.include_experimental)
    print(format_report(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-conflict-eval", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stats", help="load the CONFLICTS dataset and show counts + split")
    ps.add_argument("data", help="path to conflicts.jsonl")
    ps.add_argument("--test-size", type=float, default=0.2)
    ps.set_defaults(func=_cmd_stats)

    pc = sub.add_parser("score", help="score candidate answers for behavior adherence")
    pc.add_argument("answers", help="JSONL of CONFLICTS records + a 'candidate_answer' field")
    pc.add_argument("--model", default="gpt-4o-mini", help="litellm judge model")
    pc.add_argument(
        "--judged-text-field", default="short_text",
        choices=["snippet", "short_text", "response_str"],
    )
    pc.add_argument("--out", help="write per-item results JSONL here")
    pc.add_argument(
        "--include-experimental", action="store_true",
        help="fold experimental types (Misinformation) into the headline too",
    )
    pc.set_defaults(func=_cmd_score)

    pb = sub.add_parser(
        "bench-detector", help="benchmark the conflict-type detector vs gold labels"
    )
    pb.add_argument("data", help="path to conflicts.jsonl (gold-labeled)")
    pb.add_argument("--model", default="gpt-4o-mini", help="litellm detector model")
    pb.add_argument(
        "--judged-text-field", default="short_text",
        choices=["snippet", "short_text", "response_str"],
    )
    pb.add_argument("--limit", type=int, default=0, help="benchmark only the first N (cost control)")
    pb.add_argument("--sample", type=int, default=0, help="stratified sample of N (preferred over --limit)")
    pb.add_argument("--seed", type=int, default=0)
    pb.set_defaults(func=_cmd_bench_detector)

    pcz = sub.add_parser(
        "bench-coarse",
        help="benchmark the COARSE conflict detector (auto mode) vs gold labels",
    )
    pcz.add_argument("data", help="path to conflicts.jsonl (gold-labeled)")
    pcz.add_argument("--model", default="gpt-4o-mini", help="litellm detector model")
    pcz.add_argument(
        "--judged-text-field", default="short_text",
        choices=["snippet", "short_text", "response_str"],
    )
    pcz.add_argument("--limit", type=int, default=0, help="benchmark only the first N")
    pcz.add_argument("--sample", type=int, default=0, help="stratified sample of N (preferred over --limit)")
    pcz.add_argument("--seed", type=int, default=0)
    pcz.add_argument("--thresholds", default="0.6,0.7,0.8", help="comma-separated confidence cutoffs")
    pcz.add_argument("--out", help="write raw per-instance predictions JSONL here")
    pcz.set_defaults(func=_cmd_bench_coarse)

    pa = sub.add_parser("auto", help="find likely false consensus in your own RAG traces")
    pa.add_argument("traces", help="JSONL of traces: {question, contexts, answer, trace_id?}")
    pa.add_argument("--model", default="gpt-4o-mini", help="litellm model")
    pa.add_argument(
        "--judged-text-field", default="short_text",
        choices=["snippet", "short_text", "response_str"],
    )
    pa.add_argument("--out", help="write full per-trace results JSONL here")
    pa.set_defaults(func=_cmd_auto)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
