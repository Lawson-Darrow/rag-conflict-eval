"""End-to-end: load -> score (fake judge) -> aggregate, on real instances."""

from rag_conflict_eval import (
    BehaviorAdherenceScorer,
    ConflictType,
    aggregate,
    load_conflicts,
)


def test_load_score_aggregate_pipeline(sample_path):
    instances = load_conflicts(sample_path)

    # Fake judge: adhere unless the candidate contains "BAD".
    def judge(messages):
        user = messages[-1]["content"]
        return '{"verdict": "not_adhere"}' if "BAD" in user else '{"verdict": "adhere"}'

    scorer = BehaviorAdherenceScorer(judge_fn=judge)
    pairs = [(inst, "BAD answer" if i == 0 else "good answer")
             for i, inst in enumerate(instances)]
    results = scorer.score_batch(pairs)
    assert len(results) == len(instances)

    report = aggregate(results)  # misinformation excluded from headline by default
    # 6 instances, 1 is misinformation (experimental -> out of overall)
    assert report.n_total == 5
    assert ConflictType.MISINFORMATION in report.experimental_excluded
    # instance 0 (freshness) was BAD -> not adhered; the rest adhered
    assert report.n_scored == 5
    assert report.n_adhered == 4
    assert report.adherence_rate == 4 / 5
    # per-type still includes misinformation
    assert ConflictType.MISINFORMATION in report.per_type
    assert report.per_type[ConflictType.MISINFORMATION].n_total == 1
