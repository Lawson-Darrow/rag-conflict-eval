import json

import pytest

from rag_conflict_eval import ConflictType, load_conflicts, stratified_split
from rag_conflict_eval.loader import _make_id


def test_load_sample(sample_path):
    insts = load_conflicts(sample_path)
    assert len(insts) == 6
    types = [i.conflict_type for i in insts]
    assert ConflictType.FRESHNESS in types
    assert ConflictType.MISINFORMATION in types
    assert types.count(ConflictType.NO_CONFLICT) == 2


def test_fields_mapped(sample_path):
    inst = load_conflicts(sample_path)[0]
    assert inst.question
    assert inst.conflict_type is ConflictType.FRESHNESS
    assert inst.correct_answer == "John Doe"
    assert inst.search_results[0].short_text
    assert inst.search_results[0].raw  # original record preserved
    assert inst.id  # stable id assigned


def test_id_is_stable_and_order_independent():
    a = _make_id("freshqa", "Q?")
    b = _make_id("freshqa", "Q?")
    c = _make_id("other", "Q?")
    assert a == b
    assert a != c


def test_unknown_label_fails_loud(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "q", "conflict_type": "Totally Made Up"}) + "\n")
    with pytest.raises(ValueError, match="Unknown conflict_type"):
        load_conflicts(bad)


def test_stratified_split_preserves_types_and_is_deterministic(sample_path):
    insts = load_conflicts(sample_path)
    train1, test1 = stratified_split(insts, test_size=0.5, seed=0)
    train2, test2 = stratified_split(list(reversed(insts)), test_size=0.5, seed=0)
    assert len(train1) + len(test1) == len(insts)
    # order-independent determinism
    assert {i.id for i in test1} == {i.id for i in test2}
