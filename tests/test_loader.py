import json

import pytest

from rag_conflict_eval import ConflictInstance, ConflictType, load_conflicts, stratified_split
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


def test_make_id_stable_and_content_sensitive():
    a = {"source": "freshqa", "question": "Q?", "x": 1}
    assert _make_id(a) == _make_id(dict(a))          # identical content -> same id
    assert _make_id(a) != _make_id({**a, "x": 2})    # any field change -> new id
    assert _make_id(a) != _make_id({**a, "source": "other"})


def test_unknown_label_fails_loud(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "q", "conflict_type": "Totally Made Up"}) + "\n")
    with pytest.raises(ValueError, match="Unknown conflict_type"):
        load_conflicts(bad)


def test_duplicate_record_fails_loud(tmp_path):
    rec = {"question": "dup?", "conflict_type": "No conflict", "search_results": []}
    dup = tmp_path / "dup.jsonl"
    dup.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="Duplicate instance id"):
        load_conflicts(dup)


# --- stratified_split: build synthetic multi-item classes to expose RNG/order bugs ---

def _inst(i: int, ctype: ConflictType) -> ConflictInstance:
    return ConflictInstance(id=f"{ctype.value}-{i:03d}", question=f"q{i}",
                            search_results=[], conflict_type=ctype)


def _balanced(n_per_type: int = 10) -> list[ConflictInstance]:
    return [_inst(i, t) for t in ConflictType for i in range(n_per_type)]


def test_split_preserves_proportions():
    insts = _balanced(10)
    train, test = stratified_split(insts, test_size=0.3, seed=0)
    assert len(test) == 3 * len(ConflictType)   # 3 per type
    assert len(train) + len(test) == len(insts)
    # every type represented in both halves
    for t in ConflictType:
        assert any(i.conflict_type is t for i in test)
        assert any(i.conflict_type is t for i in train)


def test_split_is_order_independent():
    insts = _balanced(10)
    _, test_a = stratified_split(insts, test_size=0.3, seed=0)
    _, test_b = stratified_split(list(reversed(insts)), test_size=0.3, seed=0)
    assert {i.id for i in test_a} == {i.id for i in test_b}


def test_split_seed_changes_assignment():
    insts = _balanced(10)
    _, test_0 = stratified_split(insts, test_size=0.3, seed=0)
    _, test_1 = stratified_split(insts, test_size=0.3, seed=1)
    assert {i.id for i in test_0} != {i.id for i in test_1}


def test_split_edge_sizes_and_tiny_class():
    insts = _balanced(10) + [_inst(0, ConflictType.MISINFORMATION)]  # one extra tiny
    train, test = stratified_split(insts, test_size=0.0, seed=0)
    assert len(test) == 0 and len(train) == len(insts)
    train, test = stratified_split(insts, test_size=1.0, seed=0)
    assert len(train) == 0 and len(test) == len(insts)
    with pytest.raises(ValueError):
        stratified_split(insts, test_size=1.5)
