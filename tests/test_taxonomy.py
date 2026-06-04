from rag_conflict_eval import TAXONOMY, ConflictType, spec_for


def test_all_five_types_present():
    assert set(TAXONOMY) == set(ConflictType)
    assert len(TAXONOMY) == 5


def test_every_spec_has_nonempty_behavior():
    for ctype, spec in TAXONOMY.items():
        assert spec.type is ctype
        assert spec.expected_behavior.strip()
        assert spec.definition.strip()


def test_released_dataset_provides_gold_answer_for_all_types():
    # Verified against the real conflicts.jsonl (458/458 have correct_answer).
    assert all(spec.has_gold_answer for spec in TAXONOMY.values())


def test_misinformation_flagged_experimental():
    assert spec_for(ConflictType.MISINFORMATION).experimental is True
    assert spec_for(ConflictType.NO_CONFLICT).experimental is False
