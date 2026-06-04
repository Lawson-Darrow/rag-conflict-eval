"""Load the CONFLICTS dataset (``conflicts.jsonl``) into typed instances.

Label strings are pinned to the released dataset (verified against the real
``conflicts.jsonl``, 458 records). Unknown labels fail loud rather than being
silently dropped or mis-mapped.

Dataset: https://github.com/google-research-datasets/rag_conflicts (Apache-2.0).
Not vendored here; point the loader at a local copy.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from .taxonomy import ConflictType
from .types import ConflictInstance, SearchResult

#: Exact ``conflict_type`` strings in the released dataset -> our enum.
LABEL_MAP: dict[str, ConflictType] = {
    "No conflict": ConflictType.NO_CONFLICT,
    "Complementary information": ConflictType.COMPLEMENTARY,
    "Conflicting opinions and research outcomes": ConflictType.CONFLICTING_OPINIONS,
    "Conflict due to outdated information": ConflictType.FRESHNESS,
    "Conflict due to misinformation": ConflictType.MISINFORMATION,
}


def _make_id(source: str | None, question: str) -> str:
    """Stable content hash id, independent of file ordering."""
    key = f"{source or ''}\x00{question}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:12]


def _parse_record(rec: dict, *, line_no: int) -> ConflictInstance:
    raw_type = rec.get("conflict_type")
    try:
        ctype = LABEL_MAP[raw_type]
    except KeyError:
        raise ValueError(
            f"Unknown conflict_type {raw_type!r} at line {line_no}. "
            f"Known labels: {sorted(LABEL_MAP)}"
        ) from None

    if "question" not in rec:
        raise ValueError(f"Record at line {line_no} has no 'question' field.")

    results = [
        SearchResult(
            title=s.get("title"),
            url=s.get("url"),
            snippet=s.get("snippet"),
            date=s.get("date"),
            response_str=s.get("response_str"),
            short_text=s.get("short_text"),
            raw=s,
        )
        for s in (rec.get("search_results") or [])
    ]
    return ConflictInstance(
        id=_make_id(rec.get("source"), rec["question"]),
        question=rec["question"],
        search_results=results,
        conflict_type=ctype,
        source=rec.get("source"),
        correct_answer=rec.get("correct_answer"),
        raw=rec,
    )


def load_conflicts(path: str | Path) -> list[ConflictInstance]:
    """Load ``conflicts.jsonl`` into a list of :class:`ConflictInstance`.

    Raises ``ValueError`` on any unknown conflict-type label or malformed record.
    """
    path = Path(path)
    instances: list[ConflictInstance] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at line {line_no}: {e}") from e
            instances.append(_parse_record(rec, line_no=line_no))
    return instances


def stratified_split(
    instances: Iterable[ConflictInstance],
    *,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[list[ConflictInstance], list[ConflictInstance]]:
    """Split per conflict type so each type keeps its proportion in both halves.

    Deterministic: instances are sorted by id before a seeded shuffle, so the
    result is independent of input ordering.
    """
    if not 0.0 <= test_size <= 1.0:
        raise ValueError(f"test_size must be in [0, 1], got {test_size}")

    by_type: dict[ConflictType, list[ConflictInstance]] = {}
    for inst in instances:
        by_type.setdefault(inst.conflict_type, []).append(inst)

    rng = random.Random(seed)
    train: list[ConflictInstance] = []
    test: list[ConflictInstance] = []
    for items in by_type.values():
        items = sorted(items, key=lambda i: i.id)
        rng.shuffle(items)
        n_test = round(len(items) * test_size)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test
