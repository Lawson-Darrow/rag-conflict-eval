"""rag-conflict-eval: conflict-type-conditioned behavior-adherence evaluation for RAG."""

from __future__ import annotations

from .aggregate import AdherenceReport, TypeReport, aggregate
from .loader import LABEL_MAP, load_conflicts, stratified_split
from .taxonomy import TAXONOMY, ConflictType, TypeSpec, spec_for
from .types import (
    AdherenceLabel,
    AdherenceResult,
    ConflictInstance,
    JudgedTextField,
    ResultStatus,
    SearchResult,
)

__version__ = "0.1.0"

__all__ = [
    # taxonomy
    "TAXONOMY",
    "ConflictType",
    "TypeSpec",
    "spec_for",
    # types
    "AdherenceLabel",
    "AdherenceResult",
    "ConflictInstance",
    "JudgedTextField",
    "ResultStatus",
    "SearchResult",
    # loader
    "LABEL_MAP",
    "load_conflicts",
    "stratified_split",
    # aggregate
    "AdherenceReport",
    "TypeReport",
    "aggregate",
    "__version__",
]
