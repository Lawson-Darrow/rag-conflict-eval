"""rag-conflict-eval: conflict-type-conditioned behavior-adherence evaluation for RAG."""

from __future__ import annotations

from .taxonomy import TAXONOMY, ConflictType, TypeSpec, spec_for
from .types import (
    AdherenceLabel,
    AdherenceResult,
    ConflictInstance,
    SearchResult,
)

__version__ = "0.1.0"

__all__ = [
    "TAXONOMY",
    "ConflictType",
    "TypeSpec",
    "spec_for",
    "AdherenceLabel",
    "AdherenceResult",
    "ConflictInstance",
    "SearchResult",
    "__version__",
]
