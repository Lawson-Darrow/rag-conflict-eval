"""rag-conflict-eval: conflict-type-conditioned behavior-adherence evaluation for RAG."""

from __future__ import annotations

from .aggregate import AdherenceReport, TypeReport, aggregate
from .auto import AutoPipeline, AutoResult, AutoVerdict, Trace, load_traces
from .coarse import (
    CoarseConflict,
    CoarseConflictDetector,
    CoarseDetectionResult,
    FINE_TO_COARSE,
    benchmark_coarse_detector,
    resolve,
)
from .detector import (
    ConflictTypeDetector,
    DetectionResult,
    DetectorReport,
    benchmark_detector,
)
from .loader import LABEL_MAP, load_conflicts, stratified_split
from .prompts.templates import PROMPT_VERSION, build_judge_prompt
from .scorer import BehaviorAdherenceScorer, parse_verdict
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
    # scoring
    "BehaviorAdherenceScorer",
    "parse_verdict",
    "build_judge_prompt",
    "PROMPT_VERSION",
    # detection (fine, oracle)
    "ConflictTypeDetector",
    "DetectionResult",
    "DetectorReport",
    "benchmark_detector",
    # coarse detection (auto / false-consensus wedge)
    "CoarseConflict",
    "CoarseConflictDetector",
    "CoarseDetectionResult",
    "FINE_TO_COARSE",
    "benchmark_coarse_detector",
    "resolve",
    # auto mode (run on your own RAG traces)
    "AutoPipeline",
    "AutoResult",
    "AutoVerdict",
    "Trace",
    "load_traces",
    "__version__",
]
