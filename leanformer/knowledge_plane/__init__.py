"""Additive Knowledge Architecture: forge, registry, router, runtime, consolidation."""

from .dfs import DeltaFormatSpec
from .registry import DeltaRegistry
from .router import KnowledgePlaneRouter
from .runtime import KnowledgeRuntime
from .forge import KnowledgeForge
from .consolidation import KnowledgePlaneConsolidator
from .server import create_app

# Extensions
from .provenance import ConfidenceScorer, ProvenanceSignal, ProvenanceLog
from .quantization import DeltaQuantizer, DeltaQuantizationSpec, QuantizedDelta
from .few_shot import FewShotForge, FewShotSweepResult
