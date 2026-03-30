"""Delta-based belief system for learning without retraining."""

from .delta_system import BeliefDelta, DeltaRegistry, DeltaRouter, build_delta_map, forward_with_deltas
from .belief_encoder import BeliefEncoder
from .knowledge_store import KnowledgeStore, KnowledgeEntry
from .hierarchical_embedding import HierarchicalEmbedding
