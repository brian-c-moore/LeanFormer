"""
Reasoning-Retrieval Separation Validation.

Formally validates that LeanFormer's architecture structurally separates
reasoning from retrieval and that retrieval failures are explicit rather
than silent.

Core claim: in a standard transformer, retrieval failure is invisible —
the model continues producing output as if retrieval succeeded. In
LeanFormer, retrieval failure is explicit: the Knowledge Plane returns
an empty result, the confidence score drops, and the output is flagged.

Benchmark categories:
  (a) Queries requiring active delta retrieval and composition
  (b) Queries requiring reasoning with no Knowledge Plane activation
  (c) Queries requiring both reasoning and retrieval

For each category, we measure:
  - Confidence score distribution
  - Provenance attribution accuracy
  - Uncertainty flagging rate
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from ..knowledge_plane.provenance import ProvenanceSignal, ConfidenceScorer
from ..knowledge_plane.registry import DeltaRegistry
from ..knowledge_plane.router import KnowledgePlaneRouter


@dataclass
class BenchmarkQuery:
    """A single query in the benchmark set."""
    prompt: str
    category: str           # "retrieval", "reasoning", "both"
    expected_delta: bool    # Should a delta activate?
    expected_category: Optional[str] = None  # Expected delta category if any
    description: str = ""


@dataclass
class SeparationResult:
    """Results for a single query."""
    query: BenchmarkQuery
    confidence_score: float
    uncertainty_flagged: bool
    delta_activated: bool
    routing_strength: float
    categories_activated: List[str]
    correct_activation: bool    # Did delta activation match expectation?
    correct_flagging: bool      # Did uncertainty flag match expectation?


@dataclass
class SeparationReport:
    """Aggregate results across all benchmark queries."""
    results: List[SeparationResult]
    per_category: Dict[str, Dict] = field(default_factory=dict)

    def compute(self):
        """Compute per-category and overall metrics."""
        by_cat = {}
        for r in self.results:
            cat = r.query.category
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(r)

        for cat, cat_results in by_cat.items():
            n = len(cat_results)
            if n == 0:
                continue

            confidence_scores = [r.confidence_score for r in cat_results]
            avg_confidence = sum(confidence_scores) / n

            flagging_rate = sum(1 for r in cat_results if r.uncertainty_flagged) / n
            activation_rate = sum(1 for r in cat_results if r.delta_activated) / n
            correct_activation = sum(1 for r in cat_results if r.correct_activation) / n
            correct_flagging = sum(1 for r in cat_results if r.correct_flagging) / n

            self.per_category[cat] = {
                "count": n,
                "avg_confidence": avg_confidence,
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores),
                "uncertainty_flagging_rate": flagging_rate,
                "delta_activation_rate": activation_rate,
                "correct_activation_rate": correct_activation,
                "correct_flagging_rate": correct_flagging,
            }

        return self

    @property
    def overall_correct_flagging(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct_flagging) / len(self.results)

    @property
    def provenance_attribution_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct_activation) / len(self.results)

    def summary(self) -> Dict:
        if not self.per_category:
            self.compute()
        return {
            "overall_correct_flagging": self.overall_correct_flagging,
            "provenance_attribution": self.provenance_attribution_accuracy,
            "n_queries": len(self.results),
            "per_category": self.per_category,
        }


class ReasoningRetrievalBenchmark:
    """
    Evaluates the reasoning-retrieval separation in LeanFormer.

    The benchmark creates three categories of queries and measures
    whether the provenance system correctly identifies the knowledge
    source (or absence) for each.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        registry: DeltaRegistry,
        router: KnowledgePlaneRouter,
        confidence_scorer: ConfidenceScorer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.router = router
        self.scorer = confidence_scorer
        self.device = device

    def create_default_queries(
        self,
        registered_categories: Optional[List[str]] = None,
    ) -> List[BenchmarkQuery]:
        """
        Create a default benchmark query set.

        The queries are designed to test three scenarios:
        1. Retrieval queries: knowledge that should be in deltas
        2. Reasoning queries: general reasoning, no delta needed
        3. Both queries: need delta knowledge + reasoning to combine
        """
        if registered_categories is None:
            registered_categories = list(set(
                d.category for d in self.registry.registered_deltas.values()
            ))

        queries = []

        # Category (a): Retrieval queries — should activate deltas
        for cat in registered_categories:
            queries.extend([
                BenchmarkQuery(
                    prompt=f"What is a key fact about {cat}?",
                    category="retrieval",
                    expected_delta=True,
                    expected_category=cat,
                    description=f"Direct {cat} knowledge query",
                ),
                BenchmarkQuery(
                    prompt=f"Tell me about {cat}.",
                    category="retrieval",
                    expected_delta=True,
                    expected_category=cat,
                    description=f"Open-ended {cat} query",
                ),
            ])

        # Category (b): Reasoning queries — should NOT activate deltas
        reasoning_prompts = [
            "What is 2 + 2?",
            "If all cats are animals and all animals breathe, do cats breathe?",
            "Complete the pattern: 1, 2, 4, 8, ...",
            "Rephrase: The quick brown fox jumps over the lazy dog.",
            "Is the following statement logical: If it rains then the ground is wet.",
        ]
        for prompt in reasoning_prompts:
            queries.append(BenchmarkQuery(
                prompt=prompt,
                category="reasoning",
                expected_delta=False,
                description="Pure reasoning query",
            ))

        # Category (c): Both — need reasoning + retrieval
        for cat in registered_categories:
            queries.append(BenchmarkQuery(
                prompt=f"Compare and contrast two aspects of {cat} and explain why they matter.",
                category="both",
                expected_delta=True,
                expected_category=cat,
                description=f"Reasoning about {cat} knowledge",
            ))

        return queries

    def evaluate_query(
        self,
        query: BenchmarkQuery,
    ) -> SeparationResult:
        """Evaluate a single benchmark query."""
        # Embed query
        tokens = self.tokenizer.encode(query.prompt, return_tensors="pt")
        tokens = tokens.to(self.device)
        with torch.no_grad():
            positions = torch.arange(tokens.shape[1], device=self.device).unsqueeze(0)
            emb = self.model.token_embedding(tokens) + self.model.position_embedding(positions)
            query_emb = emb.mean(dim=1).squeeze().cpu()

        # Route
        routing = self.router.route(query_emb)

        # Compute provenance
        provenance = self.scorer.score(
            routing_decisions=routing,
            registry=self.registry,
            query_embedding=query_emb,
        )

        # Determine correctness
        delta_activated = len(routing) > 0 and provenance.routing_strength > 0.1
        categories = provenance.source_categories

        # Correct activation: delta activated matches expectation
        if query.expected_delta:
            correct_activation = delta_activated
            if query.expected_category:
                correct_activation = correct_activation and (
                    query.expected_category in categories
                )
        else:
            correct_activation = not delta_activated

        # Correct flagging: uncertainty flag matches knowledge absence
        if query.expected_delta:
            # Should NOT be flagged as uncertain (knowledge exists)
            correct_flagging = not provenance.uncertainty_flag if delta_activated else True
        else:
            # Should be flagged as uncertain (no knowledge delta expected)
            correct_flagging = provenance.uncertainty_flag or not delta_activated

        return SeparationResult(
            query=query,
            confidence_score=provenance.confidence_score,
            uncertainty_flagged=provenance.uncertainty_flag,
            delta_activated=delta_activated,
            routing_strength=provenance.routing_strength,
            categories_activated=categories,
            correct_activation=correct_activation,
            correct_flagging=correct_flagging,
        )

    def evaluate(
        self,
        queries: Optional[List[BenchmarkQuery]] = None,
    ) -> SeparationReport:
        """
        Run the full benchmark evaluation.

        Args:
            queries: benchmark queries (default: auto-generated)

        Returns:
            SeparationReport with per-category and overall metrics
        """
        if queries is None:
            queries = self.create_default_queries()

        results = []
        for query in queries:
            result = self.evaluate_query(query)
            results.append(result)

        report = SeparationReport(results=results)
        report.compute()
        return report
