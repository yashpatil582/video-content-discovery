"""Search relevance evaluation benchmarks."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import statistics

import numpy as np

from .search import VideoSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """Represents a single evaluation query with ground truth."""
    query: str
    relevant_video_ids: list[str]
    description: str = ""


@dataclass
class EvaluationResult:
    """Stores evaluation metrics for a query set."""
    mrr: float = 0.0
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_mean: float = 0.0
    total_queries: int = 0
    query_results: list[dict] = field(default_factory=list)


class SearchEvaluator:
    """Evaluates search relevance and performance."""

    def __init__(
        self,
        search_engine: Optional[VideoSearchEngine] = None,
        index_id: Optional[str] = None,
    ):
        self.search_engine = search_engine or VideoSearchEngine()
        self.index_id = index_id
        self.latencies: list[float] = []

    def mean_reciprocal_rank(
        self,
        relevant_ids: list[str],
        result_ids: list[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR measures the rank position of the first relevant result.

        Args:
            relevant_ids: Ground truth relevant video IDs
            result_ids: Returned video IDs in ranked order

        Returns:
            Reciprocal rank (1/rank of first relevant result)
        """
        for rank, result_id in enumerate(result_ids, 1):
            if result_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def precision_at_k(
        self,
        relevant_ids: list[str],
        result_ids: list[str],
        k: int,
    ) -> float:
        """
        Calculate Precision@K.

        Precision@K measures what fraction of the top-k results are relevant.

        Args:
            relevant_ids: Ground truth relevant video IDs
            result_ids: Returned video IDs in ranked order
            k: Number of top results to consider

        Returns:
            Precision score (relevant in top-k / k)
        """
        top_k_results = result_ids[:k]
        relevant_in_top_k = sum(1 for rid in top_k_results if rid in relevant_ids)
        return relevant_in_top_k / k if k > 0 else 0.0

    def recall_at_k(
        self,
        relevant_ids: list[str],
        result_ids: list[str],
        k: int,
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K measures what fraction of relevant items appear in top-k.

        Args:
            relevant_ids: Ground truth relevant video IDs
            result_ids: Returned video IDs in ranked order
            k: Number of top results to consider

        Returns:
            Recall score (relevant in top-k / total relevant)
        """
        if not relevant_ids:
            return 0.0
        top_k_results = result_ids[:k]
        relevant_in_top_k = sum(1 for rid in top_k_results if rid in relevant_ids)
        return relevant_in_top_k / len(relevant_ids)

    def evaluate_query(
        self,
        query: EvaluationQuery,
        search_method: str = "native",
    ) -> dict:
        """
        Evaluate a single query.

        Args:
            query: EvaluationQuery with query text and ground truth
            search_method: 'native', 'embedding', or 'hybrid'

        Returns:
            Dict with metrics for this query
        """
        start_time = time.time()

        # Execute search
        if search_method == "native" and self.index_id:
            results = self.search_engine.search_native(
                self.index_id,
                query.query,
                top_k=10,
            )
            result_ids = [r.get("video_id", "") for r in results.get("data", [])]
        elif search_method == "embedding":
            results = self.search_engine.search_with_embeddings(query.query, top_k=10)
            result_ids = [r.get("metadata", {}).get("video_id", "") for r in results]
        else:
            results = self.search_engine.hybrid_search(
                self.index_id,
                query.query,
                top_k=10,
            )
            result_ids = [r.get("video_id", "") for r in results]

        latency = (time.time() - start_time) * 1000  # ms
        self.latencies.append(latency)

        # Calculate metrics
        mrr = self.mean_reciprocal_rank(query.relevant_video_ids, result_ids)
        p_at_1 = self.precision_at_k(query.relevant_video_ids, result_ids, 1)
        p_at_5 = self.precision_at_k(query.relevant_video_ids, result_ids, 5)
        p_at_10 = self.precision_at_k(query.relevant_video_ids, result_ids, 10)
        recall = self.recall_at_k(query.relevant_video_ids, result_ids, 10)

        return {
            "query": query.query,
            "mrr": mrr,
            "precision_at_1": p_at_1,
            "precision_at_5": p_at_5,
            "precision_at_10": p_at_10,
            "recall_at_10": recall,
            "latency_ms": latency,
            "result_count": len(result_ids),
        }

    def evaluate_query_set(
        self,
        queries: list[EvaluationQuery],
        search_method: str = "native",
    ) -> EvaluationResult:
        """
        Evaluate a set of queries and aggregate metrics.

        Args:
            queries: List of EvaluationQuery objects
            search_method: 'native', 'embedding', or 'hybrid'

        Returns:
            EvaluationResult with aggregated metrics
        """
        self.latencies = []
        query_results = []

        for query in queries:
            result = self.evaluate_query(query, search_method)
            query_results.append(result)

        # Aggregate metrics
        avg_mrr = statistics.mean([r["mrr"] for r in query_results])
        avg_p1 = statistics.mean([r["precision_at_1"] for r in query_results])
        avg_p5 = statistics.mean([r["precision_at_5"] for r in query_results])
        avg_p10 = statistics.mean([r["precision_at_10"] for r in query_results])
        avg_recall = statistics.mean([r["recall_at_10"] for r in query_results])

        # Latency percentiles
        sorted_latencies = sorted(self.latencies)
        p50_idx = int(len(sorted_latencies) * 0.5)
        p95_idx = int(len(sorted_latencies) * 0.95)

        return EvaluationResult(
            mrr=avg_mrr,
            precision_at_1=avg_p1,
            precision_at_5=avg_p5,
            precision_at_10=avg_p10,
            recall_at_10=avg_recall,
            latency_p50=sorted_latencies[p50_idx] if sorted_latencies else 0,
            latency_p95=sorted_latencies[p95_idx] if sorted_latencies else 0,
            latency_mean=statistics.mean(self.latencies) if self.latencies else 0,
            total_queries=len(queries),
            query_results=query_results,
        )

    def generate_report(
        self,
        result: EvaluationResult,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            result: EvaluationResult to format
            output_path: Optional path to save report

        Returns:
            Formatted report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           VIDEO SEARCH EVALUATION REPORT                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Queries Evaluated: {result.total_queries:<36} ║
╠══════════════════════════════════════════════════════════════════╣
║  RELEVANCE METRICS                                               ║
╠──────────────────────────────────────────────────────────────────╣
║  Mean Reciprocal Rank (MRR):     {result.mrr:>6.4f}                       ║
║  Precision@1:                     {result.precision_at_1:>6.4f}                       ║
║  Precision@5:                     {result.precision_at_5:>6.4f}                       ║
║  Precision@10:                    {result.precision_at_10:>6.4f}                       ║
║  Recall@10:                       {result.recall_at_10:>6.4f}                       ║
╠══════════════════════════════════════════════════════════════════╣
║  LATENCY METRICS                                                 ║
╠──────────────────────────────────────────────────────────────────╣
║  P50 Latency:                    {result.latency_p50:>7.2f} ms                     ║
║  P95 Latency:                    {result.latency_p95:>7.2f} ms                     ║
║  Mean Latency:                   {result.latency_mean:>7.2f} ms                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)

            # Also save JSON for programmatic access
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump({
                    "mrr": result.mrr,
                    "precision_at_1": result.precision_at_1,
                    "precision_at_5": result.precision_at_5,
                    "precision_at_10": result.precision_at_10,
                    "recall_at_10": result.recall_at_10,
                    "latency_p50": result.latency_p50,
                    "latency_p95": result.latency_p95,
                    "latency_mean": result.latency_mean,
                    "total_queries": result.total_queries,
                    "query_results": result.query_results,
                }, f, indent=2)

        return report

    def load_queries_from_file(self, path: Path) -> list[EvaluationQuery]:
        """Load evaluation queries from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        return [
            EvaluationQuery(
                query=q["query"],
                relevant_video_ids=q["relevant_video_ids"],
                description=q.get("description", ""),
            )
            for q in data.get("queries", [])
        ]

    def create_sample_queries(self) -> list[EvaluationQuery]:
        """Create sample evaluation queries for testing."""
        return [
            EvaluationQuery(
                query="person walking in the park",
                relevant_video_ids=["sample_video_1"],
                description="Test query for walking scenes",
            ),
            EvaluationQuery(
                query="presentation with slides",
                relevant_video_ids=["sample_video_2"],
                description="Test query for presentation content",
            ),
            EvaluationQuery(
                query="cooking in kitchen",
                relevant_video_ids=["sample_video_3"],
                description="Test query for cooking scenes",
            ),
        ]
