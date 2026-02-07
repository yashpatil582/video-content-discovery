"""Unit tests for search functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingManager
from src.evaluation import SearchEvaluator, EvaluationQuery, EvaluationResult


class TestEmbeddingManager:
    """Tests for EmbeddingManager class."""

    @pytest.fixture
    def embedding_manager(self, tmp_path):
        """Create an embedding manager with temp path."""
        return EmbeddingManager(
            api_key="test_key",
            index_path=tmp_path / "faiss",
        )

    def test_create_faiss_index(self, embedding_manager):
        """Test FAISS index creation."""
        index = embedding_manager.create_faiss_index()
        assert index is not None
        assert embedding_manager.faiss_index is not None

    def test_add_and_search_embeddings(self, embedding_manager):
        """Test adding and searching embeddings."""
        embedding_manager.create_faiss_index()

        # Add embeddings
        embeddings = np.random.rand(5, 1024).astype(np.float32)
        metadata = [{"video_id": f"vid{i}"} for i in range(5)]
        embedding_manager.add_embeddings(embeddings, metadata)

        # Search
        query = np.random.rand(1024).astype(np.float32)
        results = embedding_manager.search(query, top_k=3)

        assert len(results) == 3
        assert all("score" in r for r in results)
        assert all("metadata" in r for r in results)

    def test_save_and_load_index(self, embedding_manager):
        """Test saving and loading FAISS index."""
        embedding_manager.create_faiss_index()

        # Add embeddings
        embeddings = np.random.rand(3, 1024).astype(np.float32)
        metadata = [{"video_id": f"vid{i}"} for i in range(3)]
        embedding_manager.add_embeddings(embeddings, metadata)

        # Save
        embedding_manager.save_index("test")

        # Create new manager and load
        new_manager = EmbeddingManager(
            api_key="test_key",
            index_path=embedding_manager.index_path,
        )
        loaded = new_manager.load_index("test")

        assert loaded is True
        assert new_manager.faiss_index.ntotal == 3
        assert len(new_manager.metadata) == 3

    def test_get_index_stats(self, embedding_manager):
        """Test getting index statistics."""
        # Before initialization
        stats = embedding_manager.get_index_stats()
        assert stats["status"] == "not_initialized"

        # After initialization
        embedding_manager.create_faiss_index()
        embeddings = np.random.rand(5, 1024).astype(np.float32)
        embedding_manager.add_embeddings(embeddings, [{}] * 5)

        stats = embedding_manager.get_index_stats()
        assert stats["status"] == "active"
        assert stats["total_vectors"] == 5


class TestSearchEvaluator:
    """Tests for SearchEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator with mock search engine."""
        mock_engine = Mock()
        return SearchEvaluator(mock_engine, "test_index")

    def test_mean_reciprocal_rank(self, evaluator):
        """Test MRR calculation."""
        relevant = ["vid1", "vid2"]

        # First result is relevant
        results1 = ["vid1", "vid3", "vid4"]
        assert evaluator.mean_reciprocal_rank(relevant, results1) == 1.0

        # Second result is relevant
        results2 = ["vid3", "vid1", "vid4"]
        assert evaluator.mean_reciprocal_rank(relevant, results2) == 0.5

        # Third result is relevant
        results3 = ["vid3", "vid4", "vid2"]
        assert evaluator.mean_reciprocal_rank(relevant, results3) == pytest.approx(1/3)

        # No relevant results
        results4 = ["vid3", "vid4", "vid5"]
        assert evaluator.mean_reciprocal_rank(relevant, results4) == 0.0

    def test_precision_at_k(self, evaluator):
        """Test Precision@K calculation."""
        relevant = ["vid1", "vid2", "vid3"]
        results = ["vid1", "vid4", "vid2", "vid5", "vid6"]

        # P@1: 1 relevant out of 1
        assert evaluator.precision_at_k(relevant, results, 1) == 1.0

        # P@2: 1 relevant out of 2
        assert evaluator.precision_at_k(relevant, results, 2) == 0.5

        # P@3: 2 relevant out of 3
        assert evaluator.precision_at_k(relevant, results, 3) == pytest.approx(2/3)

        # P@5: 2 relevant out of 5
        assert evaluator.precision_at_k(relevant, results, 5) == 0.4

    def test_recall_at_k(self, evaluator):
        """Test Recall@K calculation."""
        relevant = ["vid1", "vid2", "vid3", "vid4"]
        results = ["vid1", "vid5", "vid2", "vid6", "vid3"]

        # R@1: 1 found out of 4 relevant
        assert evaluator.recall_at_k(relevant, results, 1) == 0.25

        # R@3: 2 found out of 4 relevant
        assert evaluator.recall_at_k(relevant, results, 3) == 0.5

        # R@5: 3 found out of 4 relevant
        assert evaluator.recall_at_k(relevant, results, 5) == 0.75

    def test_create_sample_queries(self, evaluator):
        """Test sample query creation."""
        queries = evaluator.create_sample_queries()

        assert len(queries) == 3
        assert all(isinstance(q, EvaluationQuery) for q in queries)
        assert all(q.query for q in queries)
        assert all(q.relevant_video_ids for q in queries)

    def test_generate_report(self, evaluator):
        """Test report generation."""
        result = EvaluationResult(
            mrr=0.85,
            precision_at_1=0.80,
            precision_at_5=0.75,
            precision_at_10=0.70,
            recall_at_10=0.65,
            latency_p50=50.0,
            latency_p95=150.0,
            latency_mean=75.0,
            total_queries=10,
        )

        report = evaluator.generate_report(result)

        assert "0.85" in report or "0.8500" in report
        assert "50.00" in report
        assert "150.00" in report
        assert "10" in report


class TestEvaluationQuery:
    """Tests for EvaluationQuery dataclass."""

    def test_evaluation_query_creation(self):
        """Test creating evaluation queries."""
        query = EvaluationQuery(
            query="test search",
            relevant_video_ids=["vid1", "vid2"],
            description="Test query",
        )

        assert query.query == "test search"
        assert len(query.relevant_video_ids) == 2
        assert query.description == "Test query"


class TestSearchMetrics:
    """Tests for search evaluation metrics."""

    def test_mrr_perfect_score(self):
        """Test MRR with perfect first result."""
        evaluator = SearchEvaluator(Mock(), "test")
        mrr = evaluator.mean_reciprocal_rank(["vid1"], ["vid1", "vid2", "vid3"])
        assert mrr == 1.0

    def test_precision_empty_results(self):
        """Test precision with empty results."""
        evaluator = SearchEvaluator(Mock(), "test")
        precision = evaluator.precision_at_k(["vid1"], [], 5)
        assert precision == 0.0

    def test_recall_all_found(self):
        """Test recall when all relevant items found."""
        evaluator = SearchEvaluator(Mock(), "test")
        recall = evaluator.recall_at_k(["vid1", "vid2"], ["vid1", "vid2", "vid3"], 3)
        assert recall == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
