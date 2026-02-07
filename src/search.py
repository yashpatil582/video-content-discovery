"""Semantic search logic for video content."""

import logging
import time
from typing import Optional

from twelvelabs import TwelveLabs

from config.settings import (
    TWELVELABS_API_KEY,
    DEFAULT_TOP_K,
    SEARCH_THRESHOLD,
)
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VideoSearchEngine:
    """Handles semantic search operations for video content."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.api_key = api_key or TWELVELABS_API_KEY
        self.client = TwelveLabs(api_key=self.api_key)
        self.embedding_manager = embedding_manager or EmbeddingManager(api_key)

    def search_native(
        self,
        index_id: str,
        query: str,
        search_options: list[str] = None,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = SEARCH_THRESHOLD,
        group_by: str = "clip",
    ) -> dict:
        """
        Perform native search using TwelveLabs Search API.

        Args:
            index_id: The TwelveLabs index ID
            query: Natural language search query
            search_options: List of search types (visual, audio)
            top_k: Number of results to return
            threshold: Minimum confidence threshold (0-1, mapped to high/medium/low/none)
            group_by: How to group results (clip or video)

        Returns:
            Search results with video clips and confidence scores
        """
        if search_options is None:
            search_options = ["visual", "audio"]

        # Map numeric threshold to string
        if threshold >= 0.7:
            threshold_str = "high"
        elif threshold >= 0.4:
            threshold_str = "medium"
        elif threshold > 0:
            threshold_str = "low"
        else:
            threshold_str = "none"

        start_time = time.time()

        try:
            search_results = self.client.search.query(
                index_id=index_id,
                query_text=query,
                search_options=search_options,
                group_by=group_by,
                threshold=threshold_str,
                page_limit=top_k,
            )

            latency = (time.time() - start_time) * 1000

            # Convert results to dict format
            data = []
            for item in search_results:
                data.append({
                    "video_id": item.video_id if hasattr(item, 'video_id') else "",
                    "confidence": item.score if hasattr(item, 'score') else 0,
                    "start": item.start if hasattr(item, 'start') else 0,
                    "end": item.end if hasattr(item, 'end') else 0,
                    "thumbnail_url": item.thumbnail_url if hasattr(item, 'thumbnail_url') else None,
                })

            results = {
                "data": data,
                "latency_ms": latency,
            }

            logger.info(f"Search completed in {latency:.2f}ms with {len(data)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_with_embeddings(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Perform search using local FAISS embeddings.

        Args:
            query: Natural language search query
            top_k: Number of results to return

        Returns:
            List of search results with scores and metadata
        """
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embedding_manager.get_text_embedding(query)

        # Search FAISS index
        results = self.embedding_manager.search(query_embedding, top_k)

        latency = (time.time() - start_time) * 1000

        for result in results:
            result["latency_ms"] = latency

        logger.info(f"Embedding search completed in {latency:.2f}ms")
        return results

    def generate_summary(
        self,
        video_id: str,
        index_id: str,
        summary_type: str = "summary",
        prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate video summary using TwelveLabs Summarize API.

        Args:
            video_id: The TwelveLabs video ID
            index_id: The TwelveLabs index ID
            summary_type: Type of generation (summary, chapter, highlight)
            prompt: Optional custom prompt

        Returns:
            Generated summary content
        """
        try:
            if summary_type == "summary":
                result = self.client.summarize.create(
                    video_id=video_id,
                    type="summary",
                    prompt=prompt,
                )
            elif summary_type == "chapter":
                result = self.client.summarize.create(
                    video_id=video_id,
                    type="chapter",
                )
            elif summary_type == "highlight":
                result = self.client.summarize.create(
                    video_id=video_id,
                    type="highlight",
                )
            else:
                result = self.client.summarize.create(
                    video_id=video_id,
                    type="summary",
                )

            return {"summary": result.summary if hasattr(result, 'summary') else str(result)}
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise

    def generate_tags(
        self,
        video_id: str,
        index_id: str,
        types: list[str] = None,
    ) -> dict:
        """
        Generate tags for a video using Gist API.

        Args:
            video_id: The TwelveLabs video ID
            index_id: The TwelveLabs index ID
            types: Types of tags to generate (topic, hashtag)

        Returns:
            Generated tags
        """
        if types is None:
            types = ["topic", "hashtag"]

        try:
            result = self.client.gist.create(
                video_id=video_id,
                types=types,
            )
            return {
                "topics": result.topics if hasattr(result, 'topics') else [],
                "hashtags": result.hashtags if hasattr(result, 'hashtags') else [],
            }
        except Exception as e:
            logger.error(f"Failed to generate tags: {e}")
            raise

    def hybrid_search(
        self,
        index_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        native_weight: float = 0.5,
    ) -> list[dict]:
        """
        Perform hybrid search combining native and embedding-based search.

        Args:
            index_id: The TwelveLabs index ID
            query: Natural language search query
            top_k: Number of results to return
            native_weight: Weight for native search results (0-1)

        Returns:
            Combined and re-ranked search results
        """
        # Get native search results
        native_results = self.search_native(index_id, query, top_k=top_k)

        # Get embedding search results
        embedding_results = self.search_with_embeddings(query, top_k=top_k)

        # Combine and re-rank results
        combined = self._merge_results(
            native_results.get("data", []),
            embedding_results,
            native_weight,
        )

        return combined[:top_k]

    def _merge_results(
        self,
        native_results: list,
        embedding_results: list,
        native_weight: float,
    ) -> list[dict]:
        """Merge and re-rank results from different search methods."""
        embedding_weight = 1 - native_weight

        # Create a combined score map
        score_map = {}

        # Process native results
        for result in native_results:
            video_id = result.get("video_id", "")
            score = result.get("confidence", 0) * native_weight
            if video_id in score_map:
                score_map[video_id]["score"] += score
            else:
                score_map[video_id] = {
                    "video_id": video_id,
                    "score": score,
                    "native_result": result,
                }

        # Process embedding results
        for result in embedding_results:
            video_id = result.get("metadata", {}).get("video_id", "")
            score = result.get("score", 0) * embedding_weight
            if video_id in score_map:
                score_map[video_id]["score"] += score
                score_map[video_id]["embedding_result"] = result
            else:
                score_map[video_id] = {
                    "video_id": video_id,
                    "score": score,
                    "embedding_result": result,
                }

        # Sort by combined score
        sorted_results = sorted(
            score_map.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return sorted_results
