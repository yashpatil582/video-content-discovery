"""Embedding generation and management with FAISS."""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np
import requests

from config.settings import (
    TWELVELABS_API_KEY,
    TWELVELABS_API_URL,
    FAISS_INDEX_PATH,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings and FAISS vector store."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_path: Optional[Path] = None,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        self.api_key = api_key or TWELVELABS_API_KEY
        self.api_url = TWELVELABS_API_URL
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        self.index_path = index_path or FAISS_INDEX_PATH
        self.dimension = dimension
        self.faiss_index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []

        self._ensure_index_path()

    def _ensure_index_path(self):
        """Ensure the index path directory exists."""
        self.index_path.mkdir(parents=True, exist_ok=True)

    def create_faiss_index(self, use_ivf: bool = False, nlist: int = 100) -> faiss.Index:
        """Create a new FAISS index."""
        if use_ivf:
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            # Flat L2 index for smaller datasets (exact search)
            self.faiss_index = faiss.IndexFlatL2(self.dimension)

        logger.info(f"Created FAISS index with dimension {self.dimension}")
        return self.faiss_index

    def get_text_embedding(self, text: str, engine_name: str = "Marengo-retrieval-2.6") -> np.ndarray:
        """Generate embedding for text query using TwelveLabs Embed API."""
        url = f"{self.api_url}/embed"
        payload = {
            "engine_name": engine_name,
            "text": text,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        embedding_data = response.json()

        # Handle different response structures
        if "text_embedding" in embedding_data:
            embedding = embedding_data["text_embedding"]["segments"][0]["embeddings_float"]
        elif "video_embedding" in embedding_data:
            embedding = embedding_data["video_embedding"]["segments"][0]["embeddings_float"]
        else:
            raise ValueError(f"Unexpected embedding response format: {embedding_data.keys()}")

        return np.array(embedding, dtype=np.float32)

    def get_video_embedding(
        self,
        video_url: Optional[str] = None,
        video_file: Optional[str] = None,
        engine_name: str = "Marengo-retrieval-2.6",
    ) -> list[dict]:
        """Generate embeddings for video using TwelveLabs Embed API."""
        url = f"{self.api_url}/embed"

        if video_url:
            payload = {
                "engine_name": engine_name,
                "video_url": video_url,
            }
            response = requests.post(url, json=payload, headers=self.headers)
        elif video_file:
            with open(video_file, "rb") as f:
                files = {"video_file": f}
                data = {"engine_name": engine_name}
                headers = {"x-api-key": self.api_key}
                response = requests.post(url, files=files, data=data, headers=headers)
        else:
            raise ValueError("Either video_url or video_file must be provided")

        response.raise_for_status()
        embedding_data = response.json()

        # Extract segment embeddings
        segments = embedding_data.get("video_embedding", {}).get("segments", [])
        return [
            {
                "start_offset_sec": seg.get("start_offset_sec", 0),
                "end_offset_sec": seg.get("end_offset_sec", 0),
                "embedding": np.array(seg["embeddings_float"], dtype=np.float32),
            }
            for seg in segments
        ]

    def add_embeddings(
        self,
        embeddings: Union[np.ndarray, list[np.ndarray]],
        metadata: list[dict],
    ):
        """Add embeddings to the FAISS index with metadata."""
        if self.faiss_index is None:
            self.create_faiss_index()

        if isinstance(embeddings, list):
            embeddings = np.vstack(embeddings)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.faiss_index.add(embeddings)
        self.metadata.extend(metadata)

        logger.info(f"Added {len(metadata)} embeddings to index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Search the FAISS index for similar embeddings."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)

        distances, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for unfound
                continue
            result = {
                "rank": i + 1,
                "score": float(1 - dist),  # Convert L2 distance to similarity
                "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
            }
            results.append(result)

        return results

    def save_index(self, name: str = "default"):
        """Save the FAISS index and metadata to disk."""
        if self.faiss_index is None:
            raise ValueError("No index to save")

        index_file = self.index_path / f"{name}.index"
        metadata_file = self.index_path / f"{name}_metadata.json"

        faiss.write_index(self.faiss_index, str(index_file))
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved index to {index_file}")

    def load_index(self, name: str = "default") -> bool:
        """Load a FAISS index and metadata from disk."""
        index_file = self.index_path / f"{name}.index"
        metadata_file = self.index_path / f"{name}_metadata.json"

        if not index_file.exists():
            logger.warning(f"Index file not found: {index_file}")
            return False

        self.faiss_index = faiss.read_index(str(index_file))

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        logger.info(f"Loaded index with {self.faiss_index.ntotal} vectors")
        return True

    def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        if self.faiss_index is None:
            return {"status": "not_initialized"}

        return {
            "status": "active",
            "total_vectors": self.faiss_index.ntotal,
            "dimension": self.dimension,
            "metadata_count": len(self.metadata),
        }
