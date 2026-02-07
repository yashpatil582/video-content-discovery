"""Video indexing with TwelveLabs API."""

import logging
import time
from typing import Optional
from pathlib import Path

from twelvelabs import TwelveLabs

from config.settings import (
    TWELVELABS_API_KEY,
    INDEX_NAME,
    INDEX_ENGINE,
    SUPPORTED_VIDEO_FORMATS,
)

logger = logging.getLogger(__name__)


class VideoIndexer:
    """Handles video indexing operations with TwelveLabs API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or TWELVELABS_API_KEY
        self.client = TwelveLabs(api_key=self.api_key)
        self.index_id: Optional[str] = None

    def create_index(self, index_name: str = INDEX_NAME) -> str:
        """Create a new index for video content."""
        try:
            # Check if index already exists
            existing = self.get_index_by_name(index_name)
            if existing:
                logger.info(f"Index '{index_name}' already exists with ID: {existing}")
                return existing

            # Create new index
            index = self.client.indexes.create(
                index_name=index_name,
                models=[
                    {
                        "model_name": INDEX_ENGINE,
                        "model_options": ["visual", "audio"],
                    }
                ],
            )
            self.index_id = index.id
            logger.info(f"Created index '{index_name}' with ID: {self.index_id}")
            return self.index_id

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def get_index_by_name(self, index_name: str) -> Optional[str]:
        """Get index ID by name."""
        try:
            indexes = self.client.indexes.list()
            for index in indexes:
                if index.index_name == index_name:
                    self.index_id = index.id
                    return self.index_id
            return None
        except Exception as e:
            logger.error(f"Failed to get index by name: {e}")
            return None

    def list_indexes(self) -> list:
        """List all available indexes."""
        try:
            indexes = self.client.indexes.list()
            return [
                {
                    "_id": idx.id,
                    "index_name": idx.index_name,
                    "created_at": str(idx.created_at) if hasattr(idx, 'created_at') else None,
                }
                for idx in indexes
            ]
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            raise

    def upload_video(
        self,
        video_path: str,
        index_id: Optional[str] = None,
        language: str = "en",
    ) -> str:
        """Upload a video for indexing."""
        index_id = index_id or self.index_id
        if not index_id:
            raise ValueError("No index ID provided. Create or select an index first.")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")

        try:
            task = self.client.tasks.create(
                index_id=index_id,
                video_file=str(video_path),
            )
            logger.info(f"Video upload started with task ID: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            raise

    def upload_video_url(
        self,
        video_url: str,
        index_id: Optional[str] = None,
        language: str = "en",
    ) -> str:
        """Upload a video from URL for indexing."""
        index_id = index_id or self.index_id
        if not index_id:
            raise ValueError("No index ID provided. Create or select an index first.")

        try:
            task = self.client.tasks.create(
                index_id=index_id,
                video_url=video_url,
            )
            logger.info(f"Video URL upload started with task ID: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"Failed to upload video from URL: {e}")
            raise

    def get_task_status(self, task_id: str) -> dict:
        """Get the status of an indexing task."""
        try:
            task = self.client.tasks.retrieve(task_id)
            return {
                "id": task.id,
                "status": task.status,
                "video_id": getattr(task, 'video_id', None),
            }
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            raise

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: int = 10,
        timeout: int = 600,
    ) -> dict:
        """Wait for a task to complete with polling."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            task_status = status.get("status")

            if task_status == "ready":
                logger.info(f"Task {task_id} completed successfully")
                return status
            elif task_status == "failed":
                raise RuntimeError(f"Task {task_id} failed: {status}")

            logger.info(f"Task {task_id} status: {task_status}, waiting...")
            time.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")

    def list_videos(self, index_id: Optional[str] = None) -> list:
        """List all videos in an index."""
        index_id = index_id or self.index_id
        if not index_id:
            raise ValueError("No index ID provided.")

        try:
            videos = self.client.indexes.videos.list(index_id)
            return [
                {
                    "_id": vid.id,
                    "metadata": {
                        "filename": getattr(vid, 'metadata', {}).get('filename', vid.id) if hasattr(vid, 'metadata') else vid.id,
                    },
                    "created_at": str(vid.created_at) if hasattr(vid, 'created_at') else None,
                }
                for vid in videos
            ]
        except Exception as e:
            logger.error(f"Failed to list videos: {e}")
            raise

    def get_video_info(self, index_id: str, video_id: str) -> dict:
        """Get detailed information about a video."""
        try:
            video = self.client.indexes.videos.retrieve(index_id, video_id)
            return {
                "id": video.id,
                "metadata": video.metadata if hasattr(video, 'metadata') else {},
            }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise

    def delete_video(self, index_id: str, video_id: str) -> bool:
        """Delete a video from an index."""
        try:
            self.client.indexes.videos.delete(index_id, video_id)
            logger.info(f"Deleted video {video_id} from index {index_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete video: {e}")
            raise
