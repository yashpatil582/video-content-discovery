"""Configuration settings for Video Content Discovery Engine."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# TwelveLabs API Configuration
TWELVELABS_API_KEY = os.getenv("TWELVELABS_API_KEY", "")
TWELVELABS_API_URL = "https://api.twelvelabs.io/v1.3"

# Index Configuration
INDEX_NAME = os.getenv("TWELVELABS_INDEX_NAME", "video-content-discovery")
INDEX_ENGINE = "marengo2.7"  # TwelveLabs search engine

# FAISS Configuration
FAISS_INDEX_PATH = Path("data/faiss_index")
EMBEDDING_DIMENSION = 1024  # TwelveLabs embedding dimension

# Search Configuration
DEFAULT_TOP_K = 10
SEARCH_THRESHOLD = 0.5

# Video Processing
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
MAX_VIDEO_DURATION_SECONDS = 3600  # 1 hour max

# Streamlit Configuration
PAGE_TITLE = "Video Content Discovery Engine"
PAGE_ICON = "🎬"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Evaluation Configuration
EVALUATION_METRICS = ["mrr", "precision_at_k", "latency_p50", "latency_p95"]
