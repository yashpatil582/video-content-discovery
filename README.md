# Video Content Discovery Engine

A RAG-powered video search system using TwelveLabs API with semantic search and evaluation benchmarks. Search *inside* videos using natural language queries.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Semantic Video Search** - Find specific moments in videos using natural language
- **Multi-modal Understanding** - Searches visual content, audio, and speech
- **Confidence Scoring** - Results ranked by relevance with confidence percentages
- **Timestamp Navigation** - Jump directly to matched video segments
- **Search Quality Metrics** - Built-in evaluation with MRR, Precision@K, latency tracking
- **FAISS Integration** - Local vector search for hybrid retrieval

## Demo

```
Query: "person explaining machine learning"
Results:
  ✓ Result 1 - Confidence: 89.2% - ⏱️ 3:24 - 3:45
  ✓ Result 2 - Confidence: 76.8% - ⏱️ 12:10 - 12:35
  ✓ Result 3 - Confidence: 71.3% - ⏱️ 0:45 - 1:02
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   Search Engine  │────▶│ TwelveLabs  │
│     UI      │     │   + Evaluation   │     │    API      │
└─────────────┘     └────────┬─────────┘     └─────────────┘
                             │
                    ┌────────▼─────────┐
                    │   FAISS Vector   │
                    │      Store       │
                    └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- TwelveLabs API key ([Sign up free](https://api.twelvelabs.io) - 600 min/month)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-content-discovery.git
cd video-content-discovery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your TWELVELABS_API_KEY
```

### Running the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

### 1. Connect to TwelveLabs
Enter your API key in the sidebar and click **Connect**.

### 2. Create an Index
Enter an index name and click **Create Index**. The index uses the `marengo2.7` model for search.

### 3. Upload Videos
Go to **Upload Videos** tab and either:
- Upload a video file directly
- Provide a public video URL

### 4. Search
Once videos are indexed, use the **Search** tab:
- Enter natural language queries like "person walking outdoors"
- Adjust confidence threshold and result count
- View matched segments with timestamps

### 5. Evaluate
The **Evaluation** tab provides search quality metrics:
- Mean Reciprocal Rank (MRR)
- Precision@K (K=1, 5, 10)
- Latency percentiles (P50, P95)

## Project Structure

```
video-content-discovery/
├── app.py                    # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── indexer.py           # TwelveLabs SDK integration
│   ├── search.py            # Semantic search engine
│   ├── embeddings.py        # FAISS vector management
│   └── evaluation.py        # Search quality benchmarks
├── config/
│   └── settings.py          # Configuration
├── tests/
│   └── test_search.py       # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## API Reference

### VideoIndexer

```python
from src.indexer import VideoIndexer

indexer = VideoIndexer(api_key="your_key")
index_id = indexer.create_index("my-videos")
task_id = indexer.upload_video("video.mp4", index_id)
indexer.wait_for_task(task_id)
```

### VideoSearchEngine

```python
from src.search import VideoSearchEngine

engine = VideoSearchEngine(api_key="your_key")
results = engine.search_native(
    index_id="your_index_id",
    query="person giving presentation",
    top_k=10
)

for r in results["data"]:
    print(f"{r['confidence']:.1f}% - {r['start']}s to {r['end']}s")
```

### SearchEvaluator

```python
from src.evaluation import SearchEvaluator, EvaluationQuery

evaluator = SearchEvaluator(search_engine, index_id)
queries = [
    EvaluationQuery("walking in park", ["vid1", "vid2"]),
    EvaluationQuery("cooking food", ["vid3"]),
]
result = evaluator.evaluate_query_set(queries)
print(evaluator.generate_report(result))
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MRR** | Mean Reciprocal Rank - position of first relevant result |
| **Precision@K** | Fraction of top-K results that are relevant |
| **Recall@K** | Fraction of relevant items found in top-K |
| **P50/P95 Latency** | Response time percentiles |

## Configuration

Key settings in `config/settings.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `INDEX_ENGINE` | TwelveLabs model | `marengo2.7` |
| `EMBEDDING_DIMENSION` | Vector dimension | `1024` |
| `DEFAULT_TOP_K` | Default results | `10` |
| `SEARCH_THRESHOLD` | Min confidence | `0.5` |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Use Cases

- **Media Libraries** - Search across thousands of hours of video content
- **Education** - Find specific topics in lecture recordings
- **Content Creation** - Locate clips for video editing
- **Compliance** - Find specific statements in recorded meetings
- **Sports Analytics** - Search for plays, goals, or events

## Tech Stack

- **[TwelveLabs](https://twelvelabs.io)** - Video understanding API
- **[FAISS](https://github.com/facebookresearch/faiss)** - Vector similarity search
- **[Streamlit](https://streamlit.io)** - Web UI framework
- **[LangChain](https://langchain.com)** - RAG orchestration

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- TwelveLabs for the video understanding API
- Facebook AI for FAISS
- Streamlit team for the UI framework
