"""Streamlit UI for Video Content Discovery Engine."""

import os
import sys
import logging
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PAGE_TITLE, PAGE_ICON, TWELVELABS_API_KEY
from src.indexer import VideoIndexer
from src.search import VideoSearchEngine
from src.embeddings import EmbeddingManager
from src.evaluation import SearchEvaluator, EvaluationQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-card {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "indexer" not in st.session_state:
        st.session_state.indexer = None
    if "search_engine" not in st.session_state:
        st.session_state.search_engine = None
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = None
    if "current_index_id" not in st.session_state:
        st.session_state.current_index_id = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "videos" not in st.session_state:
        st.session_state.videos = []


def init_services(api_key: str):
    """Initialize TwelveLabs services."""
    st.session_state.indexer = VideoIndexer(api_key)
    st.session_state.embedding_manager = EmbeddingManager(api_key)
    st.session_state.search_engine = VideoSearchEngine(
        api_key, st.session_state.embedding_manager
    )


def render_sidebar():
    """Render the sidebar configuration."""
    with st.sidebar:
        st.header("Configuration")

        # API Key input
        api_key = st.text_input(
            "TwelveLabs API Key",
            value=os.getenv("TWELVELABS_API_KEY", ""),
            type="password",
            help="Enter your TwelveLabs API key",
        )

        if api_key and st.button("Connect"):
            try:
                init_services(api_key)
                st.success("Connected to TwelveLabs API")
            except Exception as e:
                st.error(f"Connection failed: {e}")

        st.divider()

        # Index selection
        if st.session_state.indexer:
            st.subheader("Index Management")

            try:
                indexes = st.session_state.indexer.list_indexes()
                index_names = [idx["index_name"] for idx in indexes]

                if index_names:
                    selected_index = st.selectbox(
                        "Select Index",
                        options=index_names,
                    )

                    if selected_index:
                        for idx in indexes:
                            if idx["index_name"] == selected_index:
                                st.session_state.current_index_id = idx["_id"]
                                break
                else:
                    st.info("No indexes found. Create one below.")

                # Create new index
                new_index_name = st.text_input("New Index Name")
                if new_index_name and st.button("Create Index"):
                    try:
                        index_id = st.session_state.indexer.create_index(new_index_name)
                        st.success(f"Created index: {new_index_name}")
                        st.session_state.current_index_id = index_id
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create index: {e}")

            except Exception as e:
                st.error(f"Failed to list indexes: {e}")


def render_video_upload():
    """Render video upload section."""
    st.subheader("Upload Videos")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Upload from File**")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv", "webm"],
        )

        if uploaded_file and st.button("Upload File"):
            if not st.session_state.current_index_id:
                st.error("Please select an index first")
            else:
                with st.spinner("Uploading video..."):
                    try:
                        # Save temporarily
                        temp_path = Path(f"/tmp/{uploaded_file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        task_id = st.session_state.indexer.upload_video(
                            str(temp_path),
                            st.session_state.current_index_id,
                        )
                        st.success(f"Upload started! Task ID: {task_id}")
                        st.info("Video indexing may take a few minutes.")

                    except Exception as e:
                        st.error(f"Upload failed: {e}")

    with col2:
        st.write("**Upload from URL**")
        video_url = st.text_input("Video URL")

        if video_url and st.button("Upload URL"):
            if not st.session_state.current_index_id:
                st.error("Please select an index first")
            else:
                with st.spinner("Uploading video from URL..."):
                    try:
                        task_id = st.session_state.indexer.upload_video_url(
                            video_url,
                            st.session_state.current_index_id,
                        )
                        st.success(f"Upload started! Task ID: {task_id}")
                        st.info("Video indexing may take a few minutes.")

                    except Exception as e:
                        st.error(f"Upload failed: {e}")


def render_search():
    """Render search interface."""
    st.subheader("Search Videos")

    search_query = st.text_input(
        "Enter your search query",
        placeholder="e.g., person giving a presentation",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        search_type = st.selectbox(
            "Search Type",
            ["Native (TwelveLabs)", "Embedding (FAISS)", "Hybrid"],
        )

    with col2:
        top_k = st.slider("Number of Results", 1, 20, 10)

    with col3:
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    if st.button("Search", type="primary"):
        if not search_query:
            st.warning("Please enter a search query")
        elif not st.session_state.current_index_id:
            st.warning("Please select an index first")
        else:
            with st.spinner("Searching..."):
                try:
                    if search_type == "Native (TwelveLabs)":
                        results = st.session_state.search_engine.search_native(
                            st.session_state.current_index_id,
                            search_query,
                            top_k=top_k,
                            threshold=threshold,
                        )
                        st.session_state.search_results = results.get("data", [])
                        latency = results.get("latency_ms", 0)
                    elif search_type == "Embedding (FAISS)":
                        results = st.session_state.search_engine.search_with_embeddings(
                            search_query,
                            top_k=top_k,
                        )
                        st.session_state.search_results = results
                        latency = results[0].get("latency_ms", 0) if results else 0
                    else:
                        results = st.session_state.search_engine.hybrid_search(
                            st.session_state.current_index_id,
                            search_query,
                            top_k=top_k,
                        )
                        st.session_state.search_results = results
                        latency = 0

                    st.success(f"Found {len(st.session_state.search_results)} results in {latency:.2f}ms")

                except Exception as e:
                    st.error(f"Search failed: {e}")

    # Display results
    if st.session_state.search_results:
        st.write("### Results")

        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Show thumbnail if available
                    thumbnail_url = result.get('thumbnail_url')
                    if thumbnail_url:
                        st.image(thumbnail_url, use_container_width=True)
                    else:
                        st.write("🎬 No thumbnail")

                with col2:
                    confidence = result.get('confidence', result.get('score', 0))
                    start_time = result.get('start', 0)
                    end_time = result.get('end', 0)

                    # Format timestamps nicely
                    def format_time(seconds):
                        if isinstance(seconds, str):
                            seconds = float(seconds.replace('s', ''))
                        mins = int(seconds // 60)
                        secs = int(seconds % 60)
                        return f"{mins}:{secs:02d}"

                    st.markdown(f"**Result {i + 1}** - Confidence: **{confidence:.1f}%**")
                    st.write(f"⏱️ {format_time(start_time)} - {format_time(end_time)}")
                    st.caption(f"Video ID: {result.get('video_id', 'N/A')}")

                st.divider()


def render_video_list():
    """Render list of indexed videos."""
    st.subheader("Indexed Videos")

    if st.button("Refresh Videos"):
        if st.session_state.current_index_id:
            try:
                st.session_state.videos = st.session_state.indexer.list_videos(
                    st.session_state.current_index_id
                )
            except Exception as e:
                st.error(f"Failed to list videos: {e}")

    if st.session_state.videos:
        for video in st.session_state.videos:
            with st.expander(f"Video: {video.get('metadata', {}).get('filename', video.get('_id', 'Unknown'))}"):
                st.json(video)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Generate Summary", key=f"sum_{video['_id']}"):
                        with st.spinner("Generating summary..."):
                            try:
                                summary = st.session_state.search_engine.generate_summary(
                                    video["_id"],
                                    st.session_state.current_index_id,
                                )
                                st.write(summary)
                            except Exception as e:
                                st.error(f"Failed to generate summary: {e}")

                with col2:
                    if st.button(f"Generate Tags", key=f"tags_{video['_id']}"):
                        with st.spinner("Generating tags..."):
                            try:
                                tags = st.session_state.search_engine.generate_tags(
                                    video["_id"],
                                    st.session_state.current_index_id,
                                )
                                st.write(tags)
                            except Exception as e:
                                st.error(f"Failed to generate tags: {e}")
    else:
        st.info("No videos indexed yet. Upload some videos to get started!")


def render_evaluation():
    """Render evaluation benchmarks section."""
    st.subheader("Search Evaluation Benchmarks")

    st.write("""
    Run evaluation benchmarks to measure search quality and performance.
    You can either use sample queries or upload your own evaluation dataset.
    """)

    evaluator = SearchEvaluator(
        st.session_state.search_engine,
        st.session_state.current_index_id,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Sample Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    queries = evaluator.create_sample_queries()
                    result = evaluator.evaluate_query_set(queries, "native")

                    st.write("### Evaluation Results")

                    metrics_col1, metrics_col2 = st.columns(2)

                    with metrics_col1:
                        st.metric("MRR", f"{result.mrr:.4f}")
                        st.metric("Precision@1", f"{result.precision_at_1:.4f}")
                        st.metric("Precision@5", f"{result.precision_at_5:.4f}")

                    with metrics_col2:
                        st.metric("P50 Latency", f"{result.latency_p50:.2f}ms")
                        st.metric("P95 Latency", f"{result.latency_p95:.2f}ms")
                        st.metric("Mean Latency", f"{result.latency_mean:.2f}ms")

                    # Display full report
                    report = evaluator.generate_report(result)
                    st.code(report)

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    with col2:
        uploaded_eval = st.file_uploader(
            "Upload Evaluation Dataset (JSON)",
            type=["json"],
        )

        if uploaded_eval and st.button("Run Custom Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    import json
                    data = json.load(uploaded_eval)
                    queries = [
                        EvaluationQuery(
                            query=q["query"],
                            relevant_video_ids=q["relevant_video_ids"],
                        )
                        for q in data.get("queries", [])
                    ]

                    result = evaluator.evaluate_query_set(queries, "native")
                    report = evaluator.generate_report(result)
                    st.code(report)

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")


def main():
    """Main application entry point."""
    init_session_state()

    st.markdown('<div class="main-header">Video Content Discovery Engine</div>', unsafe_allow_html=True)
    st.write("A RAG-powered video search system using TwelveLabs API with semantic search and evaluation benchmarks.")

    render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Search",
        "Upload Videos",
        "Video Library",
        "Evaluation",
    ])

    with tab1:
        if st.session_state.indexer:
            render_search()
        else:
            st.info("Please connect to TwelveLabs API using the sidebar.")

    with tab2:
        if st.session_state.indexer:
            render_video_upload()
        else:
            st.info("Please connect to TwelveLabs API using the sidebar.")

    with tab3:
        if st.session_state.indexer:
            render_video_list()
        else:
            st.info("Please connect to TwelveLabs API using the sidebar.")

    with tab4:
        if st.session_state.indexer:
            render_evaluation()
        else:
            st.info("Please connect to TwelveLabs API using the sidebar.")


if __name__ == "__main__":
    main()
