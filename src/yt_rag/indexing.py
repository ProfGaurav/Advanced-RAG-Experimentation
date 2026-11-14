"""Ingestion pipeline (used by the Streamlit UI and `scripts/build_index.py`).

    link / video ID -> transcript -> chunks (metadata video_id) -> OpenAI embeddings
        -> appended to the shared FAISS store (saved to disk) -> registered in videos.json
"""

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from yt_rag import registry
from yt_rag.config import get_settings
from yt_rag.ingestion.metadata import fetch_video_title
from yt_rag.ingestion.splitter import split_transcript
from yt_rag.ingestion.transcript import fetch_transcript
from yt_rag.ingestion.video_url import extract_video_id
from yt_rag.vectorstore.embeddings import get_embeddings
from yt_rag.vectorstore.faiss_store import add_to_vectorstore, delete_index, load_vectorstore_or_none


@dataclass
class IngestResult:
    """Outcome of `ingest_video`."""

    store: FAISS | None
    video_id: str
    title: str
    n_chunks: int  # 0 when the video was already indexed
    already_indexed: bool


def _chunks_for(video_id: str) -> list[Document]:
    """Fetch the transcript and split it into chunks tagged with `video_id`."""
    settings = get_settings()
    transcript = fetch_transcript(video_id, languages=settings.transcript_languages)
    return split_transcript(
        transcript,
        video_id,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def ingest_video(store: FAISS | None, url_or_id: str) -> IngestResult:
    """Add one video to `store` (created if None) unless it's already indexed.

    Raises:
        InvalidVideoURLError: if no video ID can be extracted.
        TranscriptUnavailableError: if the video has no usable transcript.
    """
    video_id = extract_video_id(url_or_id)
    known = registry.load_registry()
    if video_id in known:
        return IngestResult(store, video_id, known[video_id]["title"], 0, True)

    chunks = _chunks_for(video_id)
    store = add_to_vectorstore(store, chunks, get_embeddings(get_settings().embedding_model))
    title = fetch_video_title(video_id)
    registry.add_video(video_id, title, len(chunks))
    return IngestResult(store, video_id, title, len(chunks), False)


def build_index(video_id: str, force_rebuild: bool = False) -> int:
    """CLI helper: add `video_id` to the shared index.

    With `force_rebuild`, the existing index and registry are wiped first.
    Returns the number of chunks indexed (0 if the video was already indexed).
    """
    if force_rebuild:
        delete_index()
        registry.clear_registry()
    store = load_vectorstore_or_none(get_embeddings(get_settings().embedding_model))
    registry.bootstrap_from_store(store)
    return ingest_video(store, video_id).n_chunks
