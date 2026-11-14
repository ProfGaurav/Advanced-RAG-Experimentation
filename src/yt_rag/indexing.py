"""Backend-only indexing pipeline (used by `scripts/build_index.py`, never by the UI).

    video ID -> transcript -> chunks -> OpenAI embeddings -> FAISS (saved to disk)
"""

from yt_rag.config import get_settings
from yt_rag.ingestion.splitter import split_transcript
from yt_rag.ingestion.transcript import fetch_transcript
from yt_rag.vectorstore.embeddings import get_embeddings
from yt_rag.vectorstore.faiss_store import build_vectorstore, index_exists, save_vectorstore


def build_index(video_id: str, force_rebuild: bool = False) -> int:
    """Ingest and index a video.

    Skips work if an index already exists (unless `force_rebuild`).
    Returns the number of chunks indexed (0 if skipped).
    """
    if index_exists() and not force_rebuild:
        return 0

    settings = get_settings()
    transcript = fetch_transcript(video_id, languages=settings.transcript_languages)
    chunks = split_transcript(
        transcript,
        video_id,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vector_store = build_vectorstore(chunks, get_embeddings(settings.embedding_model))
    save_vectorstore(vector_store)
    return len(chunks)
