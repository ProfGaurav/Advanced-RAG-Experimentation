"""Central configuration loaded from environment variables / `.env`.

Holds settings such as:
    - OPENAI_API_KEY
    - VIDEO_ID             (default video for scripts/build_index.py)
    - EMBEDDING_MODEL      (e.g. text-embedding-3-small)
    - CHAT_MODEL           (e.g. gpt-4o-mini)
    - TEMPERATURE
    - CHUNK_SIZE / CHUNK_OVERLAP
    - RETRIEVER_K          (number of chunks to retrieve)
    - FETCH_K              (candidates fetched before filtering by video_id)
    - HISTORY_TURNS        (past user/assistant exchanges sent to the LLM, default 6)
    - RERANKER             ("cross_encoder" or "none")
    - RERANKER_MODEL       (cross-encoder model, default cross-encoder/ms-marco-MiniLM-L-6-v2)
    - RERANK_CANDIDATES    (chunks fetched from FAISS before reranking down to RETRIEVER_K)
    - MIN_RERANK_SCORE     (relevance gate: refuse without calling the LLM below this, default -6;
                            empty = off. Only applies when a reranker is configured)
    - MAX_QUESTION_CHARS   (longer questions are rejected, default 1000)
    - MODERATION           ("openai" = OpenAI moderation endpoint on each question, or "none")
    - VECTORSTORE_DIR      (where the FAISS index is persisted, default data/vectorstore)
    - REGISTRY_PATH        (JSON list of indexed videos, default data/videos.json)
    - CHAT_SESSIONS_PATH   (persisted chat sessions, default data/chat_sessions.json)
    - TRANSCRIPTS_DIR      (cached timestamped transcripts, default data/transcripts)
    - TRANSCRIPT_LANGUAGES (preferred transcript languages, e.g. ["en"])
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root = rag/  (this file is rag/src/yt_rag/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load rag/.env into os.environ as soon as this module is imported,
# regardless of which folder the script is run from.
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(value: str) -> Path:
    """Resolve a relative path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_float(value: str) -> float | None:
    """"" / "none" -> None, else float."""
    value = value.strip()
    return None if value.lower() in ("", "none") else float(value)


@dataclass(frozen=True)
class Settings:
    """Typed, immutable application settings."""

    video_id: str
    embedding_model: str
    chat_model: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    fetch_k: int
    history_turns: int
    reranker: str
    reranker_model: str
    rerank_candidates: int
    min_rerank_score: float | None
    max_question_chars: int
    moderation: str
    vectorstore_dir: Path
    registry_path: Path
    chat_sessions_path: Path
    transcripts_dir: Path
    transcript_languages: list[str]


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance built from environment variables."""
    return Settings(
        video_id=os.getenv("VIDEO_ID", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retriever_k=int(os.getenv("RETRIEVER_K", "4")),
        fetch_k=int(os.getenv("FETCH_K", "100")),
        history_turns=int(os.getenv("HISTORY_TURNS", "6")),
        reranker=os.getenv("RERANKER", "cross_encoder").strip().lower(),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_candidates=int(os.getenv("RERANK_CANDIDATES", "20")),
        min_rerank_score=_optional_float(os.getenv("MIN_RERANK_SCORE", "-6")),
        max_question_chars=int(os.getenv("MAX_QUESTION_CHARS", "1000")),
        moderation=os.getenv("MODERATION", "openai").strip().lower(),
        vectorstore_dir=_project_path(os.getenv("VECTORSTORE_DIR", "data/vectorstore")),
        registry_path=_project_path(os.getenv("REGISTRY_PATH", "data/videos.json")),
        chat_sessions_path=_project_path(os.getenv("CHAT_SESSIONS_PATH", "data/chat_sessions.json")),
        transcripts_dir=_project_path(os.getenv("TRANSCRIPTS_DIR", "data/transcripts")),
        transcript_languages=os.getenv("TRANSCRIPT_LANGUAGES", "en").split(","),
    )
