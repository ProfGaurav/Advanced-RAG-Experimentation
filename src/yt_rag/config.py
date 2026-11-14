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
    - VECTORSTORE_DIR      (where the FAISS index is persisted, default data/vectorstore)
    - REGISTRY_PATH        (JSON list of indexed videos, default data/videos.json)
    - CHAT_SESSIONS_PATH   (persisted chat sessions, default data/chat_sessions.json)
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
    vectorstore_dir: Path
    registry_path: Path
    chat_sessions_path: Path
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
        vectorstore_dir=_project_path(os.getenv("VECTORSTORE_DIR", "data/vectorstore")),
        registry_path=_project_path(os.getenv("REGISTRY_PATH", "data/videos.json")),
        chat_sessions_path=_project_path(os.getenv("CHAT_SESSIONS_PATH", "data/chat_sessions.json")),
        transcript_languages=os.getenv("TRANSCRIPT_LANGUAGES", "en").split(","),
    )
