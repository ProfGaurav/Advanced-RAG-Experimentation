"""Central configuration loaded from environment variables / `.env`.

Holds settings such as:
    - OPENAI_API_KEY
    - VIDEO_ID             (default video for scripts/build_index.py)
    - EMBEDDING_MODEL      (e.g. text-embedding-3-small)
    - CHAT_MODEL           (e.g. gpt-4o-mini)
    - TEMPERATURE
    - CHUNK_SIZE / CHUNK_OVERLAP
    - RETRIEVER_K          (number of chunks to retrieve)
    - VECTORSTORE_DIR      (where the FAISS index is persisted, default data/vectorstore)
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
    vectorstore_dir: Path
    transcript_languages: list[str]


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance built from environment variables."""
    vectorstore_dir = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
    if not vectorstore_dir.is_absolute():
        vectorstore_dir = PROJECT_ROOT / vectorstore_dir

    return Settings(
        video_id=os.getenv("VIDEO_ID", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retriever_k=int(os.getenv("RETRIEVER_K", "4")),
        vectorstore_dir=vectorstore_dir,
        transcript_languages=os.getenv("TRANSCRIPT_LANGUAGES", "en").split(","),
    )
