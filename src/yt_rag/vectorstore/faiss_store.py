"""Create, persist, load and append to the FAISS vector store.

A single shared index lives at:  <VECTORSTORE_DIR>/   (default: data/vectorstore/)
Every chunk carries `{"video_id": ...}` metadata, so one index holds many videos
and retrieval filters by video. Writes come from `indexing.py` (CLI or UI).
"""

import threading
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from yt_rag.config import get_settings
from yt_rag.exceptions import VectorStoreNotFoundError

# Streamlit serves sessions from threads: serialise writes to the shared index.
_write_lock = threading.Lock()


def index_path() -> Path:
    """Return the directory where the FAISS index is stored."""
    return get_settings().vectorstore_dir


def index_exists() -> bool:
    """Return True if a persisted index exists."""
    return (index_path() / "index.faiss").exists()


def build_vectorstore(docs: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed `docs` and build an in-memory FAISS store."""
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def save_vectorstore(store: FAISS) -> None:
    """Persist `store` to `index_path()` (writes index.faiss + index.pkl)."""
    path = index_path()
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def add_to_vectorstore(store: FAISS | None, docs: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed `docs`, add them to `store` (creating it if None), persist, and return the store."""
    with _write_lock:
        if store is None:
            store = build_vectorstore(docs, embeddings)
        else:
            store.add_documents(docs)
        save_vectorstore(store)
    return store


def delete_index() -> None:
    """Remove the persisted index files (keeps the directory and .gitkeep)."""
    for name in ("index.faiss", "index.pkl"):
        (index_path() / name).unlink(missing_ok=True)


def load_vectorstore(embeddings: Embeddings) -> FAISS:
    """Load the persisted FAISS index.

    Raises:
        VectorStoreNotFoundError: if the index has not been built yet.
    """
    if not index_exists():
        raise VectorStoreNotFoundError(
            f"No vector store found at {index_path()}. "
            "Add a video in the app, or run:  python scripts/build_index.py --video-id <ID>"
        )
    # index.pkl is a pickle; safe here because only this app writes it.
    return FAISS.load_local(str(index_path()), embeddings, allow_dangerous_deserialization=True)


def load_vectorstore_or_none(embeddings: Embeddings) -> FAISS | None:
    """Like `load_vectorstore`, but return None when no index exists yet."""
    return load_vectorstore(embeddings) if index_exists() else None
