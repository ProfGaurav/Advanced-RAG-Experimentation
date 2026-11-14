"""Create, persist and load the FAISS vector store.

A single index lives at:  <VECTORSTORE_DIR>/   (default: data/vectorstore/)
Writing happens only in the backend (`indexing.py`); the UI only loads it.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from yt_rag.config import get_settings
from yt_rag.exceptions import VectorStoreNotFoundError


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


def load_vectorstore(embeddings: Embeddings) -> FAISS:
    """Load the persisted FAISS index.

    Raises:
        VectorStoreNotFoundError: if the index has not been built yet.
    """
    if not index_exists():
        raise VectorStoreNotFoundError(
            f"No vector store found at {index_path()}. "
            "Build it first:  python scripts/build_index.py --video-id <ID>"
        )
    # index.pkl is a pickle; safe here because only our own build script writes it.
    return FAISS.load_local(str(index_path()), embeddings, allow_dangerous_deserialization=True)
