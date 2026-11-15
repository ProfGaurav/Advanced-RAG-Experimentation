"""Create, persist, load and append to the FAISS vector store.

A single shared index lives at:  <VECTORSTORE_DIR>/   (default: data/vectorstore/)
    index.faiss + index.pkl   FAISS vectors + docstore
    manifest.json             schema version + embedding model the index was built with

Every chunk carries `{"video_id", "start", "end", ...}` metadata, so one index holds many
videos and retrieval filters by video. Writes come from `indexing.py` (CLI or UI).
"""

import json
import threading
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from yt_rag.config import get_settings
from yt_rag.exceptions import IndexOutdatedError, VectorStoreNotFoundError

# Bump when chunk metadata changes in a way old indexes can't satisfy.
#   1 = no manifest (plain text chunks)   2 = timestamped chunks (start/end metadata)
SCHEMA_VERSION = 2
_MANIFEST = "manifest.json"
_REINDEX_HINT = "Re-index with:  python scripts/build_index.py --reindex-all"

# Streamlit serves sessions from threads: serialise writes to the shared index.
_write_lock = threading.Lock()


def index_path() -> Path:
    """Return the directory where the FAISS index is stored."""
    return get_settings().vectorstore_dir


def index_exists() -> bool:
    """Return True if a persisted index exists."""
    return (index_path() / "index.faiss").exists()


def _read_manifest() -> dict:
    path = index_path() / _MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1}


def _check_manifest() -> None:
    """Refuse to load an index built with another schema or embedding model."""
    manifest = _read_manifest()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise IndexOutdatedError(
            f"The vector store uses schema v{manifest.get('schema_version')} (need v{SCHEMA_VERSION}, "
            f"which adds timestamps). {_REINDEX_HINT}"
        )
    model = get_settings().embedding_model
    if manifest.get("embedding_model") != model:
        raise IndexOutdatedError(
            f"The vector store was built with {manifest.get('embedding_model')!r} but EMBEDDING_MODEL "
            f"is {model!r}. {_REINDEX_HINT}"
        )


def build_vectorstore(docs: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed `docs` and build an in-memory FAISS store."""
    return FAISS.from_documents(docs, embeddings)


def save_vectorstore(store: FAISS) -> None:
    """Persist `store` to `index_path()` (index.faiss + index.pkl + manifest.json)."""
    path = index_path()
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    manifest = {"schema_version": SCHEMA_VERSION, "embedding_model": get_settings().embedding_model}
    (path / _MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    for name in ("index.faiss", "index.pkl", _MANIFEST):
        (index_path() / name).unlink(missing_ok=True)


def load_vectorstore(embeddings: Embeddings) -> FAISS:
    """Load the persisted FAISS index.

    Raises:
        VectorStoreNotFoundError: if the index has not been built yet.
        IndexOutdatedError: if it was built with another schema or embedding model.
    """
    if not index_exists():
        raise VectorStoreNotFoundError(
            f"No vector store found at {index_path()}. "
            "Add a video in the app, or run:  python scripts/build_index.py --video-id <ID>"
        )
    _check_manifest()
    # index.pkl is a pickle; safe here because only this app writes it.
    return FAISS.load_local(str(index_path()), embeddings, allow_dangerous_deserialization=True)


def load_vectorstore_or_none(embeddings: Embeddings) -> FAISS | None:
    """Like `load_vectorstore`, but return None when no index exists yet."""
    return load_vectorstore(embeddings) if index_exists() else None
