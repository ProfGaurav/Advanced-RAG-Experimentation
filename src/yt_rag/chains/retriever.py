"""Build a retriever on top of a FAISS store."""

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(
    vector_store: FAISS,
    k: int = 4,
    video_id: str | None = None,
    fetch_k: int = 100,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """Return a retriever over `vector_store` that fetches the top-`k` chunks.

    If `video_id` is given, only chunks whose metadata `video_id` matches are returned.
    FAISS filters *after* the vector search, so `fetch_k` candidates are fetched first.
    """
    search_kwargs: dict = {"k": k}
    if video_id:
        search_kwargs.update(filter={"video_id": video_id}, fetch_k=fetch_k)
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
