"""Build a retriever on top of a FAISS store."""

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(vector_store: FAISS, k: int = 4, search_type: str = "similarity") -> VectorStoreRetriever:
    """Return a retriever over `vector_store` that fetches the top-`k` chunks."""
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
    return retriever
