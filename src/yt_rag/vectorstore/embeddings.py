"""Embedding model factory."""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings(model: str = "text-embedding-3-small") -> Embeddings:
    """Return an `OpenAIEmbeddings` instance (reads OPENAI_API_KEY from the environment)."""
    embeddings = OpenAIEmbeddings(model=model)
    return embeddings
