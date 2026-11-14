"""Question-answering service: the only module `app.py` imports.

Loads the pre-built FAISS index (read-only) and answers questions against it.
"""

from langchain_core.runnables import Runnable

from yt_rag.chains.llm import get_llm
from yt_rag.chains.prompts import get_qa_prompt
from yt_rag.chains.qa_chain import build_qa_chain
from yt_rag.chains.retriever import get_retriever
from yt_rag.config import get_settings
from yt_rag.vectorstore.embeddings import get_embeddings
from yt_rag.vectorstore.faiss_store import load_vectorstore


def load_qa_chain() -> Runnable:
    """Load the persisted FAISS store, build retriever + QA chain.

    Intended to be wrapped in `st.cache_resource` by the UI so it runs once.

    Raises:
        VectorStoreNotFoundError: if `scripts/build_index.py` hasn't been run.
    """
    settings = get_settings()
    vector_store = load_vectorstore(get_embeddings(settings.embedding_model))
    retriever = get_retriever(vector_store, k=settings.retriever_k)
    llm = get_llm(settings.chat_model, settings.temperature)
    return build_qa_chain(retriever, get_qa_prompt(), llm)


def answer(chain: Runnable, question: str) -> str:
    """Invoke `chain` with `question` and return the answer text."""
    return chain.invoke(question)
