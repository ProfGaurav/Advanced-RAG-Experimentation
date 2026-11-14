"""Question-answering service used by `app.py`.

`load_resources()` is expensive (loads FAISS + LLM) and is cached by the UI;
`build_chain_for()` is cheap and builds a chain restricted to one video.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from yt_rag import registry
from yt_rag.chains.llm import get_llm
from yt_rag.chains.prompts import get_qa_prompt
from yt_rag.chains.qa_chain import build_qa_chain
from yt_rag.chains.retriever import get_retriever
from yt_rag.config import get_settings
from yt_rag.vectorstore.embeddings import get_embeddings
from yt_rag.vectorstore.faiss_store import load_vectorstore_or_none


def load_resources() -> tuple[FAISS | None, BaseChatModel]:
    """Load the persisted FAISS store (None if empty) and the chat model.

    Also registers videos already in the store but missing from the registry.
    """
    settings = get_settings()
    store = load_vectorstore_or_none(get_embeddings(settings.embedding_model))
    registry.bootstrap_from_store(store)
    llm = get_llm(settings.chat_model, settings.temperature)
    return store, llm


def build_chain_for(store: FAISS, llm: BaseChatModel, video_id: str) -> Runnable:
    """QA chain whose retriever only sees chunks of `video_id`."""
    settings = get_settings()
    retriever = get_retriever(store, k=settings.retriever_k, video_id=video_id, fetch_k=settings.fetch_k)
    return build_qa_chain(retriever, get_qa_prompt(), llm)


def answer(chain: Runnable, question: str) -> str:
    """Invoke `chain` with `question` and return the answer text."""
    return chain.invoke(question)
