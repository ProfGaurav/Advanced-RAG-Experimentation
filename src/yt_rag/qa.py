"""Question-answering service used by `app.py`.

`load_resources()` is expensive (loads FAISS + LLM + reranker model) and is cached by the UI;
`build_chain_for()` is cheap and builds a chain restricted to one video.
"""

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from yt_rag import registry
from yt_rag.chains.llm import get_llm
from yt_rag.chains.prompts import get_condense_prompt, get_qa_prompt
from yt_rag.chains.qa_chain import build_qa_chain
from yt_rag.chains.reranker import RerankingRetriever, load_reranker
from yt_rag.chains.retriever import get_retriever
from yt_rag.config import get_settings
from yt_rag.vectorstore.embeddings import get_embeddings
from yt_rag.vectorstore.faiss_store import load_vectorstore_or_none


@dataclass
class Resources:
    """Long-lived objects shared by every chat (store is replaced when videos are added)."""

    store: FAISS | None
    llm: BaseChatModel
    reranker: BaseDocumentCompressor | None


def load_resources() -> Resources:
    """Load the persisted FAISS store (None if empty), the chat model and the reranker.

    Also registers videos already in the store but missing from the registry.
    """
    settings = get_settings()
    store = load_vectorstore_or_none(get_embeddings(settings.embedding_model))
    registry.bootstrap_from_store(store)
    llm = get_llm(settings.chat_model, settings.temperature)
    reranker = load_reranker(settings.reranker, settings.reranker_model, top_n=settings.retriever_k)
    return Resources(store, llm, reranker)


def build_chain_for(resources: Resources, video_id: str) -> Runnable:
    """Multi-turn QA chain whose retriever only sees chunks of `video_id`.

    With a reranker: FAISS fetches RERANK_CANDIDATES chunks, the reranker keeps RETRIEVER_K.
    Without one:     FAISS returns RETRIEVER_K chunks directly.
    """
    settings = get_settings()
    k = settings.rerank_candidates if resources.reranker else settings.retriever_k
    retriever = get_retriever(
        resources.store, k=k, video_id=video_id, fetch_k=max(settings.fetch_k, k)
    )
    if resources.reranker:
        retriever = RerankingRetriever(base_retriever=retriever, reranker=resources.reranker)
    return build_qa_chain(retriever, get_condense_prompt(), get_qa_prompt(), resources.llm)


def to_chat_history(messages: list[dict], max_turns: int | None = None) -> list[BaseMessage]:
    """Convert stored `{"role", "content"}` dicts to LangChain messages.

    Keeps only the last `max_turns` user/assistant exchanges (default: HISTORY_TURNS).
    """
    max_turns = get_settings().history_turns if max_turns is None else max_turns
    recent = messages[-2 * max_turns:] if max_turns > 0 else []
    role_to_cls = {"user": HumanMessage, "assistant": AIMessage}
    return [role_to_cls[m["role"]](content=m["content"]) for m in recent if m["role"] in role_to_cls]


def answer(chain: Runnable, question: str, history: list[dict] | None = None) -> str:
    """Invoke `chain` with `question` and the prior conversation; return the answer text."""
    result = chain.invoke({"question": question, "chat_history": to_chat_history(history or [])})
    return result["answer"]
