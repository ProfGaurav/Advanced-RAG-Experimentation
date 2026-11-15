"""Second-stage reranking: retrieve many candidates cheaply, then re-score them with a cross-encoder.

    query -> FAISS (top RERANK_CANDIDATES, filtered by video) -> cross-encoder scores
          -> top RETRIEVER_K by score (each doc gets metadata `rerank_score`)

A bi-encoder (the embedding model) compares query and chunk vectors computed separately;
a cross-encoder reads the (query, chunk) pair together, which is slower but more precise,
so it's only applied to a small candidate set.

Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB, runs on CPU, English).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

if TYPE_CHECKING:  # type hints only; the real import stays lazy (it pulls in torch)
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseDocumentCompressor):
    """Re-scores documents against the query with a sentence-transformers `CrossEncoder`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: "CrossEncoder"
    top_n: int = 4

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Any = None  # noqa: ARG002 (interface)
    ) -> Sequence[Document]:
        if not documents:
            return []
        scores = self.model.predict([(query, d.page_content) for d in documents], show_progress_bar=False)
        ranked = sorted(zip(documents, scores), key=lambda pair: float(pair[1]), reverse=True)
        out = []
        for doc, score in ranked[: self.top_n]:
            meta = {**doc.metadata, "rerank_score": round(float(score), 4)}
            out.append(Document(page_content=doc.page_content, metadata=meta, id=doc.id))
        return out


class RerankingRetriever(BaseRetriever):
    """`base_retriever` fetches candidates, `reranker` keeps the best ones."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever
    reranker: BaseDocumentCompressor

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        candidates = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        return list(self.reranker.compress_documents(candidates, query))


def load_reranker(name: str, model_name: str, top_n: int) -> BaseDocumentCompressor | None:
    """Return the configured reranker, or None when `name` is "none".

    Loading downloads the model on first use (cached by Hugging Face) - do it once per process.
    """
    if name == "none":
        return None
    if name == "cross_encoder":
        import os

        # HF cache falls back to copies on Windows without Developer Mode; the warning is noise.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from sentence_transformers import CrossEncoder  # heavy import, only when enabled

        return CrossEncoderReranker(model=CrossEncoder(model_name or DEFAULT_RERANKER_MODEL), top_n=top_n)
    raise ValueError(f"Unknown RERANKER {name!r} (expected 'none' or 'cross_encoder')")
