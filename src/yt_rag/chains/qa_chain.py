"""Compose a multi-turn (history-aware) RAG chain with LangChain LCEL.

    {question, chat_history}
      -> input guard          (too long / injection / moderation / small talk -> canned reply, stop)
      -> standalone_question  (LLM rewrite of the follow-up; skipped when there is no history)
      -> docs                 (retriever on the standalone question)
      -> relevance gate       (no chunk above MIN_RERANK_SCORE -> NOT_FOUND_ANSWER, no LLM call)
      -> answer               (LLM with context + history + original question)
      -> citations            (CitationCheck: cited times that do / don't match a retrieved chunk)
      -> formatted_answer     (answer without inline markers + numbered **Citations** list)
"""

import re
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from yt_rag.chains.citations import CitationCheck, check_citations, with_citation_footer
from yt_rag.chains.guardrails import NO_RELEVANT_CONTEXT, GuardResult, InputGuard, has_relevant_context
from yt_rag.chains.prompts import NOT_FOUND_ANSWER
from yt_rag.ingestion.metadata import format_timestamp

# Tags the prompts use as delimiters. Untrusted text (transcripts, past answers) must not be
# able to open/close them and smuggle text outside the delimited block.
_DELIMITER_TAGS = re.compile(r"</?\s*(context|conversation)\s*>", re.IGNORECASE)

# Prefixes / quotes the rewriter sometimes adds despite instructions.
_REWRITE_PREFIX = re.compile(r"^\s*(standalone question|output|question)\s*:\s*", re.IGNORECASE)


def sanitize(text: str) -> str:
    """Remove prompt delimiter tags from untrusted text."""
    return _DELIMITER_TAGS.sub("", text)


def format_docs(retrieved_docs: list[Document]) -> str:
    """Join retrieved chunks into one context string, in video order, each with a time label:

        [12:34–13:32]
        so protein folding is the problem of ...
    """
    def start(doc: Document) -> float:
        return doc.metadata.get("start", 0.0)

    blocks = []
    for doc in sorted(retrieved_docs, key=start):
        meta = doc.metadata
        text = sanitize(doc.page_content)
        if "start" in meta and "end" in meta:
            label = f"[{format_timestamp(meta['start'])}–{format_timestamp(meta['end'])}]"
            blocks.append(f"{label}\n{text}")
        else:
            blocks.append(text)
    return "\n\n".join(blocks)


def format_history(messages: list[BaseMessage]) -> str:
    """Render chat history as plain text for the condense prompt."""
    names = {"human": "User", "ai": "Assistant"}
    lines = [f"{names.get(m.type, m.type)}: {sanitize(str(m.content))}" for m in messages]
    return "\n".join(lines)


def clean_rewrite(text: str) -> str:
    """Strip a leading 'Standalone question:' style prefix and wrapping quotes."""
    text = _REWRITE_PREFIX.sub("", text.strip())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text.strip('“”').strip()


def _stopped(x: dict, reason: str, reply: str) -> dict:
    """Chain output for a turn answered by a guardrail instead of the LLM (same keys as normal)."""
    return {
        **x,
        "standalone_question": x.get("standalone_question", x["question"]),
        "docs": x.get("docs", []),
        "answer": reply,
        "citations": CitationCheck(),
        "formatted_answer": reply,
        "guardrail": reason,
    }


def build_qa_chain(
    retriever: VectorStoreRetriever,
    condense_prompt: ChatPromptTemplate,
    qa_prompt: ChatPromptTemplate,
    llm: BaseChatModel,
    input_guard: InputGuard | None = None,
    min_rerank_score: float | None = None,
) -> Runnable:
    """Return the QA chain.

    `input_guard`: checks the question first (None = no input checks).
    `min_rerank_score`: relevance gate threshold on the reranker score (None = only the
    empty-retrieval check).

    Usage:
        result = chain.invoke({"question": "...", "chat_history": [HumanMessage(...), AIMessage(...)]})
        result["answer"], result["docs"], result["standalone_question"], result["citations"],
        result["formatted_answer"], result["guardrail"]  # None, or which guardrail produced the reply
    """
    parser = StrOutputParser()

    rewrite = (
        RunnablePassthrough.assign(conversation=lambda x: format_history(x["chat_history"]))
        | condense_prompt
        | llm
        | parser
        | clean_rewrite
    )
    standalone_question = RunnableBranch(
        (lambda x: not x.get("chat_history"), itemgetter("question")),  # first turn: no rewrite
        # fall back to the original question if the rewrite comes back empty
        RunnablePassthrough.assign(rewritten=rewrite) | (lambda x: x["rewritten"] or x["question"]),
    )

    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | qa_prompt
        | llm
        | parser
    )

    generate = (
        RunnablePassthrough.assign(answer=answer)
        .assign(citations=lambda x: check_citations(x["answer"], x["docs"]))
        .assign(formatted_answer=lambda x: with_citation_footer(x["answer"], x["citations"], x["docs"]))
        .assign(guardrail=lambda x: None)
    )

    rag = (
        RunnablePassthrough.assign(standalone_question=standalone_question)
        .assign(docs=itemgetter("standalone_question") | retriever)
        | RunnableBranch(
            (
                lambda x: not has_relevant_context(x["docs"], min_rerank_score),
                lambda x: _stopped(x, NO_RELEVANT_CONTEXT, NOT_FOUND_ANSWER),
            ),
            generate,
        )
    )

    check = input_guard or (lambda _q: GuardResult())
    return (
        RunnablePassthrough.assign(input_check=lambda x: check(x["question"]))
        | RunnableBranch(
            (
                lambda x: not x["input_check"].allowed,
                lambda x: _stopped(x, x["input_check"].reason, x["input_check"].reply),
            ),
            rag,
        )
    )
