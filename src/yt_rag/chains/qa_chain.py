"""Compose a multi-turn (history-aware) RAG chain with LangChain LCEL.

    {question, chat_history}
      -> standalone_question  (LLM rewrite of the follow-up; skipped when there is no history)
      -> docs                 (retriever on the standalone question)
      -> answer               (LLM with context + history + original question)
"""

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from yt_rag.ingestion.metadata import format_timestamp


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
        if "start" in meta and "end" in meta:
            label = f"[{format_timestamp(meta['start'])}–{format_timestamp(meta['end'])}]"
            blocks.append(f"{label}\n{doc.page_content}")
        else:
            blocks.append(doc.page_content)
    return "\n\n".join(blocks)


def build_qa_chain(
    retriever: VectorStoreRetriever,
    condense_prompt: ChatPromptTemplate,
    qa_prompt: ChatPromptTemplate,
    llm: BaseChatModel,
) -> Runnable:
    """Return the QA chain.

    Usage:
        result = chain.invoke({"question": "...", "chat_history": [HumanMessage(...), AIMessage(...)]})
        result["answer"], result["docs"], result["standalone_question"]
    """
    parser = StrOutputParser()

    standalone_question = RunnableBranch(
        (lambda x: not x.get("chat_history"), itemgetter("question")),  # first turn: no rewrite
        condense_prompt | llm | parser,
    )

    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | qa_prompt
        | llm
        | parser
    )

    return (
        RunnablePassthrough.assign(standalone_question=standalone_question)
        .assign(docs=itemgetter("standalone_question") | retriever)
        .assign(answer=answer)
    )
