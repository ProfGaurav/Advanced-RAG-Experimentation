"""Compose retriever + prompt + LLM into a LangChain (LCEL) runnable chain."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever


def format_docs(retrieved_docs: list[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


def build_qa_chain(
    retriever: VectorStoreRetriever,
    prompt: PromptTemplate,
    llm: BaseChatModel,
) -> Runnable:
    """Return the QA chain.

    Usage:
        main_chain = build_qa_chain(retriever, prompt, llm)
        answer = main_chain.invoke("your question")
    """
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain
