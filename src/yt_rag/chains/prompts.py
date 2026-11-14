"""Prompt templates for the QA chain."""

from langchain_core.prompts import PromptTemplate


def get_qa_prompt() -> PromptTemplate:
    """Prompt that instructs the LLM to answer ONLY from the transcript context
    and to say "I don't know" when the answer isn't in the context.

    Input variables: `context`, `question`.
    """
    prompt = PromptTemplate(
        template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
        input_variables=["context", "question"],
    )
    return prompt
