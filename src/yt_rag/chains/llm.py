"""Chat model factory."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> BaseChatModel:
    """Return a `ChatOpenAI` instance (reads OPENAI_API_KEY from the environment)."""
    llm = ChatOpenAI(model=model, temperature=temperature)
    return llm
