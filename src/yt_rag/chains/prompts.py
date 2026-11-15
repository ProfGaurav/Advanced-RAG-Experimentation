"""Chat prompt templates for the multi-turn QA chain."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_CONDENSE_SYSTEM = """\
Given the conversation so far and a follow-up question, rewrite the follow-up as a \
standalone question that can be understood without the conversation. Resolve pronouns \
and references ("he", "that", "the second point") using the conversation. \
Do NOT answer the question. If it is already standalone, return it unchanged. \
Return only the question."""

_QA_SYSTEM = """\
You are a helpful assistant answering questions about a YouTube video.
Answer ONLY from the transcript excerpts in <context>. Use the conversation history only \
to understand what the user is referring to; every fact in your answer must come from the context.
If the context is insufficient, just say you don't know.
Each excerpt starts with a time label such as [12:34–13:32]. After each claim, cite the start \
time of the excerpt(s) it comes from, e.g. (12:34). Only cite times that appear in the labels.

<context>
{context}
</context>"""


def get_condense_prompt() -> ChatPromptTemplate:
    """Rewrites a follow-up into a standalone question for retrieval.

    Input variables: `chat_history` (list of messages), `question`.
    """
    return ChatPromptTemplate.from_messages([
        ("system", _CONDENSE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])


def get_qa_prompt() -> ChatPromptTemplate:
    """Answers from the retrieved context, aware of the conversation.

    Input variables: `context`, `chat_history` (list of messages), `question`.
    """
    return ChatPromptTemplate.from_messages([
        ("system", _QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
