"""Chat area: message rendering and question input."""

import streamlit as st


def render_messages(messages: list[dict]) -> None:
    """Render previous user/assistant messages."""
    for message in messages:
        show_message(message["role"], message["content"])


def show_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)


def get_user_question(title: str = "the video") -> str | None:
    """Return the question from `st.chat_input`, if any."""
    return st.chat_input(f"Ask a question about {title}...")
