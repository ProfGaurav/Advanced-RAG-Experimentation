"""Chat area: history rendering and question input."""

import streamlit as st


def init_chat_state() -> None:
    """Initialise `st.session_state.messages`."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_history() -> None:
    """Render previous user/assistant messages with `st.chat_message`."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def add_message(role: str, content: str) -> None:
    """Append a message to the history and render it."""
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)


def get_user_question() -> str | None:
    """Return the question from `st.chat_input`, if any."""
    return st.chat_input("Ask a question about the video...")
