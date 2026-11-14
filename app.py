"""Streamlit entry point - question answering only.

Run with (from the rag/ folder):  streamlit run app.py
Requires the index to be built first:  python scripts/build_index.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make `src/` importable even without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yt_rag import qa  # noqa: E402
from yt_rag.exceptions import VectorStoreNotFoundError  # noqa: E402
from ui.chat import add_message, get_user_question, init_chat_state, render_history  # noqa: E402


@st.cache_resource(show_spinner="Loading video index...")
def get_chain():
    """Build the QA chain once per server process and reuse it across reruns."""
    return qa.load_qa_chain()


def main() -> None:
    """Compose the Streamlit page using components from `ui/`."""
    st.set_page_config(page_title="YouTube RAG", page_icon="🎬")
    st.title("🎬 Ask the Video")
    st.caption("Answers come only from the video's transcript.")

    try:
        chain = get_chain()
    except VectorStoreNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    init_chat_state()
    render_history()

    question = get_user_question()
    if question:
        add_message("user", question)
        with st.spinner("Thinking..."):
            try:
                reply = qa.answer(chain, question)
            except Exception as exc:  # surface API/network errors in the UI
                reply = f"Sorry, something went wrong: {exc}"
        add_message("assistant", reply)


if __name__ == "__main__":
    main()
