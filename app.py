"""Streamlit entry point.

Layout:
    left sidebar  - previous chat sessions (persisted in data/chat_sessions.json)
    center        - chat with the selected video
    right panel   - add a YouTube link (appended to the shared FAISS store) + video list

Answers are retrieved only from the selected video's chunks (metadata filter on `video_id`).

Run with (from the rag/ folder):  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make `src/` importable even without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yt_rag import chat_store, indexing, qa, registry  # noqa: E402
from yt_rag.chains.guardrails import BLOCKING  # noqa: E402
from yt_rag.exceptions import YTRagError  # noqa: E402
from ui.chat import get_user_question, render_messages, show_message  # noqa: E402
from ui.sessions_panel import render_sessions_sidebar  # noqa: E402
from ui.state import ACTIVE_SESSION, init_state, select_video  # noqa: E402
from ui.video_panel import render_add_video_form, render_video_selector  # noqa: E402

CHAT_HEIGHT = 560


@st.cache_resource(show_spinner="Loading video index and reranker...")
def get_resources() -> qa.Resources:
    """Load store + LLM + reranker once per server process.

    The returned object is shared (and mutated) by all sessions, so a store created or
    extended by an ingestion is visible everywhere without reloading from disk.
    """
    return qa.load_resources()


def handle_add_video(resources: qa.Resources, url: str) -> None:
    """Ingest `url` into the shared store and select it."""
    with st.status("Adding video...", expanded=True) as status:
        try:
            st.write("Fetching transcript, embedding and saving...")
            result = indexing.ingest_video(resources.store, url)
        except YTRagError as exc:
            status.update(label="Could not add video", state="error")
            st.error(str(exc))
            return
        except Exception as exc:  # API/network errors
            status.update(label="Could not add video", state="error")
            st.error(f"Something went wrong: {exc}")
            return

        resources.store = result.store
        select_video(result.video_id)
        if result.already_indexed:
            status.update(label=f"Already indexed: {result.title}", state="complete", expanded=False)
        else:
            status.update(
                label=f"Added: {result.title} ({result.n_chunks} chunks)", state="complete", expanded=False
            )


def render_chat(resources: qa.Resources, video: dict) -> None:
    """Chat for the active session (or a new one) about `video`."""
    video_id = video["video_id"]
    session = chat_store.get_session(st.session_state[ACTIVE_SESSION] or "")
    if session and session["video_id"] != video_id:  # stale: its video was deselected
        session = None
        st.session_state[ACTIVE_SESSION] = None

    st.subheader(session["title"] if session else "New chat")
    st.caption(f"Answers come only from the transcript of **{video['title']}**.")
    with st.expander("Watch video"):
        st.video(video["url"])

    messages_box = st.container(height=CHAT_HEIGHT, border=True)
    with messages_box:
        if session:
            render_messages(session["messages"])
        else:
            st.caption("Ask anything about this video to start a new chat.")

    question = get_user_question(video["title"])
    if not question:
        return

    history = session["messages"] if session else []
    with messages_box:
        show_message("user", question)
        chain = qa.build_chain_for(resources, video_id)
        with st.spinner("Thinking..."):
            try:
                reply = qa.ask(chain, question, history)
            except Exception as exc:  # surface API/network errors, but keep them out of the history
                st.error(f"Sorry, something went wrong: {exc}")
                return
        if reply.guardrail in BLOCKING:  # show the refusal, but don't save it or send it as history
            st.warning(reply.text)
            return

    if session is None:  # first answered question creates the session
        st.session_state[ACTIVE_SESSION] = chat_store.create_session(video_id)
    session_id = st.session_state[ACTIVE_SESSION]
    chat_store.append_message(session_id, "user", question)
    chat_store.append_message(session_id, "assistant", reply.text)
    st.rerun()  # refresh the title and the sessions sidebar


def main() -> None:
    """Compose the page: sessions sidebar, chat (center), videos (right)."""
    st.set_page_config(page_title="YouTube RAG", page_icon="🎬", layout="wide")
    init_state()
    try:
        resources = get_resources()
    except YTRagError as exc:  # e.g. index built with an old schema / other embedding model
        st.error(str(exc))
        st.stop()

    st.title("🎬 Ask the Video")
    chat_col, video_col = st.columns([3, 1], gap="large")

    with video_col:
        url = render_add_video_form()
        if url:
            handle_add_video(resources, url)
        videos = registry.list_videos()
        video_id = render_video_selector(videos)

    with chat_col:
        if video_id is None or resources.store is None:
            st.info("Add a YouTube video from the panel on the right to get started.")
        else:
            video = next(v for v in videos if v["video_id"] == video_id)
            render_chat(resources, video)

    # Drawn last so it reflects sessions created/updated during this run.
    render_sessions_sidebar(chat_store.list_sessions(), {v["video_id"]: v["title"] for v in videos})


if __name__ == "__main__":
    main()
