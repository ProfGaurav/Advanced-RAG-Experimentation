"""Session-state keys and callbacks shared by the UI panels.

Callbacks run before the script reruns, so they can safely set widget keys
(e.g. the video radio) before those widgets are drawn.
"""

import streamlit as st

from yt_rag import chat_store

SELECTED_VIDEO = "selected_video_id"   # key of the video radio (right panel)
ACTIVE_SESSION = "active_session_id"   # open chat session, None = new chat


def init_state() -> None:
    st.session_state.setdefault(ACTIVE_SESSION, None)


def new_chat() -> None:
    """Start a fresh chat (the session is created on the first question)."""
    st.session_state[ACTIVE_SESSION] = None


def open_session(session_id: str, video_id: str) -> None:
    """Open a previous session and select its video."""
    st.session_state[ACTIVE_SESSION] = session_id
    st.session_state[SELECTED_VIDEO] = video_id


def delete_session(session_id: str) -> None:
    chat_store.delete_session(session_id)
    if st.session_state.get(ACTIVE_SESSION) == session_id:
        st.session_state[ACTIVE_SESSION] = None


def select_video(video_id: str) -> None:
    """Select a video and start a new chat for it."""
    st.session_state[SELECTED_VIDEO] = video_id
    st.session_state[ACTIVE_SESSION] = None
