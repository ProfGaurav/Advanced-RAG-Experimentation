"""Right panel: add-a-video form and the list of indexed videos."""

import streamlit as st

from ui.state import SELECTED_VIDEO, new_chat


def render_add_video_form() -> str | None:
    """Render the "add video" form; return the submitted link (or None)."""
    with st.form("add_video", clear_on_submit=True):
        st.subheader("➕ Add a video")
        url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
        submitted = st.form_submit_button("Add video", width="stretch")
    return url.strip() if submitted and url.strip() else None


def render_video_selector(videos: list[dict]) -> str | None:
    """Render the indexed videos; return the selected video ID.

    Picking a different video starts a new chat for it.
    """
    st.subheader("🎞️ Videos")
    if not videos:
        st.caption("No videos yet - add one above.")
        return None

    titles = {v["video_id"]: v["title"] for v in videos}
    ids = list(titles)
    if st.session_state.get(SELECTED_VIDEO) not in titles:
        st.session_state[SELECTED_VIDEO] = ids[0]

    return st.radio(
        "Chat with",
        ids,
        key=SELECTED_VIDEO,
        format_func=lambda vid: titles[vid],
        on_change=new_chat,
        label_visibility="collapsed",
    )
