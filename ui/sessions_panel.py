"""Left sidebar: previous chat sessions."""

import streamlit as st

from ui.state import ACTIVE_SESSION, delete_session, new_chat, open_session

_VIDEO_TITLE_LEN = 38


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rstrip() + "…"


def render_sessions_sidebar(sessions: list[dict], video_titles: dict[str, str]) -> None:
    """List sessions (newest first) with open/delete buttons and a New chat button."""
    with st.sidebar:
        st.title("💬 Chats")
        st.button("➕ New chat", on_click=new_chat, width="stretch", type="primary")

        if not sessions:
            st.caption("No previous chats yet.")
            return

        active = st.session_state.get(ACTIVE_SESSION)
        for s in sessions:
            video_title = video_titles.get(s["video_id"], s["video_id"])
            open_col, del_col = st.columns([6, 1], vertical_alignment="center")
            open_col.button(
                s["title"],
                key=f"open_{s['id']}",
                on_click=open_session,
                args=(s["id"], s["video_id"]),
                type="secondary" if s["id"] == active else "tertiary",
                help=video_title,
                width="stretch",
            )
            del_col.button("🗑", key=f"del_{s['id']}", on_click=delete_session, args=(s["id"],), help="Delete chat")
            open_col.caption(f"🎞️ {_short(video_title, _VIDEO_TITLE_LEN)}")
