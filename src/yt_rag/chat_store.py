"""Persisted chat sessions (default data/chat_sessions.json).

Each session belongs to one video:
    {"<session_id>": {"video_id", "title", "created_at", "updated_at",
                      "messages": [{"role", "content"}, ...]}, ...}
"""

import json
import threading
import uuid
from datetime import datetime, timezone

from yt_rag.config import get_settings

_lock = threading.Lock()
_TITLE_LEN = 40


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load() -> dict[str, dict]:
    path = get_settings().chat_sessions_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(sessions: dict[str, dict]) -> None:
    path = get_settings().chat_sessions_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8")


def list_sessions() -> list[dict]:
    """All sessions (each with an `id` key), most recently updated first."""
    sessions = [{"id": sid, **s} for sid, s in _load().items()]
    return sorted(sessions, key=lambda s: s["updated_at"], reverse=True)


def get_session(session_id: str) -> dict | None:
    session = _load().get(session_id)
    return {"id": session_id, **session} if session else None


def create_session(video_id: str) -> str:
    """Create an empty session for `video_id` and return its ID."""
    session_id = uuid.uuid4().hex[:12]
    with _lock:
        sessions = _load()
        sessions[session_id] = {
            "video_id": video_id,
            "title": "New chat",
            "created_at": _now(),
            "updated_at": _now(),
            "messages": [],
        }
        _save(sessions)
    return session_id


def append_message(session_id: str, role: str, content: str) -> None:
    """Add a message; the first user message becomes the session title."""
    with _lock:
        sessions = _load()
        session = sessions[session_id]
        if role == "user" and not session["messages"]:
            session["title"] = content if len(content) <= _TITLE_LEN else content[:_TITLE_LEN].rstrip() + "…"
        session["messages"].append({"role": role, "content": content})
        session["updated_at"] = _now()
        _save(sessions)


def delete_session(session_id: str) -> None:
    with _lock:
        sessions = _load()
        sessions.pop(session_id, None)
        _save(sessions)
