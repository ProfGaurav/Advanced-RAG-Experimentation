"""JSON registry of indexed videos (drives the sidebar).

File (default data/videos.json):
    {"<video_id>": {"title": ..., "url": ..., "n_chunks": ..., "added_at": ...}, ...}
"""

import json
import threading
from datetime import datetime, timezone

from langchain_community.vectorstores import FAISS

from yt_rag.config import get_settings
from yt_rag.ingestion.metadata import fetch_video_title, video_url

_lock = threading.Lock()


def load_registry() -> dict[str, dict]:
    """Return the registry dict (empty if the file doesn't exist)."""
    path = get_settings().registry_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(registry: dict[str, dict]) -> None:
    path = get_settings().registry_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def list_videos() -> list[dict]:
    """Return registered videos (each with a `video_id` key), oldest first."""
    registry = load_registry()
    videos = [{"video_id": vid, **info} for vid, info in registry.items()]
    return sorted(videos, key=lambda v: v.get("added_at", ""))


def is_indexed(video_id: str) -> bool:
    return video_id in load_registry()


def add_video(video_id: str, title: str, n_chunks: int) -> None:
    """Register (or overwrite) a video entry."""
    with _lock:
        registry = load_registry()
        registry[video_id] = {
            "title": title,
            "url": video_url(video_id),
            "n_chunks": n_chunks,
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _save(registry)


def clear_registry() -> None:
    """Delete the registry file (used by a forced rebuild)."""
    with _lock:
        get_settings().registry_path.unlink(missing_ok=True)


def bootstrap_from_store(store: FAISS | None) -> None:
    """Register videos already in `store` that the registry doesn't know about.

    Migrates indexes built before the registry existed.
    """
    if store is None:
        return
    counts: dict[str, int] = {}
    for doc in store.docstore._dict.values():  # InMemoryDocstore
        vid = doc.metadata.get("video_id")
        if vid:
            counts[vid] = counts.get(vid, 0) + 1
    known = load_registry()
    for vid, n in counts.items():
        if vid not in known:
            add_video(vid, fetch_video_title(vid), n)
