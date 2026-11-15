"""Fetch video metadata (title) via YouTube's public oEmbed endpoint (no API key)."""

import json
from urllib.parse import quote
from urllib.request import urlopen


def video_url(video_id: str) -> str:
    """Canonical watch URL for `video_id`."""
    return f"https://www.youtube.com/watch?v={video_id}"


def format_timestamp(seconds: float) -> str:
    """`754.2` -> `"12:34"`, `3723` -> `"1:02:03"`."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def timestamp_url(video_id: str, seconds: float) -> str:
    """Link that opens `video_id` at `seconds`."""
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def fetch_video_title(video_id: str, timeout: float = 5.0) -> str:
    """Return the video's title, or the video ID if it can't be fetched."""
    endpoint = f"https://www.youtube.com/oembed?format=json&url={quote(video_url(video_id), safe='')}"
    try:
        with urlopen(endpoint, timeout=timeout) as resp:
            return json.load(resp).get("title") or video_id
    except Exception:  # network error, private/removed video, bad JSON...
        return video_id
