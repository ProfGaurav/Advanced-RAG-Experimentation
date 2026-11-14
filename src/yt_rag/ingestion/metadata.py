"""Fetch video metadata (title) via YouTube's public oEmbed endpoint (no API key)."""

import json
from urllib.parse import quote
from urllib.request import urlopen


def video_url(video_id: str) -> str:
    """Canonical watch URL for `video_id`."""
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_video_title(video_id: str, timeout: float = 5.0) -> str:
    """Return the video's title, or the video ID if it can't be fetched."""
    endpoint = f"https://www.youtube.com/oembed?format=json&url={quote(video_url(video_id), safe='')}"
    try:
        with urlopen(endpoint, timeout=timeout) as resp:
            return json.load(resp).get("title") or video_id
    except Exception:  # network error, private/removed video, bad JSON...
        return video_id
