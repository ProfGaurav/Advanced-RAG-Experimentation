"""Extract a YouTube video ID from a link (or a bare ID)."""

import re
from urllib.parse import parse_qs, urlparse

from yt_rag.exceptions import InvalidVideoURLError

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_PREFIXES = ("shorts", "embed", "live", "v")


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-character video ID from a YouTube URL or a bare ID.

    Supports youtube.com/watch?v=..., youtu.be/..., /shorts/, /embed/, /live/, /v/
    (with or without scheme, www./m./music. subdomains and extra query params).

    Raises:
        InvalidVideoURLError: if no valid video ID can be found.
    """
    text = (url_or_id or "").strip()
    if _ID_RE.match(text):
        return text

    if "://" not in text:
        text = "https://" + text
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]

    candidate = None
    if host == "youtu.be" and parts:
        candidate = parts[0]
    elif host == "youtube.com" or host.endswith(".youtube.com") or host == "youtube-nocookie.com" \
            or host.endswith(".youtube-nocookie.com"):
        if parts[:1] == ["watch"]:
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
            candidate = parts[1]

    if candidate and _ID_RE.match(candidate):
        return candidate
    raise InvalidVideoURLError(f"Could not find a YouTube video ID in: {url_or_id!r}")
