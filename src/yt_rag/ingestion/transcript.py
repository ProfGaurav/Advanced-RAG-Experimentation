"""Fetch a video's transcript as timestamped snippets (via `youtube-transcript-api`).

Snippets are cached as JSON (default data/transcripts/<video_id>.json), so re-chunking,
re-indexing and evals never need to call YouTube again.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from yt_rag.exceptions import TranscriptUnavailableError

# Non-speech caption tags such as [Music], [Applause], [Laughter]
_TAG_RE = re.compile(r"\[[^\]]*\]")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Snippet:
    """One caption line: `text` spoken from `start` for `duration` seconds."""

    text: str
    start: float
    duration: float


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def fetch_snippets(video_id: str, languages: list[str] | None = None) -> list[Snippet]:
    """Download the transcript of `video_id` (the ID, not the URL) as cleaned snippets.

    Raises:
        TranscriptUnavailableError: if the video has no usable transcript.
    """
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages or ["en"])
    except TranscriptsDisabled as exc:
        raise TranscriptUnavailableError("No captions available for this video.") from exc
    except NoTranscriptFound as exc:
        raise TranscriptUnavailableError(
            f"No transcript in the configured languages ({', '.join(languages or ['en'])})."
        ) from exc
    except VideoUnavailable as exc:
        raise TranscriptUnavailableError("This video is unavailable (private, removed or region-locked).") from exc
    except CouldNotRetrieveTranscript as exc:  # IP blocks, age restriction, ...
        raise TranscriptUnavailableError(f"Could not retrieve the transcript: {type(exc).__name__}.") from exc

    snippets = [Snippet(_clean(s.text), float(s.start), float(s.duration)) for s in fetched]
    snippets = [s for s in snippets if s.text]
    if not snippets:
        raise TranscriptUnavailableError("The transcript is empty.")
    return snippets


def load_or_fetch_snippets(video_id: str, cache_dir: Path, languages: list[str] | None = None) -> list[Snippet]:
    """Return cached snippets for `video_id`, fetching (and caching) them on a miss."""
    path = cache_dir / f"{video_id}.json"
    if path.exists():
        return [Snippet(**s) for s in json.loads(path.read_text(encoding="utf-8"))]

    snippets = fetch_snippets(video_id, languages)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in snippets], ensure_ascii=False), encoding="utf-8")
    return snippets
