"""Fetch a video's transcript (via `youtube-transcript-api`)."""

from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

from yt_rag.exceptions import TranscriptUnavailableError


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> str:
    """Return the full transcript of `video_id` as a single string.

    `video_id` must be only the ID, not the full URL.

    Raises:
        TranscriptUnavailableError: if transcripts are disabled or not found.
    """
    try:
        # If you don't care which language, this returns the "best" one
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages or ["en"])

        # Flatten it to plain text
        transcript = " ".join(snippet.text for snippet in transcript_list)
        return transcript

    except TranscriptsDisabled as exc:
        raise TranscriptUnavailableError("No captions available for this video.") from exc


if __name__ == "__main__":
    print(fetch_transcript("g9zGQdrAlMw"))  # Example usage
