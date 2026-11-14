"""Domain-specific exceptions surfaced to the user as friendly messages."""


class YTRagError(Exception):
    """Base exception for the project."""


class TranscriptUnavailableError(YTRagError):
    """Raised when the video has no transcript (disabled or not found)."""


class VectorStoreNotFoundError(YTRagError):
    """Raised when no persisted FAISS index exists."""


class InvalidVideoURLError(YTRagError):
    """Raised when a YouTube video ID cannot be extracted from the user's input."""
