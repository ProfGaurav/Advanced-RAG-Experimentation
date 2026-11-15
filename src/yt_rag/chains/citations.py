"""Check the (mm:ss) citations in an answer against the excerpts that were actually retrieved.

A citation is valid only if it equals the start-time label of one of the retrieved chunks;
anything else is a hallucinated timestamp. Useful as a runtime guardrail and as an eval metric.
"""

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from yt_rag.ingestion.metadata import format_timestamp, timestamp_url

_TIME = r"\d{1,2}:\d{2}(?::\d{2})?"
# "(12:34)" or "(12:34, 15:02)"
_CITATION_GROUP = re.compile(rf"\(\s*({_TIME}(?:\s*,\s*{_TIME})*)\s*\)")
# same, plus the whitespace before it — used to remove inline citations from the text
_INLINE_CITATION = re.compile(rf"[ \t]*{_CITATION_GROUP.pattern}")

CITATIONS_HEADING = "**Citations**"


@dataclass
class CitationCheck:
    valid: list[str] = field(default_factory=list)    # cited and present in a chunk label
    invalid: list[str] = field(default_factory=list)  # cited but not a retrieved start time

    @property
    def ok(self) -> bool:
        return not self.invalid


def extract_citations(answer: str) -> list[str]:
    """All cited times in order of appearance (duplicates kept)."""
    times: list[str] = []
    for group in _CITATION_GROUP.findall(answer):
        times.extend(t.strip() for t in group.split(","))
    return times


def check_citations(answer: str, docs: list[Document]) -> CitationCheck:
    """Split the answer's citations into valid / invalid w.r.t. the retrieved `docs`."""
    allowed = {format_timestamp(d.metadata["start"]) for d in docs if "start" in d.metadata}
    result = CitationCheck()
    for t in dict.fromkeys(extract_citations(answer)):  # de-duplicate, keep order
        (result.valid if t in allowed else result.invalid).append(t)
    return result


def strip_inline_citations(answer: str) -> str:
    """Remove "(12:34)" / "(12:34, 15:02)" markers from the answer text."""
    return _INLINE_CITATION.sub("", answer).strip()


def strip_citation_footer(message: str) -> str:
    """Drop the citations footer from a stored assistant message (before reusing it as history)."""
    return message.split(f"\n\n{CITATIONS_HEADING}", 1)[0].rstrip()


def with_citation_footer(answer: str, check: CitationCheck, docs: list[Document]) -> str:
    """Answer text without inline markers, then a numbered list of the valid cited times:

        <answer>

        **Citations**
        1. [0:05](https://youtu.be/<id>?t=5)
        2. [12:34](https://youtu.be/<id>?t=754)

    Times are listed chronologically and link to that moment in the video. Invalid
    (hallucinated) times are dropped. No footer when nothing valid was cited.
    """
    text = strip_inline_citations(answer)
    if not check.valid:
        return text

    starts: dict[str, Document] = {}
    for d in docs:
        if "start" in d.metadata:
            starts.setdefault(format_timestamp(d.metadata["start"]), d)

    def line(label: str) -> str:
        doc = starts[label]
        vid = doc.metadata.get("video_id")
        return f"[{label}]({timestamp_url(vid, doc.metadata['start'])})" if vid else label

    ordered = sorted(check.valid, key=lambda t: starts[t].metadata["start"])
    items = "\n".join(f"{i}. {line(t)}" for i, t in enumerate(ordered, 1))
    return f"{text}\n\n{CITATIONS_HEADING}\n\n{items}"
