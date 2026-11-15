"""Split timestamped transcript snippets into LangChain `Document` chunks.

The snippets are joined into one text with a char-offset -> snippet map. Any text splitter
can then run on the plain text; each chunk's `start_index` is mapped back to the snippets
it spans to get its time range. Timestamps live in metadata, never in the embedded text.
"""

from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from yt_rag.ingestion.transcript import Snippet


def _join(snippets: list[Snippet]) -> tuple[str, list[int]]:
    """Join snippet texts with spaces; return the text and each snippet's start offset."""
    offsets, pos = [], 0
    for s in snippets:
        offsets.append(pos)
        pos += len(s.text) + 1  # +1 for the joining space
    return " ".join(s.text for s in snippets), offsets


def _snippet_ends(snippets: list[Snippet]) -> list[float]:
    """End time of each snippet, capped at the next snippet's start (auto-captions overlap)."""
    ends = []
    for i, s in enumerate(snippets):
        end = s.start + s.duration
        if i + 1 < len(snippets):
            end = min(end, snippets[i + 1].start)
        ends.append(max(end, s.start))
    return ends


def split_transcript(
    snippets: list[Snippet],
    video_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Chunk the transcript with `RecursiveCharacterTextSplitter`.

    Each Document carries metadata:
        video_id, start, end (seconds), chunk_index, chunker
    """
    text, offsets = _join(snippets)
    ends = _snippet_ends(snippets)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    docs = splitter.create_documents([text], metadatas=[{"video_id": video_id}])

    search_from = 0
    for i, doc in enumerate(docs):
        start_char = doc.metadata.pop("start_index", -1)
        if start_char < 0:  # defensive: locate the chunk ourselves
            start_char = max(text.find(doc.page_content, search_from), 0)
        search_from = start_char
        end_char = start_char + len(doc.page_content) - 1

        first = bisect_right(offsets, start_char) - 1
        last = bisect_right(offsets, end_char) - 1
        doc.metadata.update(
            start=round(snippets[first].start, 2),
            end=round(ends[last], 2),
            chunk_index=i,
            chunker=f"recursive-{chunk_size}-{chunk_overlap}",
        )
    return docs
