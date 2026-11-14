"""Split a transcript into LangChain `Document` chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_transcript(
    transcript: str,
    video_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Chunk the transcript with `RecursiveCharacterTextSplitter`.

    Each Document carries metadata `{"video_id": video_id}`.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([transcript], metadatas=[{"video_id": video_id}])
    return chunks
