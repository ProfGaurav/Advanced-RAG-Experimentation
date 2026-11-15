"""Backend CLI: add a YouTube video to the shared FAISS vector store, or rebuild it.

Usage (from the rag/ folder):
    python scripts/build_index.py --video-id <VIDEO_ID> [--force]
    python scripts/build_index.py --reindex-all

If --video-id is omitted, VIDEO_ID from `.env` is used.
--force wipes the index and registry first, then indexes that one video.
--reindex-all rebuilds the index from every registered video (cached transcripts) with the
current chunk settings; use it after changing CHUNK_SIZE / EMBEDDING_MODEL or the index schema.
Writes the index to VECTORSTORE_DIR (default: data/vectorstore/).
"""

import argparse
import sys
from pathlib import Path

# Make `src/` importable even without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt_rag.config import get_settings  # noqa: E402
from yt_rag.exceptions import YTRagError  # noqa: E402
from yt_rag.indexing import build_index, reindex_all  # noqa: E402
from yt_rag.vectorstore.faiss_store import index_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse --video-id / --force, or --reindex-all."""
    parser = argparse.ArgumentParser(description="Build the FAISS index for YouTube videos.")
    parser.add_argument("--video-id", default=get_settings().video_id, help="YouTube video ID (not the URL)")
    parser.add_argument("--force", action="store_true", help="Wipe the whole index and registry, then index this video")
    parser.add_argument("--reindex-all", action="store_true", help="Rebuild the index from all registered videos")
    return parser.parse_args()


def main() -> None:
    """Dispatch to `indexing.reindex_all()` or `indexing.build_index()` and print a summary."""
    args = parse_args()

    if args.reindex_all:
        try:
            counts = reindex_all()
        except YTRagError as exc:
            sys.exit(f"Error: {exc}")
        if not counts:
            sys.exit("No registered videos to re-index.")
        for vid, n in counts.items():
            print(f"  {vid}: {n} chunks")
        print(f"Re-indexed {len(counts)} video(s), {sum(counts.values())} chunks saved to {index_path()}")
        return

    if not args.video_id:
        sys.exit("No video ID given. Pass --video-id or set VIDEO_ID in .env")
    try:
        n_chunks = build_index(args.video_id, force_rebuild=args.force)
    except YTRagError as exc:
        sys.exit(f"Error: {exc}")

    if n_chunks == 0:
        print(f"Video {args.video_id} is already indexed at {index_path()} (use --force to wipe and rebuild).")
    else:
        print(f"Indexed video {args.video_id}: {n_chunks} chunks saved to {index_path()}")


if __name__ == "__main__":
    main()
