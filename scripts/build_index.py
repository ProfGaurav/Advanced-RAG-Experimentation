"""Backend CLI: add a YouTube video to the shared FAISS vector store.

Usage (from the rag/ folder):
    python scripts/build_index.py --video-id <VIDEO_ID> [--force]

If --video-id is omitted, VIDEO_ID from `.env` is used.
Writes the index to VECTORSTORE_DIR (default: data/vectorstore/).
Optional: videos can also be added from the Streamlit sidebar. --force wipes the index first.
"""

import argparse
import sys
from pathlib import Path

# Make `src/` importable even without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt_rag.config import get_settings  # noqa: E402
from yt_rag.exceptions import YTRagError  # noqa: E402
from yt_rag.indexing import build_index  # noqa: E402
from yt_rag.vectorstore.faiss_store import index_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse --video-id and --force (rebuild even if an index already exists)."""
    parser = argparse.ArgumentParser(description="Build the FAISS index for a YouTube video.")
    parser.add_argument("--video-id", default=get_settings().video_id, help="YouTube video ID (not the URL)")
    parser.add_argument("--force", action="store_true", help="Wipe the whole index and registry, then index this video")
    return parser.parse_args()


def main() -> None:
    """Resolve the video ID, call `indexing.build_index()`, print a summary."""
    args = parse_args()
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
