# YouTube RAG

Ask questions about YouTube videos. Paste a link in the sidebar: the transcript is
embedded and **appended** to one shared FAISS index, with `video_id` metadata on
every chunk. Pick a video in the sidebar and the chat answers only from that
video's chunks (FAISS metadata filter).

## Flow

```
ADD VIDEO (sidebar form, or scripts/build_index.py)
  link -> video ID -> transcript -> chunks {video_id} -> OpenAI embeddings
       -> FAISS.add_documents -> data/vectorstore/   +   data/videos.json (title, chunks)

CHAT (selected video)
  question -> retriever (top-k, filter video_id) -> prompt -> ChatOpenAI -> answer
```

## Layout

| Path | Role |
|---|---|
| `app.py` | Streamlit entry point: sessions (left), chat (center), videos (right) |
| `ui/sessions_panel.py` | Left sidebar: previous chat sessions, New chat, delete |
| `ui/video_panel.py` | Right panel: add-video form + list of indexed videos |
| `ui/chat.py` | Message rendering + question input |
| `ui/state.py` | Session-state keys + callbacks shared by the panels |
| `src/yt_rag/chat_store.py` | Persisted chat sessions (`data/chat_sessions.json`) |
| `scripts/build_index.py` | CLI: add a video (`--force` wipes the index first) |
| `src/yt_rag/config.py` | Loads `.env`, exposes `get_settings()` |
| `src/yt_rag/exceptions.py` | Errors shown to the user |
| `src/yt_rag/indexing.py` | `ingest_video()` - link -> chunks -> store + registry |
| `src/yt_rag/registry.py` | `data/videos.json` registry of indexed videos |
| `src/yt_rag/qa.py` | `load_resources()` / `build_chain_for()` / `answer()` |
| `src/yt_rag/ingestion/` | Link parsing, title lookup (oEmbed), transcript, chunking |
| `src/yt_rag/vectorstore/` | Embeddings + FAISS build/append/save/load |
| `src/yt_rag/chains/` | Retriever (video filter), prompt, LLM, LCEL QA chain |
| `data/vectorstore/` | Persisted FAISS index (all videos) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env               # add OPENAI_API_KEY
streamlit run app.py               # add videos from the sidebar
```

`FETCH_K` (default 100) is how many nearest chunks FAISS fetches before filtering
to the selected video; raise it if you index many long videos.
