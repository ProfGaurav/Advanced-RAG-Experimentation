# YouTube RAG

Ask questions about YouTube videos. Paste a link in the sidebar: the transcript is
embedded and **appended** to one shared FAISS index, with `video_id` metadata on
every chunk. Pick a video in the sidebar and the chat answers only from that
video's chunks (FAISS metadata filter).

## Flow

```
ADD VIDEO (sidebar form, or scripts/build_index.py)
  link -> video ID -> timestamped snippets (cached: data/transcripts/<id>.json)
       -> chunks {video_id, start, end} -> OpenAI embeddings (text only, times stay in metadata)
       -> FAISS.add_documents -> data/vectorstore/ (+ manifest.json)  +  data/videos.json

CHAT (selected video, multi-turn)
  question + chat history (last HISTORY_TURNS exchanges of the session)
    -> ChatOpenAI rewrites follow-up into a standalone question (skipped on the first turn)
    -> FAISS (RERANK_CANDIDATES, filter video_id) on the standalone question
    -> cross-encoder reranker keeps the best RETRIEVER_K (RERANKER=none to skip)
    -> chat prompt (system + time-labelled context, history, question) -> ChatOpenAI
    -> answer citing timestamps, e.g. (40:36)
```

## Re-indexing

`data/vectorstore/manifest.json` records the index schema version and embedding model. If
either no longer matches (e.g. you changed `EMBEDDING_MODEL`, or pulled a schema change),
the app refuses to load the index and asks you to run:

```bash
python scripts/build_index.py --reindex-all   # rebuilds from cached transcripts, keeps titles
```

Also run it after changing `CHUNK_SIZE` / `CHUNK_OVERLAP`.

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
| `src/yt_rag/chains/` | Retriever (video filter), cross-encoder reranker, condense + QA chat prompts, LLM, history-aware LCEL chain |
| `data/vectorstore/` | Persisted FAISS index (all videos) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env               # add OPENAI_API_KEY
streamlit run app.py               # add videos from the sidebar
```

`FETCH_K` (default 100) is how many nearest chunks FAISS fetches before filtering
to the selected video; raise it if you index many long videos.
