# YouTube RAG

Ask questions about a YouTube video. The FAISS index is built **once** on the
backend from a video ID; the Streamlit app only answers questions against it.

## Flow

```
BACKEND (once)
  scripts/build_index.py --video-id X
    -> transcript -> chunks -> OpenAI embeddings -> FAISS (data/vectorstore/)

FRONTEND (Streamlit)
  question -> retriever (top-k) -> prompt -> ChatOpenAI -> answer
```

## Layout

| Path | Role |
|---|---|
| `scripts/build_index.py` | Backend CLI: builds the vector store once |
| `app.py` | Streamlit entry point (Q&A only) |
| `ui/chat.py` | Chat history + question input |
| `src/yt_rag/config.py` | Loads `.env`, exposes `get_settings()` |
| `src/yt_rag/exceptions.py` | Errors shown to the user |
| `src/yt_rag/indexing.py` | `build_index()` - backend ingestion pipeline |
| `src/yt_rag/qa.py` | `load_qa_chain()` / `answer()` - used by the UI |
| `src/yt_rag/ingestion/` | Transcript fetch, chunking |
| `src/yt_rag/vectorstore/` | Embeddings + FAISS build/save/load |
| `src/yt_rag/chains/` | Retriever, prompt, LLM, LCEL QA chain |
| `data/vectorstore/` | Persisted FAISS index |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env               # add OPENAI_API_KEY and VIDEO_ID
python scripts/build_index.py      # once (or --video-id <ID>, --force to rebuild)
streamlit run app.py
```
