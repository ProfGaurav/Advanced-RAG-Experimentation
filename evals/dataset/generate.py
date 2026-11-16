"""Generate golden-set CANDIDATES with Ragas TestsetGenerator over the cached transcripts.

Run from rag/ with venv-eval active:
    python -m evals.dataset.generate                 # all indexed videos, 45 candidates
    python -m evals.dataset.generate --size 60 --video Gfr50f6ZBvo

Output (evals/dataset/):
    candidates.csv      one row per generated question, for hand review -> copy to reviewed.csv
    candidates.jsonl    raw Ragas output (full reference contexts)
    kg_<videos>.json    Ragas knowledge graph cache: re-runs skip the (paid) transform step

How it works:
  1. Transcripts (data/transcripts/<video_id>.json, timestamped snippets) are cut into fixed
     time windows (default 120 s). Each window is one Ragas chunk, so every generated question's
     reference context maps back to an exact [start_s, end_s] span. No chunk IDs are stored:
     spans stay valid whatever chunking strategy the app uses.
  2. Ragas builds a knowledge graph (summaries, themes, entities, similarity links; gpt-4o-mini)
     and synthesises single-hop and multi-hop questions with reference answers (gpt-4o).
  3. gpt-4o-mini suggests a `question_type` for each candidate (you confirm it during review).

Unanswerable and follow-up questions are hand-written in evals/dataset/manual.py.
Generation traces go to the LangSmith project `yt-rag-testgen`, not the app's project.
"""

import yt_rag.config  # noqa: F401  (side effect: loads rag/.env)

import argparse
import json
import os
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms import apply_transforms, default_transforms_for_prechunked

from yt_rag import registry
from yt_rag.config import get_settings

os.environ["LANGSMITH_PROJECT"] = "yt-rag-testgen"  # keep generation traces out of the app project

OUT_DIR = Path(__file__).resolve().parent

# Guides Ragas' question/answer writing. Lesson from the smoke run: reference answers padded with
# facts the question didn't ask for drag answer_correctness down for correct answers.
LLM_CONTEXT = (
    "The source is an auto-generated YouTube transcript of a long podcast interview: lowercase, "
    "no punctuation, with filler words and occasional mis-transcribed names. Write questions a "
    "viewer of the video would naturally ask. Never mention 'the transcript', 'the context', "
    "'the passage' or 'the text'. Reference answers must be concise and answer exactly what is "
    "asked, without extra facts; spell names correctly."
)

QuestionType = Literal["factual", "keyword", "paraphrase", "broad", "multi_hop", "why"]


class TypeSuggestion(BaseModel):
    question_type: QuestionType
    reason: str = Field(description="One short sentence")


CLASSIFY_PROMPT = """Classify this question about a podcast video into ONE type.

- factual: asks for one specific fact stated in one place, worded similarly to the speech
- keyword: hinges on an exact name, number, date or jargon term (e.g. a person, a year, a system name)
- paraphrase: answerable from one place, but worded clearly differently from how the speaker said it
- broad: asks for main points / overall views spread across the conversation
- multi_hop: needs facts from two or more separate places (comparisons, sequences, combinations)
- why: asks for a reason, explanation or motivation that needs background reasoning

Question: {question}
Reference answer: {answer}
Number of source passages: {n_contexts}
Source passage(s), as spoken (truncated):
{contexts}"""


# --------------------------------------------------------------------------------------------
# 1. Transcripts -> time-window chunks with spans
# --------------------------------------------------------------------------------------------

def load_snippets(video_id: str) -> list[dict]:
    path = get_settings().transcripts_dir / f"{video_id}.json"
    if not path.exists():
        raise SystemExit(f"No cached transcript at {path}. Index the video in the app first.")
    return json.loads(path.read_text(encoding="utf-8"))


def time_windows(snippets: list[dict], video_id: str, window_s: float) -> list[Document]:
    """Group consecutive snippets into ~window_s windows. metadata: video_id, start, end."""
    def to_doc(group: list[dict]) -> Document:
        text = " ".join(s["text"].replace("\n", " ") for s in group)
        start = group[0]["start"]
        end = group[-1]["start"] + group[-1].get("duration", 0.0)
        return Document(
            page_content=re.sub(r"\s+", " ", text).strip(),
            metadata={"video_id": video_id, "start": round(start, 1), "end": round(end, 1)},
        )

    docs, group = [], []
    for snip in snippets:
        if group and snip["start"] - group[0]["start"] >= window_s:
            docs.append(to_doc(group))
            group = []
        group.append(snip)
    if group:
        docs.append(to_doc(group))
    return docs


# --------------------------------------------------------------------------------------------
# 2. Knowledge graph (cached) + generation
# --------------------------------------------------------------------------------------------

def build_or_load_kg(chunks: list[Document], path: Path, transforms_llm, emb) -> KnowledgeGraph:
    if path.exists():
        print(f"Loading cached knowledge graph {path.name}")
        return KnowledgeGraph.load(path)
    print(f"Building knowledge graph over {len(chunks)} chunks (one-off cost)...")
    nodes = [
        Node(type=NodeType.CHUNK, properties={"page_content": d.page_content, "document_metadata": d.metadata})
        for d in chunks
    ]
    kg = KnowledgeGraph(nodes=nodes)
    apply_transforms(kg, default_transforms_for_prechunked(llm=transforms_llm, embedding_model=emb), RunConfig(max_workers=8))
    kg.save(path)
    print(f"Saved {path.name}: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
    return kg


def query_distribution(gen_llm, kg: KnowledgeGraph) -> list[tuple]:
    """Half single-hop, half multi-hop; drop synthesizers the graph can't support."""
    wanted = [
        (SingleHopSpecificQuerySynthesizer(llm=gen_llm, llm_context=LLM_CONTEXT), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=gen_llm, llm_context=LLM_CONTEXT), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=gen_llm, llm_context=LLM_CONTEXT), 0.25),
    ]
    usable = []
    for synth, weight in wanted:
        try:
            if synth.get_node_clusters(kg):
                usable.append((synth, weight))
            else:
                print(f"  skipping {synth.name}: no suitable node clusters")
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {synth.name}: {exc}")
    if not usable:
        raise SystemExit("No query synthesizer can run on this knowledge graph.")
    total = sum(w for _, w in usable)
    return [(s, w / total) for s, w in usable]


# --------------------------------------------------------------------------------------------
# 3. Map reference contexts back to spans
# --------------------------------------------------------------------------------------------

_HOP_PREFIX = re.compile(r"^<\d+-hop>\s*")


def spans_for(contexts: list[str], chunks: list[Document]) -> tuple[list[list[float]], str, str]:
    """(spans, video_id, note): merged, sorted [[start, end], ...] of the chunks the reference
    contexts came from, the video they belong to, and any mapping problem."""
    by_text = {d.page_content: d for d in chunks}
    spans, video_ids, missing = [], set(), 0
    for ctx in contexts:
        ctx = _HOP_PREFIX.sub("", ctx).strip()
        doc = by_text.get(ctx) or next(
            (d for d in chunks if ctx[:200] and ctx[:200] in d.page_content), None
        )
        if doc is None:
            missing += 1
            continue
        spans.append([doc.metadata["start"], doc.metadata["end"]])
        video_ids.add(doc.metadata["video_id"])
    spans.sort()
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    notes = []
    if missing:
        notes.append(f"{missing} context(s) not mapped to a span")
    if len(video_ids) > 1:
        notes.append("spans several videos")
    video_id = next(iter(video_ids)) if len(video_ids) == 1 else ""
    return merged, video_id, "; ".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", help="video_id (repeatable); default: all indexed videos")
    ap.add_argument("--size", type=int, default=45, help="number of candidates to generate")
    ap.add_argument("--window", type=float, default=120, help="chunk window in seconds")
    ap.add_argument("--gen-model", default="gpt-4o", help="writes questions + reference answers")
    ap.add_argument("--kg-model", default="gpt-4o-mini", help="knowledge-graph transforms (many calls)")
    args = ap.parse_args()

    video_ids = args.video or [v["video_id"] for v in registry.list_videos()]
    if not video_ids:
        raise SystemExit("No indexed videos.")

    chunks: list[Document] = []
    for vid in video_ids:
        chunks += time_windows(load_snippets(vid), vid, args.window)
    print(f"{len(video_ids)} video(s) -> {len(chunks)} chunks of ~{args.window:.0f}s")

    # generous retries: flaky connections otherwise abort the whole run (KG is cached, generation isn't)
    client_kw = {"max_retries": 8, "timeout": 120}
    gen_llm = LangchainLLMWrapper(ChatOpenAI(model=args.gen_model, **client_kw))
    kg_llm = LangchainLLMWrapper(ChatOpenAI(model=args.kg_model, **client_kw))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small", **client_kw))

    kg_path = OUT_DIR / f"kg_{'_'.join(sorted(video_ids))}_w{args.window:.0f}.json"
    kg = build_or_load_kg(chunks, kg_path, kg_llm, emb)

    generator = TestsetGenerator(llm=gen_llm, embedding_model=emb, knowledge_graph=kg, llm_context=LLM_CONTEXT)
    print(f"Generating {args.size} candidates...")
    testset = generator.generate(
        testset_size=args.size,
        query_distribution=query_distribution(gen_llm, kg),
        run_config=RunConfig(max_workers=8),
        raise_exceptions=False,
    )
    raw = testset.to_pandas()
    raw.to_json(OUT_DIR / "candidates.jsonl", orient="records", lines=True, force_ascii=False)

    # suggest a question_type for each candidate
    classifier = ChatOpenAI(model=args.kg_model, temperature=0, **client_kw).with_structured_output(TypeSuggestion)
    prompts = [
        CLASSIFY_PROMPT.format(
            question=r.user_input,
            answer=r.reference,
            n_contexts=len(r.reference_contexts),
            contexts="\n---\n".join(_HOP_PREFIX.sub("", c)[:1500] for c in r.reference_contexts),
        )
        for r in raw.itertuples()
    ]
    suggestions = classifier.batch(prompts, config={"max_concurrency": 8}, return_exceptions=True)

    rows = []
    for i, (r, sug) in enumerate(zip(raw.itertuples(), suggestions), 1):
        spans, video_id, note = spans_for(list(r.reference_contexts), chunks)
        if re.search(r"\b(transcript|passage|the context|the text)\b", r.user_input, re.I):
            note = (note + "; " if note else "") + "mentions transcript/context"
        ok = not isinstance(sug, Exception)
        rows.append({
            "id": f"gen-{i:03d}",
            "keep": "",                      # review: y / n
            "question_type": sug.question_type if ok else "",
            "question": r.user_input,
            "reference_answer": r.reference,
            "video_id": video_id,
            "spans": json.dumps(spans),
            "synthesizer": r.synthesizer_name,
            "query_style": getattr(r, "query_style", ""),
            "type_reason": sug.reason if ok else f"classifier failed: {sug}",
            "notes": note,
        })

    out = OUT_DIR / "candidates.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    df = pd.DataFrame(rows)
    print(f"\nWrote {len(rows)} candidates to {out}")
    print(df["synthesizer"].value_counts().to_string())
    print(df["question_type"].value_counts().to_string())
    print("\nNext: review candidates.csv, save the kept/edited rows as reviewed.csv, then "
          "python -m evals.dataset.upload --dry-run")


if __name__ == "__main__":
    main()
