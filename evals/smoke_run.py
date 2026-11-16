"""Run the app's real QA pipeline over the `yt-rag-smoke` dataset and score it with Ragas.

Run from rag/ with venv-eval active (after `python -m evals.smoke_dataset`):
    python -m evals.smoke_run

Creates one LangSmith experiment (`smoke-baseline-...`) on the dataset: a row per question with
every evaluator's score and the full trace. The pipeline comes from `yt_rag.qa`, exactly as the
Streamlit app builds it (same .env settings, reranker, guardrails), so results describe the app.
"""

import yt_rag.config  # noqa: F401  (side effect: loads rag/.env, incl. LangSmith + OpenAI keys)

import argparse
import asyncio
import subprocess
from datetime import datetime

from langchain_core.runnables import Runnable
from langsmith import Client, aevaluate

from evals.evaluators import DETERMINISTIC_EVALS, GENERATION_EVALS, JUDGE_MODEL, RETRIEVAL_EVALS
from yt_rag.config import PROJECT_ROOT, get_settings
from yt_rag.qa import build_chain_for, load_resources, to_chat_history

DATASET = "yt-rag-smoke"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"

resources = load_resources()  # FAISS store + LLM + reranker: load once, reuse for every question
_chains: dict[str, Runnable] = {}


def chain_for(video_id: str) -> Runnable:
    if video_id not in _chains:
        _chains[video_id] = build_chain_for(resources, video_id)
    return _chains[video_id]


async def target(inputs: dict) -> dict:
    """One question through the app chain. Returns the answer AND the retrieved contexts."""
    history = to_chat_history(inputs.get("chat_history") or [])  # follow-up questions carry prior turns
    result = await chain_for(inputs["video_id"]).ainvoke(
        {"question": inputs["question"], "chat_history": history}
    )
    docs = result["docs"]
    return {
        "standalone_question": result["standalone_question"],  # the app's rewrite of a follow-up
        "answer": result["answer"],  # raw answer, without the display-only Citations footer
        "contexts": [d.page_content for d in docs],
        "timestamps": [d.metadata.get("start") for d in docs],
        "ends": [d.metadata.get("end") for d in docs],
        "rerank_scores": [d.metadata.get("rerank_score") for d in docs],
        "guardrail": result["guardrail"],
    }


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the app pipeline over a LangSmith dataset and score it.")
    ap.add_argument("--dataset", default=DATASET, help="LangSmith dataset name (default: yt-rag-smoke)")
    ap.add_argument("--version", help="dataset version tag, e.g. v1 (default: latest)")
    ap.add_argument("--prefix", help="experiment name prefix (default: <dataset>-baseline)")
    ap.add_argument("--concurrency", type=int, default=2)
    return ap.parse_args()


async def main() -> None:
    args = parse_args()
    s = get_settings()
    metadata = {
        # what was tested (LangSmith can filter / group experiments by these keys)
        "chat_model": s.chat_model,
        "temperature": s.temperature,
        "embedding_model": s.embedding_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "retriever_k": s.retriever_k,
        "fetch_k": s.fetch_k,
        "reranker": s.reranker,
        "reranker_model": s.reranker_model if s.reranker != "none" else None,
        "rerank_candidates": s.rerank_candidates,
        "min_rerank_score": s.min_rerank_score,
        # how it was judged
        "judge_model": JUDGE_MODEL,
        "git_sha": git_sha(),
        "dataset_version": args.version or "latest",
    }

    # a version tag pins the exact examples, so later experiments compare like with like
    data = (
        list(Client().list_examples(dataset_name=args.dataset, as_of=args.version))
        if args.version else args.dataset
    )
    prefix = args.prefix or ("smoke-baseline" if args.dataset == DATASET else f"{args.dataset}-baseline")

    results = await aevaluate(
        target,
        data=data,
        evaluators=[*RETRIEVAL_EVALS, *GENERATION_EVALS, *DETERMINISTIC_EVALS],
        experiment_prefix=prefix,
        metadata=metadata,
        max_concurrency=args.concurrency,  # set explicitly: the default differs between versions
    )

    df = results.to_pandas()
    score_cols = [c for c in df.columns if c.startswith("feedback.")]

    print(f"\nExperiment: {results.experiment_name}")
    if "error" in df.columns and df["error"].notna().any():
        errs = df["error"].dropna().astype(str)
        print(f"\n!! {len(errs)} of {len(df)} pipeline runs FAILED; their scores are blank, not zero.")
        print("   Most common error:", errs.str.slice(0, 160).value_counts().index[0])
        print("   Fix the cause and re-run before reading anything into the scores below.")
    print("\nMean scores:")
    print(df[score_cols].mean(numeric_only=True).round(3).to_string())

    qtype_col = "reference.question_type"
    if qtype_col in df.columns:
        print("\nMean scores per question_type:")
        by_type = df.groupby(qtype_col)[score_cols].mean(numeric_only=True)
        by_type.columns = [c.removeprefix("feedback.") for c in by_type.columns]
        by_type.insert(0, "n", df[qtype_col].value_counts())
        print(by_type.round(2).to_string())

    print("\nPer question:")
    per_q = df[["inputs.question", *score_cols]].copy()
    per_q["inputs.question"] = per_q["inputs.question"].str.slice(0, 50)
    per_q.columns = [c.removeprefix("feedback.").removeprefix("inputs.") for c in per_q.columns]
    print(per_q.round(2).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved local copy: {out}")


if __name__ == "__main__":
    asyncio.run(main())
