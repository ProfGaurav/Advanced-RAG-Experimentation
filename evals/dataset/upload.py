"""Upload the reviewed golden set (+ hand-written questions) to LangSmith and tag the version.

Run from rag/ with venv-eval active:
    python -m evals.dataset.upload --dry-run      # validate + show counts, upload nothing
    python -m evals.dataset.upload --tag v1       # create yt-rag-golden and tag it v1

Sources:
    evals/dataset/reviewed.csv   generated candidates after review (rows with keep = y)
    evals/dataset/manual.py      unanswerable + follow-up questions

LangSmith example shape (what the eval target and evaluators read):
    inputs  = {"question", "video_id", "chat_history": [{"role", "content"}, ...]}
    outputs = {"answer", "spans": [[start_s, end_s], ...], "question_type"}
    metadata = {"id", "question_type", "video_id", "source"};  split = question_type

The dataset is never deleted (experiments hang off it). It must not exist yet for a first
upload; later edits become v2 (see the EVAL_PLAN "Golden dataset" section).
"""

import yt_rag.config  # noqa: F401  (side effect: loads rag/.env)

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langsmith import Client

from evals.dataset.manual import manual_rows

DATASET = "yt-rag-golden"
REVIEWED = Path(__file__).resolve().parent / "reviewed.csv"
TYPES = {"factual", "keyword", "paraphrase", "broad", "multi_hop", "why", "follow_up", "unanswerable"}
MIN_PER_TYPE = 8


def reviewed_rows(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for r in df.to_dict("records"):
        if r.get("keep", "").strip().lower() not in ("y", "yes", "1"):
            continue
        rows.append({
            "id": r["id"],
            "question_type": r["question_type"].strip(),
            "question": r["question"].strip(),
            "reference_answer": r["reference_answer"].strip(),
            "video_id": r["video_id"].strip(),
            "spans": json.loads(r["spans"] or "[]"),
            "chat_history": json.loads(r.get("chat_history") or "[]"),
            "source": f"ragas:{r.get('synthesizer', '')}",
        })
    return rows


def validate(rows: list[dict]) -> list[str]:
    errors, seen = [], set()
    for r in rows:
        where = f"{r['id']}: "
        if r["id"] in seen:
            errors.append(where + "duplicate id")
        seen.add(r["id"])
        if r["question_type"] not in TYPES:
            errors.append(where + f"question_type {r['question_type']!r} not in {sorted(TYPES)}")
        if not r["question"] or not r["reference_answer"] or not r["video_id"]:
            errors.append(where + "empty question / reference_answer / video_id")
        if r["question_type"] != "unanswerable":
            if not r["spans"] or not all(len(s) == 2 and s[0] <= s[1] for s in r["spans"]):
                errors.append(where + f"needs spans [[start_s, end_s], ...], got {r['spans']}")
        if r["question_type"] == "follow_up" and not r["chat_history"]:
            errors.append(where + "follow_up needs chat_history")
    return errors


def to_example(r: dict) -> dict:
    return {
        "inputs": {"question": r["question"], "video_id": r["video_id"], "chat_history": r["chat_history"]},
        "outputs": {"answer": r["reference_answer"], "spans": r["spans"], "question_type": r["question_type"]},
        "metadata": {"id": r["id"], "question_type": r["question_type"], "video_id": r["video_id"],
                     "source": r["source"]},
        "split": r["question_type"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="validate and print counts only")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()

    if not REVIEWED.exists():
        raise SystemExit(f"{REVIEWED} not found: review candidates.csv and save it as reviewed.csv")
    rows = reviewed_rows(REVIEWED) + manual_rows()

    errors = validate(rows)
    counts = Counter(r["question_type"] for r in rows)
    print(f"{len(rows)} examples")
    for t in sorted(TYPES):
        flag = "" if counts[t] >= MIN_PER_TYPE else f"   <- below {MIN_PER_TYPE}"
        print(f"  {t:13} {counts[t]:3}{flag}")
    if errors:
        print("\nErrors:\n  " + "\n  ".join(errors))
        raise SystemExit(1)
    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return

    client = Client()
    if client.has_dataset(dataset_name=args.dataset):
        raise SystemExit(f"Dataset {args.dataset!r} already exists. Not overwriting: experiments "
                         "reference it. Edit it as a new version instead.")
    ds = client.create_dataset(args.dataset, description="Golden set for the YouTube RAG app (Ragas-generated + "
                               "hand-reviewed, plus hand-written unanswerable / follow-up questions)")
    client.create_examples(dataset_id=ds.id, examples=[to_example(r) for r in rows])
    client.update_dataset_tag(dataset_id=ds.id, as_of=datetime.now(timezone.utc), tag=args.tag)
    print(f"\nUploaded {len(rows)} examples to {args.dataset!r} and tagged the version {args.tag!r}")


if __name__ == "__main__":
    main()
