"""Upload a tiny 5-question dataset to LangSmith to check the Ragas + LangSmith setup end to end.

Run from rag/ with venv-eval active:
    python -m evals.smoke_dataset

Questions come from the Lex Fridman x Demis Hassabis interview (video Gfr50f6ZBvo).
Spans are [start_s, end_s] of the transcript passage that answers the question,
taken from data/transcripts/Gfr50f6ZBvo.json. The dataset is deleted and recreated
on every run: this is a smoke test, not the versioned golden set.
"""

import yt_rag.config  # noqa: F401  (side effect: loads rag/.env, incl. LANGSMITH_API_KEY)
from langsmith import Client

DATASET = "yt-rag-smoke"
VIDEO_ID = "Gfr50f6ZBvo"

# (question, reference answer, [start_s, end_s], question_type)
ROWS = [
    (
        "What game did Demis Hassabis write that had AI as a core gameplay component?",
        "Theme Park, a sandbox simulation game in which the AI reacted to the way the player "
        "played, so every game was unique. It sold millions of copies.",
        [805, 845],
        "factual",
    ),
    (
        "Who called chess the drosophila of intelligence?",
        "Garry Kasparov. Hassabis says he loves the phrase and thinks Kasparov is right, because "
        "chess has gone hand in hand with AI since the start of the field.",
        [1225, 1250],
        "keyword",
    ),
    (
        "What is the protein folding problem?",
        "Predicting a protein's 3D structure computationally from its one-dimensional amino acid "
        "sequence alone. It has been a grand challenge in biology for over 50 years.",
        [2365, 2395],
        "factual",
    ),
    (
        "Who first articulated the protein folding problem, and when?",
        "Christian Anfinsen, a Nobel Prize winner, in 1972.",
        [2385, 2400],
        "keyword",
    ),
    (
        "Does Hassabis think Turing intended the Turing test to be a rigorous formal test?",
        "No. He thinks Turing meant it more as a thought experiment, almost philosophy: the 1950 "
        "paper didn't specify key parameters such as the judge's knowledge or how much time the "
        "judge would have.",
        [137, 175],
        "why",
    ),
]


def main() -> None:
    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        client.delete_dataset(dataset_name=DATASET)
    dataset = client.create_dataset(DATASET, description="5-question smoke test of the eval setup")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": question, "video_id": VIDEO_ID},
                "outputs": {"answer": answer, "span": span, "question_type": qtype},
                "metadata": {"video_id": VIDEO_ID, "question_type": qtype},
            }
            for question, answer, span, qtype in ROWS
        ],
    )
    print(f"Uploaded {len(ROWS)} examples to LangSmith dataset {DATASET!r}")


if __name__ == "__main__":
    main()
