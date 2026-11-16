"""Hand-written golden-set questions that Ragas can't generate: unanswerable and follow-up.

Video Gfr50f6ZBvo (Lex Fridman Podcast #299, Demis Hassabis). Every span was checked against
data/transcripts/Gfr50f6ZBvo.json; every unanswerable topic has zero mentions in it (2026-11-16).

- unanswerable: not covered by the video. Reference = the app's NOT_FOUND_ANSWER. The app should
  refuse (relevance gate or "I couldn't find this"); `refusal_correct` scores that.
- follow_up: only answerable with `chat_history` (1 previous exchange). The app must rewrite the
  follow-up into a standalone question first; condense on/off is tested on this split.
"""

from yt_rag.chains.prompts import NOT_FOUND_ANSWER

VIDEO = "Gfr50f6ZBvo"


def _turn(question: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


UNANSWERABLE = [
    "How much did Google pay to acquire DeepMind?",
    "What is Demis Hassabis's favourite movie?",
    "What does Hassabis think about Bitcoin and cryptocurrency?",
    "Which programming language is AlphaFold written in?",
    "How many employees does DeepMind have?",
    "What does Hassabis think of ChatGPT?",
    "What does Hassabis say about Tesla's self-driving cars?",
    "What does Hassabis think about AI systems that play poker?",
]

FOLLOW_UPS = [
    # (history question, history answer, follow-up, reference answer, [start_s, end_s])
    (
        "What game did Demis Hassabis write that had AI as a core gameplay component?",
        "He wrote Theme Park, a simulation game where the AI was a core part of the gameplay.",
        "What other game does he compare it to?",
        "SimCity: Theme Park was one of the first sandbox games, along with SimCity.",
        [805, 845],
    ),
    (
        "What game did Demis Hassabis write that had AI as a core gameplay component?",
        "He wrote Theme Park, a simulation game where the AI was a core part of the gameplay.",
        "How well did it sell?",
        "It sold millions of copies around the world.",
        [805, 845],
    ),
    (
        "Who called chess the drosophila of intelligence?",
        "Garry Kasparov.",
        "Why does Hassabis agree with him?",
        "Because chess has gone hand in hand with AI from the beginning of the field; every AI "
        "practitioner, starting with Turing and Claude Shannon, worked on it.",
        [1225, 1265],
    ),
    (
        "What is the protein folding problem?",
        "Predicting a protein's 3D structure computationally from its amino acid sequence.",
        "Who first articulated it, and when?",
        "Christian Anfinsen, a Nobel Prize winner, in 1972.",
        [2385, 2400],
    ),
    (
        "Does Hassabis think Turing intended the Turing test to be a rigorous formal test?",
        "No, he sees it as more of a thought experiment; Turing didn't specify key parameters.",
        "What example does he give of someone claiming it had been passed?",
        "About a decade earlier someone claimed a pass with a fairly standard program that pretended "
        "to be a child, so the judges thought they were talking to a kid.",
        [170, 200],
    ),
    (
        "What did Hassabis think was missing from the hand-coded chess programs?",
        "Generality and learning: they distilled grandmaster knowledge but couldn't even play tic-tac-toe.",
        "Which DeepMind systems were built to address that?",
        "AlphaGo, then AlphaZero and MuZero.",
        [1325, 1365],
    ),
    (
        "Which areas does Hassabis see as the biggest opportunities for AI to help science?",
        "Disease and biology, and energy and climate.",
        "Who did DeepMind collaborate with on the fusion work?",
        "EPFL, the Swiss technical institute, which let them use its test reactor.",
        [4440, 4500],
    ),
    (
        "When does Hassabis do his deep thinking?",
        "In the quiet hours of the morning, roughly between one and three a.m., with music on.",
        "Which philosopher has been his recent favourite?",
        "Spinoza.",
        [7385, 7415],
    ),
]


def manual_rows() -> list[dict]:
    """Rows in the same shape as reviewed.csv (spans / chat_history as Python lists)."""
    rows = []
    for i, q in enumerate(UNANSWERABLE, 1):
        rows.append({
            "id": f"man-u{i:02d}", "question_type": "unanswerable", "question": q,
            "reference_answer": NOT_FOUND_ANSWER, "video_id": VIDEO, "spans": [], "chat_history": [],
            "source": "manual",
        })
    for i, (hq, ha, q, ref, span) in enumerate(FOLLOW_UPS, 1):
        rows.append({
            "id": f"man-f{i:02d}", "question_type": "follow_up", "question": q,
            "reference_answer": ref, "video_id": VIDEO, "spans": [span], "chat_history": _turn(hq, ha),
            "source": "manual",
        })
    return rows
