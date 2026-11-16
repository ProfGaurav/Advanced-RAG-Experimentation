"""Ragas 0.4 metrics wrapped as LangSmith evaluators, plus deterministic checks.

LangSmith is the system of record (datasets, experiments, traces); Ragas only supplies the
metric logic. Each evaluator is a plain async function that LangSmith calls once per example.

The target (see smoke_run.py) must return:
    {"answer": str, "contexts": list[str], "timestamps": list[float | None],
     "ends": list[float | None], "guardrail": str | None}
Dataset examples carry:
    inputs  = {"question", "video_id"}
    outputs = {"answer", "span": [start_s, end_s]} (or "spans": [[s, e], ...]), "question_type"}

LangSmith passes arguments by parameter NAME (inputs / outputs / reference_outputs),
so don't rename those parameters.
"""

import yt_rag.config  # noqa: F401  (side effect: loads rag/.env before AsyncOpenAI() is created)

import os

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from yt_rag.chains.guardrails import NO_RELEVANT_CONTEXT
from yt_rag.chains.prompts import NOT_FOUND_ANSWER

_client = AsyncOpenAI()

# The judge is FIXED across all experiments; if it changes, old and new scores can't be compared.
# It should be stronger than the model under test (gpt-4o judging gpt-4o-mini).
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
JUDGE = llm_factory(JUDGE_MODEL, client=_client, temperature=0)
EMB = OpenAIEmbeddings(client=_client, model="text-embedding-3-small")

# Ragas 0.4 metrics take different ascore() arguments and reject extra ones, so each metric
# declares which fields it needs. This maps a Ragas argument name to where it comes from.
def _user_input(inputs: dict) -> str:
    """The question as the judge should see it. A follow-up ("and when was that?") is meaningless
    alone, so prior turns are included; the app's own rewrite is NOT used, so a bad rewrite
    still shows up as a bad score."""
    history = inputs.get("chat_history") or []
    if not history:
        return inputs["question"]
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history]
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nFollow-up question: {inputs['question']}"


_FIELDS = {
    "user_input": lambda inputs, outputs, ref: _user_input(inputs),
    "response": lambda inputs, outputs, ref: outputs["answer"],
    "retrieved_contexts": lambda inputs, outputs, ref: outputs["contexts"],
    "reference": lambda inputs, outputs, ref: ref["answer"],
}


def _target_failed(outputs: dict | None) -> bool:
    """The pipeline raised (e.g. OpenAI 429), so LangSmith passes empty outputs: nothing to score.
    Scoring them anyway turns failures into fake results (an empty answer 'correctly' didn't refuse)."""
    return not outputs or "answer" not in outputs


def ragas_evaluator(metric, fields: tuple[str, ...]):
    """Wrap a Ragas collections metric as a LangSmith evaluator named after the metric."""
    key = metric.name

    async def _eval(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
        if _target_failed(outputs):
            return {"key": key, "score": None, "comment": "skipped: pipeline run failed"}
        if "reference" in fields and (reference_outputs or {}).get("question_type") == "unanswerable":
            # the reference is just the refusal text; refusal_correct scores these instead
            return {"key": key, "score": None, "comment": "skipped: unanswerable question"}
        if "retrieved_contexts" in fields and not outputs.get("contexts"):
            # a guardrail stopped the chain before retrieval: nothing to judge
            return {"key": key, "score": None, "comment": f"skipped: guardrail={outputs.get('guardrail')}"}
        kwargs = {f: _FIELDS[f](inputs, outputs, reference_outputs or {}) for f in fields}
        result = await metric.ascore(**kwargs)
        return {"key": key, "score": result.value, "comment": result.reason}

    # LangSmith names an evaluator (and its error feedback) after the function, so both must be set;
    # otherwise an evaluator that raises shows up as a column called "_eval"
    _eval.__name__ = _eval.__qualname__ = key
    return _eval


GENERATION_EVALS = [
    ragas_evaluator(Faithfulness(llm=JUDGE), ("user_input", "response", "retrieved_contexts")),
    ragas_evaluator(AnswerRelevancy(llm=JUDGE, embeddings=EMB), ("user_input", "response")),
    ragas_evaluator(AnswerCorrectness(llm=JUDGE, embeddings=EMB), ("user_input", "response", "reference")),
]

RETRIEVAL_EVALS = [
    ragas_evaluator(ContextPrecision(llm=JUDGE), ("user_input", "reference", "retrieved_contexts")),
    ragas_evaluator(ContextRecall(llm=JUDGE), ("user_input", "retrieved_contexts", "reference")),
]


# --- Deterministic evaluators: free, no LLM calls, no judge noise ---

def _spans(reference_outputs: dict) -> list[list[float]]:
    """Reference evidence spans; [] for unanswerable questions. Accepts `spans` or legacy `span`."""
    if reference_outputs.get("spans"):
        return reference_outputs["spans"]
    return [reference_outputs["span"]] if reference_outputs.get("span") else []


def _overlaps(start, end, lo, hi, slack: float = 30) -> bool:
    """Chunk [start, end] overlaps reference span [lo, hi], with `slack` seconds of tolerance."""
    if start is None:
        return False
    end = start if end is None else end
    return start <= hi + slack and end >= lo - slack


def n_contexts(outputs: dict) -> dict:
    """Chunks that reached the LLM. Below k exposes post-filter starvation (GAPS_AND_RISKS gap #1)."""
    if _target_failed(outputs):
        return {"key": "n_contexts", "score": None, "comment": "skipped: pipeline run failed"}
    return {"key": "n_contexts", "score": len(outputs.get("contexts") or [])}


def timestamp_hit(outputs: dict, reference_outputs: dict) -> dict:
    """1 if any retrieved chunk overlaps any reference span (±30 s), else 0. None if no span."""
    if _target_failed(outputs):
        return {"key": "timestamp_hit@k", "score": None, "comment": "skipped: pipeline run failed"}
    spans = _spans(reference_outputs)
    if not spans:
        return {"key": "timestamp_hit@k", "score": None, "comment": "no reference span (unanswerable)"}
    chunks = list(zip(outputs.get("timestamps") or [], outputs.get("ends") or []))
    hit = any(_overlaps(s, e, lo, hi) for s, e in chunks for lo, hi in spans)
    return {"key": "timestamp_hit@k", "score": int(hit)}


def refusal_correct(outputs: dict, reference_outputs: dict) -> dict:
    """Unanswerable -> the app must refuse; answerable -> it must not. 1 = right call.

    Refusal = relevance gate fired, or the LLM replied with the NOT_FOUND_ANSWER sentence.
    """
    if _target_failed(outputs):
        return {"key": "refusal_correct", "score": None, "comment": "skipped: pipeline run failed"}
    answer = (outputs.get("answer") or "").strip()
    refused = outputs.get("guardrail") == NO_RELEVANT_CONTEXT or NOT_FOUND_ANSWER.rstrip(".") in answer
    should_refuse = reference_outputs.get("question_type") == "unanswerable"
    return {"key": "refusal_correct", "score": int(refused == should_refuse),
            "comment": f"refused={refused}, should_refuse={should_refuse}"}


DETERMINISTIC_EVALS = [n_contexts, timestamp_hit, refusal_correct]
