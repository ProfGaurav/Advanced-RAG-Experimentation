"""Input and retrieval guardrails that run before (or instead of) the answering LLM call.

Input guard, in order (first match wins):
    too_long      question longer than MAX_QUESTION_CHARS
    injection     obvious attempts to override the instructions ("ignore previous instructions", ...)
    moderation    OpenAI moderation endpoint flags the question (fails open on API errors)
    small_talk    greetings / thanks: canned reply, no retrieval

Relevance gate (after retrieval):
    no_relevant_context   no chunks, or best reranker score < MIN_RERANK_SCORE
                          -> fixed NOT_FOUND_ANSWER, the answering LLM is not called.

Off-topic questions ("capital of France?") are caught by the relevance gate: their
cross-encoder scores sit around -11, while on-topic questions score > -5 (see .env.example).
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document

log = logging.getLogger(__name__)

# Guardrail names (the chain's `guardrail` output key; None when the answer came from the LLM).
TOO_LONG = "too_long"
INJECTION = "injection"
MODERATION = "moderation"
SMALL_TALK = "small_talk"
NO_RELEVANT_CONTEXT = "no_relevant_context"

# Exchanges blocked for these reasons should not be persisted or sent back to the LLM as history.
BLOCKING = frozenset({TOO_LONG, INJECTION, MODERATION})

_INJECTION_PATTERNS = re.compile(
    r"""
    \b(ignore|disregard|forget|override|bypass)\b .{0,40}
        \b(previous|prior|above|earlier|all|your|the|any)\b .{0,20}
        \b(instructions?|rules?|prompts?|guidelines?|context|constraints?)\b
    | \b(reveal|show|print|repeat|leak|output)\b .{0,30} \b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b
    | \bwhat\s+(is|are)\s+your\s+(system\s+prompt|instructions|rules)\b
    | \byou\s+are\s+now\b
    | \b(act|behave|pretend)\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?(unrestricted|unfiltered|jailbroken|different)\b
    | \b(jailbreak|DAN\s+mode|developer\s+mode)\b
    | </?\s*(system|context|conversation)\s*>
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

_SMALL_TALK = re.compile(
    r"""^\s*(
        hi|hello|hey|hiya|yo|good\s+(morning|afternoon|evening)
      | thanks?(\s+you)?(\s+(so|very)\s+much)?|thank\s+you|thx|ty|cheers
      | ok(ay)?|cool|great|nice|awesome|got\s+it|perfect
      | bye|goodbye|see\s+you
    )(\s+(there|a\s+lot|again))?\s*[!.?]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class GuardResult:
    """`reason` is None when the question may proceed; otherwise `reply` is shown instead."""

    reason: str | None = None
    reply: str | None = None

    @property
    def allowed(self) -> bool:
        return self.reason is None


ALLOWED = GuardResult()

Moderator = Callable[[str], bool]  # returns True when the text is flagged


def openai_moderator(model: str = "omni-moderation-latest") -> Moderator:
    """Moderator backed by OpenAI's (free) moderation endpoint. Fails open on API errors."""
    from openai import OpenAI  # installed with langchain-openai

    client = OpenAI()

    def is_flagged(text: str) -> bool:
        try:
            return bool(client.moderations.create(model=model, input=text).results[0].flagged)
        except Exception as exc:  # network / quota / outage: don't block the user, but log it
            log.warning("Moderation check failed, allowing question: %s", exc)
            return False

    return is_flagged


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_PATTERNS.search(text))


def is_small_talk(text: str) -> bool:
    return bool(_SMALL_TALK.match(text))


@dataclass
class InputGuard:
    """Checks a user question before any retrieval or LLM call."""

    video_title: str = "this video"
    max_chars: int = 1000
    moderator: Moderator | None = None

    def __call__(self, question: str) -> GuardResult:
        q = question.strip()
        if len(q) > self.max_chars:
            return GuardResult(
                TOO_LONG,
                f"Your question is too long ({len(q)} characters). "
                f"Please keep it under {self.max_chars} characters.",
            )
        if looks_like_injection(q):
            log.warning("Blocked possible prompt injection: %r", q[:200])
            return GuardResult(
                INJECTION,
                f'I can only answer questions about "{self.video_title}" from its transcript, '
                "and I can't change or reveal my instructions.",
            )
        if self.moderator is not None and self.moderator(q):
            log.warning("Blocked question flagged by moderation: %r", q[:200])
            return GuardResult(MODERATION, "I can't help with that request.")
        if is_small_talk(q):
            return GuardResult(SMALL_TALK, f'Happy to help! Ask me anything about "{self.video_title}".')
        return ALLOWED


def has_relevant_context(docs: list[Document], min_score: float | None) -> bool:
    """False when nothing was retrieved, or every chunk's `rerank_score` is below `min_score`.

    Chunks without a `rerank_score` (no reranker configured) only fail on the empty check.
    """
    if not docs:
        return False
    if min_score is None:
        return True
    scores = [d.metadata["rerank_score"] for d in docs if "rerank_score" in d.metadata]
    return not scores or max(scores) >= min_score
