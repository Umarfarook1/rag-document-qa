"""Anthropic Claude answer generator.

Behaviour:
  1. Builds a context block from the retrieved chunks, each tagged `[chunk_N]`.
  2. Asks Claude to answer with mandatory `[chunk_N]` citations + a self-rated
     confidence on a 0-10 scale. The system prompt makes this a hard requirement.
  3. Parses the response. If no citations were emitted, retries ONCE with a
     stricter follow-up. Still no citations → returns an unconfident result with
     reason `no_citations_emitted`.
  4. Runs the confidence gate. If it fails, returns the "I don't know" path with
     the gate's reason recorded.

The Anthropic SDK is imported lazily so the package can be tested with the
`Fake` generator alone in CI.
"""

from __future__ import annotations

import os
import re
from typing import Any

from rag_document_qa.confidence import ConfidenceConfig, evaluate_confidence
from rag_document_qa.errors import RagError
from rag_document_qa.types import AnswerResult, Question, RetrievedChunk

CITATION_RE = re.compile(r"\[chunk_(\d+)\]", re.IGNORECASE)
SELF_RATED_RE = re.compile(r"confidence[:=\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)

_SYSTEM_PROMPT = """\
You answer questions strictly from the provided context chunks.

Rules (MANDATORY):
- Cite supporting chunks with `[chunk_N]` for every factual claim.
- If the chunks don't contain the answer, say "I don't know" - do NOT invent.
- End your response with: `Confidence: X/10` (where X is your self-rated 0-10
  confidence in the answer based on how well the chunks support it).
- Do not output anything past the Confidence line.
"""


class AnthropicAnswerGenerator:
    DEFAULT_MODEL = "claude-haiku-4-5"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        confidence_config: ConfidenceConfig | None = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RagError(
                code="loader_failed",
                message="anthropic SDK not installed; run `pip install anthropic`",
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RagError(
                code="answer_generation_failed",
                message="ANTHROPIC_API_KEY env var is required",
            )

        self._model_id = model_id or self.DEFAULT_MODEL
        self._client: Any = anthropic.Anthropic(api_key=key)
        self._cfg = confidence_config or ConfidenceConfig()
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"anthropic:{self._model_id}"

    def generate(
        self,
        question: Question,
        retrieved: list[RetrievedChunk],
    ) -> AnswerResult:
        if not retrieved:
            return AnswerResult(
                text="I couldn't find any relevant context.",
                cited_chunk_ids=[],
                retrieved=[],
                confident=False,
                confidence_score=0.0,
                no_answer_reason="no_chunks_retrieved",
            )

        context_block = _format_context(retrieved)
        user_msg = (
            f"Context:\n\n{context_block}\n\nQuestion: {question.text}\n\nAnswer with citations:"
        )

        text = self._call(user_msg)
        cited_ids = _parse_citations(text, retrieved)

        if not cited_ids:
            # Single retry with a stricter follow-up.
            retry_msg = (
                user_msg + "\n\nReminder: every claim must include a [chunk_N] citation. Try again."
            )
            text = self._call(retry_msg)
            cited_ids = _parse_citations(text, retrieved)

        self_rated = _parse_self_rated(text)
        verdict = evaluate_confidence(retrieved, self_rated, self._cfg)

        if not cited_ids and verdict.confident:
            # Citations are mandatory; downgrade to unconfident.
            return AnswerResult(
                text=text,
                cited_chunk_ids=[],
                retrieved=retrieved,
                confident=False,
                confidence_score=verdict.score,
                no_answer_reason="no_citations_emitted",
            )

        if not verdict.confident:
            return AnswerResult(
                text="I couldn't find a confident answer. Closest evidence below.",
                cited_chunk_ids=cited_ids,
                retrieved=retrieved,
                confident=False,
                confidence_score=verdict.score,
                no_answer_reason=verdict.reason,
            )

        return AnswerResult(
            text=text,
            cited_chunk_ids=cited_ids,
            retrieved=retrieved,
            confident=True,
            confidence_score=verdict.score,
            no_answer_reason=None,
        )

    def _call(self, user_msg: str) -> str:
        try:
            response: Any = self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            raise RagError(
                code="answer_generation_failed",
                message=f"Anthropic call failed: {e}",
                details={"model": self._model_id},
            ) from e
        return "".join(block.text for block in response.content if hasattr(block, "text"))


def _format_context(retrieved: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for r in retrieved:
        lines.append(f"[{r.chunk.id}] (score={r.score:.3f})")
        lines.append(r.chunk.text)
        lines.append("")
    return "\n".join(lines)


def _parse_citations(text: str, retrieved: list[RetrievedChunk]) -> list[str]:
    """Match `[chunk_N]` references back to actual chunk ids.

    The model writes `[chunk_3]`; we map that to the third retrieved chunk's id
    (1-based). Only return ids that correspond to a retrieved chunk to avoid
    accepting hallucinated citations.
    """
    matches = CITATION_RE.findall(text)
    out: list[str] = []
    seen: set[str] = set()
    for m in matches:
        try:
            idx = int(m) - 1
        except ValueError:
            continue
        if 0 <= idx < len(retrieved):
            cid = retrieved[idx].chunk.id
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


def _parse_self_rated(text: str) -> float | None:
    """Extract `Confidence: X/10` -> X. Returns None if missing."""
    matches = SELF_RATED_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])  # take the last occurrence
    except ValueError:
        return None
