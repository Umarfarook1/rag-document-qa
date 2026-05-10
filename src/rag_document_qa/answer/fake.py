"""Configurable fake answer generator for tests.

Behaviour modes:
  - "echo_first":     answer = top-1 chunk's text, citing it
  - "no_citation":    answer = static text, no citations (forces re-prompt path)
  - "low_confidence": answer = "I don't know", confidence below threshold
  - "raises":         raises RagError(answer_generation_failed)

Tests can swap the mode to drive every branch of the pipeline without an API.
"""

from __future__ import annotations

from typing import Literal

from rag_document_qa.errors import RagError
from rag_document_qa.types import AnswerResult, Question, RetrievedChunk

FakeMode = Literal["echo_first", "no_citation", "low_confidence", "raises"]


class FakeAnswerGenerator:
    def __init__(self, mode: FakeMode = "echo_first") -> None:
        self._mode = mode

    @property
    def name(self) -> str:
        return f"fake-{self._mode}"

    def generate(
        self,
        question: Question,
        retrieved: list[RetrievedChunk],
    ) -> AnswerResult:
        if self._mode == "raises":
            raise RagError(
                code="answer_generation_failed",
                message="FakeAnswerGenerator(raises) mode triggered",
            )
        if self._mode == "low_confidence":
            return AnswerResult(
                text="I couldn't find a confident answer.",
                cited_chunk_ids=[],
                retrieved=retrieved,
                confident=False,
                confidence_score=0.2,
                no_answer_reason="model_self_rated_below_threshold",
            )
        if self._mode == "no_citation":
            return AnswerResult(
                text="Some answer with no citations.",
                cited_chunk_ids=[],
                retrieved=retrieved,
                confident=True,
                confidence_score=0.7,
                no_answer_reason=None,
            )
        # echo_first
        if not retrieved:
            return AnswerResult(
                text="No context retrieved.",
                cited_chunk_ids=[],
                retrieved=retrieved,
                confident=False,
                confidence_score=0.0,
                no_answer_reason="no_chunks_retrieved",
            )
        top = retrieved[0]
        return AnswerResult(
            text=f"Based on [{top.chunk.id}]: {top.chunk.text}",
            cited_chunk_ids=[top.chunk.id],
            retrieved=retrieved,
            confident=True,
            confidence_score=top.score,
            no_answer_reason=None,
        )
