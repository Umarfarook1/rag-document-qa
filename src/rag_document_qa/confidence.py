"""Confidence gate for the answer pipeline.

Two thresholds, both checked:

  1. Top-1 retrieval cosine similarity must be >= cosine_threshold (default 0.5).
     If the retriever's best match is too weak, no amount of LLM cleverness
     should produce a confident answer.

  2. Model self-rated confidence (parsed from the LLM's own output) must be
     >= self_rated_threshold (default 6 on a 0-10 scale).

If either gate fails, the pipeline returns an "I don't know" AnswerResult with
the failing reason recorded.

Returns the structured reason rather than a boolean so the caller can record
which gate triggered for the eval harness + observability.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_document_qa.types import RetrievedChunk

DEFAULT_COSINE_THRESHOLD = 0.5
DEFAULT_SELF_RATED_THRESHOLD = 6.0  # on a 0-10 scale


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Confidence Config."""
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD
    self_rated_threshold: float = DEFAULT_SELF_RATED_THRESHOLD


@dataclass(frozen=True, slots=True)
class ConfidenceVerdict:
    confident: bool
    score: float  # combined score in [0, 1] for downstream display
    reason: str | None  # None when confident; structured reason otherwise


def evaluate_confidence(
    retrieved: list[RetrievedChunk],
    self_rated: float | None,
    config: ConfidenceConfig | None = None,
) -> ConfidenceVerdict:
    """Apply both gates. self_rated may be None when the model didn't emit one;
    that is treated as failing the self-rated gate."""
    cfg = config or ConfidenceConfig()
    if not retrieved:
        return ConfidenceVerdict(confident=False, score=0.0, reason="no_chunks_retrieved")
    top_score = retrieved[0].score
    if top_score < cfg.cosine_threshold:
        return ConfidenceVerdict(
            confident=False,
            score=max(0.0, top_score),
            reason="top1_below_cosine_threshold",
        )
    if self_rated is None or self_rated < cfg.self_rated_threshold:
        return ConfidenceVerdict(
            confident=False,
            score=max(0.0, top_score),
            reason="model_self_rated_below_threshold",
        )
    # Combined score: average of normalised cosine and self-rated.
    combined = (max(0.0, top_score) + (self_rated / 10.0)) / 2.0
    return ConfidenceVerdict(confident=True, score=min(1.0, combined), reason=None)
