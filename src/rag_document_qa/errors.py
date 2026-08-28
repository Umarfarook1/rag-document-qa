"""Domain exceptions for the RAG pipeline.

Every public-surface failure path raises a `RagError` with one of the stable
codes in `ErrorCode`. Downstream callers (CLI, agent, dashboard) can switch
on `err.code` and recover or surface a structured response.

Implementations that wrap third-party libraries should catch the library's
exception type at the seam and translate it via a helper rather than letting
raw `httpx.ReadTimeout` or `faiss.RuntimeError` leak through.
"""

from __future__ import annotations

from typing import Literal

ErrorCode = Literal[
    "corpus_empty",
    "index_not_built",
    "embedding_dim_mismatch",
    "embedder_mismatch",
    "invalid_corpus",
    "loader_failed",
    "answer_generation_failed",
    "no_confident_answer",
    "rate_limited",
    "unknown",
]


class RagError(Exception):
    """A structured error from any layer of the RAG pipeline.

    Attributes:
        code: One of the stable `ErrorCode` literals. Safe for downstream
            consumers (an LLM agent, the CLI, a dashboard) to switch on.
        message: Human-readable description, safe to surface to the caller.
        details: Optional dict of extra fields. Keys are not part of the public
            contract, so keep them additive (don't rename existing ones).
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Init."""
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"error": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result
