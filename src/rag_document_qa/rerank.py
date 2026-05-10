"""Reranker implementations.

`IdentityReranker` passes through unchanged (used in tests + when a real
cross-encoder isn't installed). `ReverseReranker` flips the order; useful for
tests that want to verify the retriever respects the reranker's output.

A real cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
can be added later as a separate module; the `Reranker` Protocol is the seam.
"""

from __future__ import annotations

from rag_document_qa.types import RetrievedChunk


class IdentityReranker:
    @property
    def name(self) -> str:
        return "identity"

    def rerank(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        return list(retrieved)


class ReverseReranker:
    """Reverses the input order. Test-only impl for pipeline ordering tests."""

    @property
    def name(self) -> str:
        return "reverse"

    def rerank(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        # Re-rank ascending of original score (worst first), with re-numbered ranks.
        flipped = list(reversed(retrieved))
        return [
            RetrievedChunk(chunk=r.chunk, score=r.score, rank=i)
            for i, r in enumerate(flipped, start=1)
        ]
