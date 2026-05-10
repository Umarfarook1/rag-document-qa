"""Retriever orchestrator: embed query → search index → optionally rerank.

Stateless. Composes Embedder + VectorIndex + (optional) Reranker. The embedder
must match the one stamped into the index's metadata; mismatch is detected and
raised as `RagError(embedder_mismatch)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_document_qa.errors import RagError
from rag_document_qa.protocols import Embedder, Reranker, VectorIndex
from rag_document_qa.types import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Knobs for retrieval. `top_k_retrieve` is the candidate pool fed to the
    reranker (if any). `top_k_return` is what the caller actually sees.
    """

    top_k_retrieve: int = 20
    top_k_return: int = 5


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        reranker: Reranker | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._reranker = reranker
        self._cfg = config or RetrieverConfig()
        self._validate_embedder_matches_index()

    def _validate_embedder_matches_index(self) -> None:
        meta = self._index.metadata
        if meta.embedder_name != self._embedder.name:
            raise RagError(
                code="embedder_mismatch",
                message=(
                    f"index built with embedder {meta.embedder_name!r} but query "
                    f"embedder is {self._embedder.name!r}"
                ),
                details={
                    "index_embedder": meta.embedder_name,
                    "query_embedder": self._embedder.name,
                },
            )
        if meta.embedding_dim != self._embedder.dim:
            raise RagError(
                code="embedding_dim_mismatch",
                message=(f"index dim {meta.embedding_dim} != embedder dim {self._embedder.dim}"),
            )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        embeddings = self._embedder.encode([query])
        candidates = self._index.search(embeddings[0], k=self._cfg.top_k_retrieve)
        if self._reranker is not None and candidates:
            candidates = self._reranker.rerank(query, candidates)
        return candidates[: self._cfg.top_k_return]
