"""Pure-numpy in-memory vector index with cosine similarity search.

Used as the default for tests + small-scale local use. For production-scale
indexing (1M+ chunks) the FAISS-backed impl in `index/faiss.py` is the right
choice; both satisfy the same `VectorIndex` Protocol.

Persistence is via numpy's `.npz` for the embeddings + a sidecar JSON for the
chunks and metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rag_document_qa.errors import RagError
from rag_document_qa.types import Chunk, IndexMetadata, RetrievedChunk


class InMemoryVectorIndex:
    """Numpy cosine-similarity index. Embeddings are L2-normalised on build,
    so cosine similarity is just an inner product at search time.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: NDArray[np.float32] | None = None
        self._metadata: IndexMetadata | None = None

    @property
    def metadata(self) -> IndexMetadata:
        if self._metadata is None:
            raise RagError(code="index_not_built", message="index has not been built yet")
        return self._metadata

    def build(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32],
        metadata: IndexMetadata,
    ) -> None:
        if not chunks:
            raise RagError(code="corpus_empty", message="cannot build an index from zero chunks")
        if embeddings.shape[0] != len(chunks):
            raise RagError(
                code="invalid_corpus",
                message=(
                    f"chunk count {len(chunks)} does not match embedding count "
                    f"{embeddings.shape[0]}"
                ),
            )
        if embeddings.shape[1] != metadata.embedding_dim:
            raise RagError(
                code="embedding_dim_mismatch",
                message=(
                    f"embeddings have dim {embeddings.shape[1]} but metadata claims "
                    f"{metadata.embedding_dim}"
                ),
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid divide-by-zero for any all-zero vectors; replace those rows with zeros.
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = (embeddings / norms).astype(np.float32)
        self._chunks = list(chunks)
        self._metadata = metadata

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int,
    ) -> list[RetrievedChunk]:
        if self._embeddings is None or self._metadata is None:
            raise RagError(code="index_not_built", message="search() called before build()")
        if query_embedding.ndim == 1:
            query = query_embedding.reshape(1, -1)
        else:
            query = query_embedding
        if query.shape[1] != self._metadata.embedding_dim:
            raise RagError(
                code="embedding_dim_mismatch",
                message=(
                    f"query dim {query.shape[1]} != index dim {self._metadata.embedding_dim}"
                ),
            )
        q_norm = float(np.linalg.norm(query))
        if q_norm > 0:
            query = query / q_norm
        scores = (self._embeddings @ query.T).flatten()  # cosine since both are normalised
        if k <= 0:
            return []
        k_clamped = min(k, len(self._chunks))
        # argpartition for top-k, then sort the top-k slice.
        top_idx = np.argpartition(-scores, k_clamped - 1)[:k_clamped]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]
        return [
            RetrievedChunk(
                chunk=self._chunks[int(i)],
                score=float(scores[int(i)]),
                rank=rank,
            )
            for rank, i in enumerate(top_idx_sorted, start=1)
        ]

    def persist(self, path: Path) -> None:
        if self._embeddings is None or self._metadata is None:
            raise RagError(code="index_not_built", message="persist() called before build()")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "embeddings.npz", embeddings=self._embeddings)
        sidecar = {
            "metadata": asdict(self._metadata),
            "chunks": [asdict(c) for c in self._chunks],
        }
        (path / "index.json").write_text(json.dumps(sidecar, indent=2))

    @classmethod
    def load(cls, path: Path) -> InMemoryVectorIndex:
        sidecar_path = path / "index.json"
        emb_path = path / "embeddings.npz"
        if not sidecar_path.exists() or not emb_path.exists():
            raise RagError(
                code="index_not_built",
                message=f"index files missing at {path}",
                details={"path": str(path)},
            )
        sidecar = json.loads(sidecar_path.read_text())
        metadata = IndexMetadata(**sidecar["metadata"])
        chunks = [Chunk(**c) for c in sidecar["chunks"]]
        with np.load(emb_path) as data:
            embeddings = data["embeddings"].astype(np.float32)
        idx = cls()
        idx._chunks = chunks
        idx._embeddings = embeddings
        idx._metadata = metadata
        return idx
