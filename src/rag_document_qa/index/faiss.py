"""FAISS-backed vector index (CPU).

Uses an `IndexFlatIP` (inner product) over L2-normalised vectors, which is
mathematically identical to cosine similarity. For corpora up to ~1M chunks
this is fast enough on CPU and exact (no recall-vs-speed tradeoff). For larger
corpora, swap to `IndexHNSWFlat` or `IndexIVFFlat` behind the same Protocol.

Persistence writes:
  - `embeddings.faiss`  -- the FAISS index, via faiss.write_index
  - `index.json`        -- chunks + IndexMetadata sidecar (mirrors InMemoryVectorIndex)

Load fails fast with `RagError(index_not_built)` if either file is missing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rag_document_qa.errors import RagError
from rag_document_qa.types import Chunk, IndexMetadata, RetrievedChunk


class FaissVectorIndex:
    """Faiss Vector Index."""
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._metadata: IndexMetadata | None = None
        self._index: Any = None
        self._faiss: Any = self._import_faiss()

    @staticmethod
    def _import_faiss() -> Any:
        try:
            import faiss
        except ImportError as e:
            raise RagError(
                code="loader_failed",
                message="faiss-cpu not installed; run `pip install rag-document-qa[faiss]`",
            ) from e
        return faiss

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
        # FAISS expects float32, contiguous, normalised for IP-as-cosine.
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._faiss.normalize_L2(emb)
        index = self._faiss.IndexFlatIP(metadata.embedding_dim)
        index.add(emb)
        self._index = index
        self._chunks = list(chunks)
        self._metadata = metadata

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int,
    ) -> list[RetrievedChunk]:
        if self._index is None or self._metadata is None:
            raise RagError(code="index_not_built", message="search() called before build()")
        q = (
            query_embedding.reshape(1, -1)
            if query_embedding.ndim == 1
            else np.ascontiguousarray(query_embedding, dtype=np.float32)
        )
        if q.shape[1] != self._metadata.embedding_dim:
            raise RagError(
                code="embedding_dim_mismatch",
                message=(f"query dim {q.shape[1]} != index dim {self._metadata.embedding_dim}"),
            )
        if k <= 0:
            return []
        q = np.ascontiguousarray(q, dtype=np.float32)
        self._faiss.normalize_L2(q)
        k_clamped = min(k, len(self._chunks))
        scores, idx = self._index.search(q, k_clamped)
        out: list[RetrievedChunk] = []
        for rank, (score, i) in enumerate(zip(scores[0], idx[0], strict=True), start=1):
            if i < 0:
                continue
            out.append(RetrievedChunk(chunk=self._chunks[int(i)], score=float(score), rank=rank))
        return out

    def persist(self, path: Path) -> None:
        if self._index is None or self._metadata is None:
            raise RagError(code="index_not_built", message="persist() called before build()")
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / "embeddings.faiss"))
        sidecar = {
            "metadata": asdict(self._metadata),
            "chunks": [asdict(c) for c in self._chunks],
        }
        (path / "index.json").write_text(json.dumps(sidecar, indent=2))

    @classmethod
    def load(cls, path: Path) -> FaissVectorIndex:
        sidecar_path = path / "index.json"
        index_path = path / "embeddings.faiss"
        if not sidecar_path.exists() or not index_path.exists():
            raise RagError(
                code="index_not_built",
                message=f"index files missing at {path}",
                details={"path": str(path)},
            )
        sidecar = json.loads(sidecar_path.read_text())
        metadata = IndexMetadata(**sidecar["metadata"])
        chunks = [Chunk(**c) for c in sidecar["chunks"]]
        instance = cls()
        instance._index = instance._faiss.read_index(str(index_path))
        instance._chunks = chunks
        instance._metadata = metadata
        return instance
