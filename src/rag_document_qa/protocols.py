"""PEP 544 Protocols defining the seams between RAG pipeline components.

Every external dependency (embedding model, vector store, reranker, LLM, doc
loader) is consumed via a Protocol so the pipeline can be exercised end-to-end
in tests with in-memory fakes.

`@runtime_checkable` lets `isinstance(x, Protocol)` work in tests, but the real
guarantee is structural: anything that satisfies the method signatures is a
valid impl, no inheritance required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rag_document_qa.types import (
    AnswerResult,
    Chunk,
    Document,
    IndexMetadata,
    Question,
    RetrievedChunk,
)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors.

    `encode` accepts a batch and returns a 2D float32 array of shape (n, dim).
    The same vector space MUST be used for ingest and query, so impls expose
    `name` + `version` + `dim` for the index to stamp into its metadata.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Stores chunk embeddings + supports nearest-neighbour search.

    Build is a one-shot operation; reusing an index for multiple builds is
    undefined. Impls MUST stamp `IndexMetadata` so a query-time embedder
    mismatch can be detected before the user gets garbage results.
    """

    @property
    def metadata(self) -> IndexMetadata: ...

    def build(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32],
        metadata: IndexMetadata,
    ) -> None: ...

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int,
    ) -> list[RetrievedChunk]: ...

    def persist(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> VectorIndex: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores a list of retrieved chunks against the original query.

    Returned list is the same length as the input, sorted by new score desc,
    with `rank` rewritten 1..n.
    """

    @property
    def name(self) -> str: ...

    def rerank(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """Produces a final answer from a question + retrieved context.

    Impls own the prompt template, citation enforcement, and the confidence
    gate. The pipeline is otherwise dumb: it hands over the chunks and trusts
    the generator to produce an `AnswerResult` with the right shape.
    """

    @property
    def name(self) -> str: ...

    def generate(
        self,
        question: Question,
        retrieved: list[RetrievedChunk],
    ) -> AnswerResult: ...


@runtime_checkable
class DocLoader(Protocol):
    """Reads a source path (file or URL) into one or more Documents.

    A loader can return zero documents if the source is empty or filtered out;
    it should NOT raise for that case. It SHOULD raise `RagError(loader_failed)`
    for unrecoverable parse failures.
    """

    @property
    def name(self) -> str: ...

    def load(self, source: str) -> list[Document]: ...
