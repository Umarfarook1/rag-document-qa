"""Domain types for the RAG pipeline.

All types are frozen dataclasses. Equality and hashability are by-value, so they
work cleanly as dict keys, set members, and inputs to multiset comparators in the
eval harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A single source document (one PDF, one Markdown file, one 10-K filing)."""

    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of a Document, ready to be embedded and indexed.

    `chunk_index` is the chunk's position within its parent document (0-based).
    `char_start` / `char_end` are character offsets into the parent's `text`,
    used for citation back to the original source.
    """

    id: str
    doc_id: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk surfaced by the retriever, with its score and rank in the result list.

    `score` is whatever similarity metric the index uses (cosine for dense, BM25
    for sparse). Higher is more similar by convention. `rank` is 1-based.
    """

    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class Question:
    """A user question fed into the retriever / answer generator."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GoldPair:
    """A gold (question, evidence) pair for retrieval evaluation.

    `gold_passages` is a list because some benchmarks (HotpotQA, MultiHop-RAG)
    have multiple supporting passages per question. `gold_doc_id` lets us also
    score doc-level retrieval (was the right document found at all).
    """

    question: Question
    gold_passages: list[str]
    gold_doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    """Stamped into every persisted vector index.

    Used at query time to detect embedder mismatch (the most common silent
    bug in RAG codebases: ingest with model A, query with model B).
    """

    embedder_name: str
    embedder_version: str
    embedding_dim: int
    chunk_count: int
    index_kind: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """The final response returned to the caller of the ask pipeline.

    `confident=False` means the system explicitly declined to answer; `text`
    will then carry the "I don't know" message and `no_answer_reason` will name
    which gate triggered (e.g. `top1_below_cosine_threshold`,
    `model_self_rated_below_threshold`).
    """

    text: str
    cited_chunk_ids: list[str]
    retrieved: list[RetrievedChunk]
    confident: bool
    confidence_score: float
    no_answer_reason: str | None
