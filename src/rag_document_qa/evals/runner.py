"""Eval runner orchestrator.

For each gold pair: run the retriever, compute per-query metrics, accumulate.
Reports aggregate Recall@1, Recall@5, Recall@10, MRR, nDCG@10 plus per-query
detail rows so failures can be slice-and-diced offline.

The runner is independent of which retriever you plug in; FAISS vs InMemory vs
a future BM25 hybrid all flow through the same harness, which is the whole
point of the Protocol seam.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_document_qa.errors import RagError
from rag_document_qa.evals.metrics import (
    GoldMatcher,
    aggregate,
    ndcg_at_k,
    passage_overlap_match,
    recall_at_k,
    reciprocal_rank,
)
from rag_document_qa.types import GoldPair, RetrievedChunk

RetrieveFn = Callable[[str], list[RetrievedChunk]]


@dataclass(frozen=True, slots=True)
class PerQueryResult:
    """Per Query Result."""
    question_id: str
    nl: str
    gold_doc_id: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    top_chunk_id: str | None
    top_score: float
    latency_ms: int
    error: str | None


@dataclass(frozen=True, slots=True)
class EvalReport:
    total: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    avg_latency_ms: int
    errors: int
    per_query: list[PerQueryResult] = field(default_factory=list)


def run_retrieval_eval(
    gold_pairs: list[GoldPair],
    retrieve_fn: RetrieveFn,
    matcher: GoldMatcher = passage_overlap_match,
    limit: int | None = None,
) -> EvalReport:
    """Run the retrieval eval over every gold pair, return an EvalReport.

    `retrieve_fn` is any callable that maps query text to a list of retrieved
    chunks. Pass `Retriever.retrieve` directly; tests pass a mock.
    """
    if not gold_pairs:
        raise RagError(
            code="invalid_corpus",
            message="run_retrieval_eval() requires at least one gold pair",
        )
    work = gold_pairs[:limit] if limit is not None else gold_pairs

    rows: list[PerQueryResult] = []
    errors = 0

    for pair in work:
        start = time.perf_counter()
        retrieved: list[RetrievedChunk] = []
        err: str | None = None
        try:
            retrieved = retrieve_fn(pair.question.text)
        except RagError as e:
            err = f"{e.code}: {e.message}"
            errors += 1
        except Exception as e:  # pragma: no cover - defensive
            err = f"runner_error: {type(e).__name__}: {e}"
            errors += 1
        latency_ms = int((time.perf_counter() - start) * 1000)

        r1 = recall_at_k(retrieved, pair, 1, matcher) if not err else 0.0
        r5 = recall_at_k(retrieved, pair, 5, matcher) if not err else 0.0
        r10 = recall_at_k(retrieved, pair, 10, matcher) if not err else 0.0
        rr = reciprocal_rank(retrieved, pair, matcher) if not err else 0.0
        ndcg = ndcg_at_k(retrieved, pair, 10, matcher) if not err else 0.0

        rows.append(
            PerQueryResult(
                question_id=pair.question.id,
                nl=pair.question.text,
                gold_doc_id=pair.gold_doc_id,
                recall_at_1=r1,
                recall_at_5=r5,
                recall_at_10=r10,
                mrr=rr,
                ndcg_at_10=ndcg,
                top_chunk_id=retrieved[0].chunk.id if retrieved else None,
                top_score=retrieved[0].score if retrieved else 0.0,
                latency_ms=latency_ms,
                error=err,
            )
        )

    return EvalReport(
        total=len(rows),
        recall_at_1=aggregate([r.recall_at_1 for r in rows]),
        recall_at_5=aggregate([r.recall_at_5 for r in rows]),
        recall_at_10=aggregate([r.recall_at_10 for r in rows]),
        mrr=aggregate([r.mrr for r in rows]),
        ndcg_at_10=aggregate([r.ndcg_at_10 for r in rows]),
        avg_latency_ms=int(aggregate([float(r.latency_ms) for r in rows])),
        errors=errors,
        per_query=rows,
    )
