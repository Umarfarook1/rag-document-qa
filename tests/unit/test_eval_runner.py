import pytest

from rag_document_qa.errors import RagError
from rag_document_qa.evals.runner import run_retrieval_eval
from rag_document_qa.types import Chunk, GoldPair, Question, RetrievedChunk


def _gold(qid: str, text: str, doc_id: str = "d1") -> GoldPair:
    return GoldPair(
        question=Question(id=qid, text=f"q-{qid}", metadata={}),
        gold_passages=[text],
        gold_doc_id=doc_id,
        metadata={},
    )


def _r(rank: int, text: str, doc_id: str = "d1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=f"c{rank}",
            doc_id=doc_id,
            text=text,
            chunk_index=rank - 1,
            char_start=0,
            char_end=len(text),
            metadata={},
        ),
        score=1.0 - 0.05 * rank,
        rank=rank,
    )


def test_perfect_retriever_yields_perfect_metrics() -> None:
    pairs = [_gold("q1", "alpha"), _gold("q2", "beta")]

    def perfect(q: str) -> list[RetrievedChunk]:
        if "q1" in q:
            return [_r(1, "alpha")]
        return [_r(1, "beta")]

    rep = run_retrieval_eval(pairs, perfect)
    assert rep.total == 2
    assert rep.recall_at_1 == 1.0
    assert rep.recall_at_5 == 1.0
    assert rep.mrr == 1.0
    assert rep.ndcg_at_10 == 1.0
    assert rep.errors == 0


def test_broken_retriever_yields_zero_metrics() -> None:
    pairs = [_gold("q1", "alpha")]

    def broken(q: str) -> list[RetrievedChunk]:
        return [_r(1, "totally unrelated")]

    rep = run_retrieval_eval(pairs, broken)
    assert rep.recall_at_1 == 0.0
    assert rep.mrr == 0.0


def test_retriever_match_at_rank_3_lowers_recall_at_1() -> None:
    pairs = [_gold("q1", "alpha")]

    def retriever(q: str) -> list[RetrievedChunk]:
        return [_r(1, "wrong"), _r(2, "wrong"), _r(3, "alpha")]

    rep = run_retrieval_eval(pairs, retriever)
    assert rep.recall_at_1 == 0.0
    assert rep.recall_at_5 == 1.0
    assert abs(rep.mrr - (1.0 / 3.0)) < 1e-9


def test_retriever_error_recorded_per_query() -> None:
    pairs = [_gold("q1", "alpha"), _gold("q2", "beta")]

    def flaky(q: str) -> list[RetrievedChunk]:
        if "q2" in q:
            raise RagError(code="rate_limited", message="quota")
        return [_r(1, "alpha")]

    rep = run_retrieval_eval(pairs, flaky)
    assert rep.errors == 1
    assert rep.per_query[1].error is not None
    assert "rate_limited" in rep.per_query[1].error
    # The first query still scored.
    assert rep.per_query[0].recall_at_1 == 1.0


def test_runner_rejects_empty_gold_set() -> None:
    with pytest.raises(RagError) as info:
        run_retrieval_eval([], lambda q: [])
    assert info.value.code == "invalid_corpus"


def test_runner_limit_truncates() -> None:
    pairs = [_gold(f"q{i}", "alpha") for i in range(10)]
    rep = run_retrieval_eval(pairs, lambda q: [_r(1, "alpha")], limit=3)
    assert rep.total == 3
    assert len(rep.per_query) == 3


def test_runner_records_top_chunk_metadata() -> None:
    pairs = [_gold("q1", "alpha")]
    rep = run_retrieval_eval(pairs, lambda q: [_r(1, "alpha")])
    assert rep.per_query[0].top_chunk_id == "c1"
    assert rep.per_query[0].top_score > 0
