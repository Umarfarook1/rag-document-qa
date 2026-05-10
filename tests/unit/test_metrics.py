import math

from rag_document_qa.evals.metrics import (
    aggregate,
    doc_id_match,
    ndcg_at_k,
    passage_overlap_match,
    recall_at_k,
    reciprocal_rank,
)
from rag_document_qa.types import Chunk, GoldPair, Question, RetrievedChunk


def _retrieved(items: list[tuple[str, str, str]]) -> list[RetrievedChunk]:
    """items: [(chunk_id, doc_id, text), ...]; returns RetrievedChunks at ranks 1..n."""
    out: list[RetrievedChunk] = []
    for i, (cid, did, text) in enumerate(items):
        out.append(
            RetrievedChunk(
                chunk=Chunk(
                    id=cid,
                    doc_id=did,
                    text=text,
                    chunk_index=i,
                    char_start=0,
                    char_end=len(text),
                    metadata={},
                ),
                score=1.0 - 0.05 * i,
                rank=i + 1,
            )
        )
    return out


def _gold(text: str, doc_id: str = "d1") -> GoldPair:
    return GoldPair(
        question=Question(id="q", text="?", metadata={}),
        gold_passages=[text],
        gold_doc_id=doc_id,
        metadata={},
    )


def test_recall_at_1_hit() -> None:
    r = _retrieved([("c1", "d1", "the answer is 42")])
    assert recall_at_k(r, _gold("the answer is 42"), 1) == 1.0


def test_recall_at_1_miss_when_match_lower_in_list() -> None:
    r = _retrieved(
        [
            ("c1", "d1", "wrong context"),
            ("c2", "d1", "the answer is 42"),
        ]
    )
    assert recall_at_k(r, _gold("the answer is 42"), 1) == 0.0


def test_recall_at_5_finds_match_at_rank_3() -> None:
    r = _retrieved(
        [
            ("c1", "d1", "wrong"),
            ("c2", "d1", "wrong again"),
            ("c3", "d1", "the answer is 42"),
            ("c4", "d1", "more wrong"),
            ("c5", "d1", "even more wrong"),
        ]
    )
    assert recall_at_k(r, _gold("the answer is 42"), 5) == 1.0


def test_recall_filters_by_doc_id_when_provided() -> None:
    r = _retrieved(
        [
            ("c1", "d2", "the answer is 42"),  # right text, wrong doc
        ]
    )
    assert recall_at_k(r, _gold("the answer is 42", "d1"), 5) == 0.0


def test_recall_overlap_match_partial() -> None:
    """Soft overlap: chunk contains 50%+ of the gold passage."""
    r = _retrieved([("c1", "d1", "long preamble. the answer is 42 is the answer here.")])
    assert recall_at_k(r, _gold("the answer is 42"), 1) == 1.0


def test_recall_no_retrieval_returns_zero() -> None:
    assert recall_at_k([], _gold("anything"), 5) == 0.0


def test_recall_k_zero() -> None:
    r = _retrieved([("c1", "d1", "the answer is 42")])
    assert recall_at_k(r, _gold("the answer is 42"), 0) == 0.0


def test_reciprocal_rank_first_hit() -> None:
    r = _retrieved(
        [
            ("c1", "d1", "wrong"),
            ("c2", "d1", "the answer is 42"),
            ("c3", "d1", "the answer is 42"),  # later duplicate ignored
        ]
    )
    rr = reciprocal_rank(r, _gold("the answer is 42"))
    assert abs(rr - 0.5) < 1e-9


def test_reciprocal_rank_no_hit() -> None:
    r = _retrieved([("c1", "d1", "wrong")])
    assert reciprocal_rank(r, _gold("the answer")) == 0.0


def test_ndcg_at_k_first_position_is_one() -> None:
    r = _retrieved([("c1", "d1", "the answer is 42")])
    assert ndcg_at_k(r, _gold("the answer is 42"), 10) == 1.0


def test_ndcg_at_k_third_position() -> None:
    r = _retrieved(
        [
            ("c1", "d1", "wrong"),
            ("c2", "d1", "wrong"),
            ("c3", "d1", "the answer is 42"),
        ]
    )
    score = ndcg_at_k(r, _gold("the answer is 42"), 10)
    expected = 1.0 / math.log2(4)
    assert abs(score - expected) < 1e-9


def test_ndcg_no_hit() -> None:
    r = _retrieved([("c1", "d1", "wrong")])
    assert ndcg_at_k(r, _gold("never seen"), 10) == 0.0


def test_aggregate_mean() -> None:
    assert abs(aggregate([1.0, 0.0, 1.0, 0.0]) - 0.5) < 1e-9
    assert aggregate([]) == 0.0


def test_doc_id_match_loose_lookup() -> None:
    chunk = Chunk(
        id="c",
        doc_id="aapl-10k-2024",
        text="x",
        chunk_index=0,
        char_start=0,
        char_end=1,
        metadata={},
    )
    g = GoldPair(
        question=Question(id="q", text="?", metadata={}),
        gold_passages=["irrelevant"],
        gold_doc_id="aapl-10k-2024",
        metadata={},
    )
    assert doc_id_match(chunk, g) is True


def test_passage_overlap_returns_false_for_empty_inputs() -> None:
    chunk = Chunk(
        id="c",
        doc_id="d",
        text="",
        chunk_index=0,
        char_start=0,
        char_end=0,
        metadata={},
    )
    g = GoldPair(
        question=Question(id="q", text="?", metadata={}),
        gold_passages=[""],
        gold_doc_id="d",
        metadata={},
    )
    assert passage_overlap_match(chunk, g) is False
