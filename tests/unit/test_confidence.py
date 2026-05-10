from rag_document_qa.confidence import (
    ConfidenceConfig,
    evaluate_confidence,
)
from rag_document_qa.types import Chunk, RetrievedChunk


def _r(score: float, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=f"c{rank}",
            doc_id="d",
            text="x",
            chunk_index=rank - 1,
            char_start=0,
            char_end=1,
            metadata={},
        ),
        score=score,
        rank=rank,
    )


def test_no_retrieval_is_unconfident() -> None:
    v = evaluate_confidence([], self_rated=9.0)
    assert v.confident is False
    assert v.reason == "no_chunks_retrieved"


def test_top1_below_threshold_is_unconfident() -> None:
    v = evaluate_confidence([_r(0.3)], self_rated=9.0)
    assert v.confident is False
    assert v.reason == "top1_below_cosine_threshold"


def test_self_rated_below_threshold_is_unconfident() -> None:
    v = evaluate_confidence([_r(0.9)], self_rated=4.0)
    assert v.confident is False
    assert v.reason == "model_self_rated_below_threshold"


def test_self_rated_missing_is_unconfident() -> None:
    v = evaluate_confidence([_r(0.9)], self_rated=None)
    assert v.confident is False
    assert v.reason == "model_self_rated_below_threshold"


def test_both_gates_pass() -> None:
    v = evaluate_confidence([_r(0.9)], self_rated=8.0)
    assert v.confident is True
    assert v.reason is None
    # Combined score: (0.9 + 0.8) / 2 = 0.85
    assert abs(v.score - 0.85) < 1e-6


def test_thresholds_are_configurable() -> None:
    cfg = ConfidenceConfig(cosine_threshold=0.95, self_rated_threshold=8.5)
    v = evaluate_confidence([_r(0.9)], self_rated=9.0, config=cfg)
    assert v.confident is False  # cosine 0.9 < 0.95
    assert v.reason == "top1_below_cosine_threshold"
