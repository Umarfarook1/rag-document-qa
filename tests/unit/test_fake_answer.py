import pytest

from rag_document_qa.answer.fake import FakeAnswerGenerator
from rag_document_qa.errors import RagError
from rag_document_qa.protocols import AnswerGenerator
from rag_document_qa.types import Chunk, Question, RetrievedChunk


def _mk_retrieved(text: str, score: float, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id="c1",
            doc_id="d",
            text=text,
            chunk_index=0,
            char_start=0,
            char_end=len(text),
            metadata={},
        ),
        score=score,
        rank=rank,
    )


def test_implements_protocol() -> None:
    assert isinstance(FakeAnswerGenerator(), AnswerGenerator)


def test_echo_first_returns_top_chunk_with_citation() -> None:
    g = FakeAnswerGenerator(mode="echo_first")
    q = Question(id="q", text="?", metadata={})
    r = [_mk_retrieved("the answer is 42", 0.85)]
    out = g.generate(q, r)
    assert out.confident is True
    assert out.cited_chunk_ids == ["c1"]
    assert "the answer is 42" in out.text
    assert out.confidence_score == 0.85


def test_echo_first_no_retrieval_falls_back_to_no_answer() -> None:
    g = FakeAnswerGenerator(mode="echo_first")
    q = Question(id="q", text="?", metadata={})
    out = g.generate(q, [])
    assert out.confident is False
    assert out.no_answer_reason == "no_chunks_retrieved"


def test_no_citation_mode() -> None:
    g = FakeAnswerGenerator(mode="no_citation")
    q = Question(id="q", text="?", metadata={})
    r = [_mk_retrieved("ctx", 0.9)]
    out = g.generate(q, r)
    assert out.cited_chunk_ids == []
    assert out.confident is True


def test_low_confidence_mode() -> None:
    g = FakeAnswerGenerator(mode="low_confidence")
    q = Question(id="q", text="?", metadata={})
    r = [_mk_retrieved("ctx", 0.9)]
    out = g.generate(q, r)
    assert out.confident is False
    assert out.no_answer_reason == "model_self_rated_below_threshold"


def test_raises_mode() -> None:
    g = FakeAnswerGenerator(mode="raises")
    q = Question(id="q", text="?", metadata={})
    with pytest.raises(RagError) as info:
        g.generate(q, [])
    assert info.value.code == "answer_generation_failed"
