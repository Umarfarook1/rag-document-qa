from rag_document_qa.types import (
    AnswerResult,
    Chunk,
    Document,
    GoldPair,
    IndexMetadata,
    Question,
    RetrievedChunk,
)


def test_document_minimal() -> None:
    d = Document(id="doc1", source="path/to/file.pdf", text="hello world", metadata={})
    assert d.id == "doc1"
    assert d.source == "path/to/file.pdf"
    assert d.text == "hello world"
    assert d.metadata == {}


def test_document_with_metadata() -> None:
    d = Document(
        id="aapl-10k-2024",
        source="https://www.sec.gov/Archives/edgar/data/320193/...",
        text="...",
        metadata={"ticker": "AAPL", "fiscal_year": "2024", "filing_type": "10-K"},
    )
    assert d.metadata["ticker"] == "AAPL"


def test_chunk_minimal() -> None:
    c = Chunk(
        id="chunk_3",
        doc_id="doc1",
        text="this is a chunk",
        chunk_index=3,
        char_start=120,
        char_end=170,
        metadata={},
    )
    assert c.id == "chunk_3"
    assert c.chunk_index == 3
    assert c.char_end - c.char_start == 50


def test_retrieved_chunk_carries_score() -> None:
    base = Chunk(id="c", doc_id="d", text="x", chunk_index=0, char_start=0, char_end=1, metadata={})
    r = RetrievedChunk(chunk=base, score=0.91, rank=1)
    assert r.score == 0.91
    assert r.rank == 1
    assert r.chunk is base


def test_question_minimal() -> None:
    q = Question(id="q42", text="What were Apple's R&D expenses in FY24?", metadata={})
    assert q.id == "q42"
    assert "Apple" in q.text


def test_gold_pair_minimal() -> None:
    g = GoldPair(
        question=Question(id="q1", text="?", metadata={}),
        gold_passages=["page 47 paragraph 3 text"],
        gold_doc_id="aapl-10k-2024",
        metadata={},
    )
    assert len(g.gold_passages) == 1
    assert g.gold_doc_id == "aapl-10k-2024"


def test_gold_pair_supports_multiple_passages() -> None:
    g = GoldPair(
        question=Question(id="q1", text="?", metadata={}),
        gold_passages=["passage A", "passage B", "passage C"],
        gold_doc_id="doc1",
        metadata={},
    )
    assert len(g.gold_passages) == 3


def test_index_metadata_records_embedder_identity() -> None:
    m = IndexMetadata(
        embedder_name="BAAI/bge-small-en-v1.5",
        embedder_version="1.5",
        embedding_dim=384,
        chunk_count=12_000,
        index_kind="faiss-flat-ip",
    )
    assert m.embedder_name.startswith("BAAI")
    assert m.embedding_dim == 384
    assert m.chunk_count == 12_000


def test_answer_result_confident() -> None:
    chunk = Chunk(
        id="c", doc_id="d", text="x", chunk_index=0, char_start=0, char_end=1, metadata={}
    )
    ar = AnswerResult(
        text="The R&D expense was $X.",
        cited_chunk_ids=["c"],
        retrieved=[RetrievedChunk(chunk=chunk, score=0.9, rank=1)],
        confident=True,
        confidence_score=0.92,
        no_answer_reason=None,
    )
    assert ar.confident is True
    assert ar.no_answer_reason is None
    assert ar.cited_chunk_ids == ["c"]


def test_answer_result_unconfident_carries_reason() -> None:
    chunk = Chunk(
        id="c", doc_id="d", text="x", chunk_index=0, char_start=0, char_end=1, metadata={}
    )
    ar = AnswerResult(
        text="I couldn't find a confident answer.",
        cited_chunk_ids=[],
        retrieved=[RetrievedChunk(chunk=chunk, score=0.3, rank=1)],
        confident=False,
        confidence_score=0.3,
        no_answer_reason="top1_below_cosine_threshold",
    )
    assert ar.confident is False
    assert ar.no_answer_reason == "top1_below_cosine_threshold"
    assert ar.cited_chunk_ids == []


def test_dataclasses_are_frozen() -> None:
    """Domain types must be frozen so they can be used as dict keys / hashed safely."""
    import dataclasses

    d = Document(id="d", source="s", text="t", metadata={})
    try:
        d.id = "other"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Document should be frozen")
