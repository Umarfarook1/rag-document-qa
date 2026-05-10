import pytest

from rag_document_qa.chunking import (
    ChunkConfig,
    RecursiveTextSplitter,
    split_documents,
)
from rag_document_qa.errors import RagError
from rag_document_qa.types import Document


def test_chunk_config_validates() -> None:
    with pytest.raises(ValueError, match="positive"):
        RecursiveTextSplitter(ChunkConfig(chunk_size=0))
    with pytest.raises(ValueError, match="non-negative"):
        RecursiveTextSplitter(ChunkConfig(chunk_size=10, chunk_overlap=-1))
    with pytest.raises(ValueError, match="smaller"):
        RecursiveTextSplitter(ChunkConfig(chunk_size=10, chunk_overlap=10))


def test_split_empty_document_returns_empty() -> None:
    s = RecursiveTextSplitter()
    assert s.split(Document(id="d", source="s", text="", metadata={})) == []


def test_split_short_document_returns_one_chunk() -> None:
    s = RecursiveTextSplitter(ChunkConfig(chunk_size=1024))
    out = s.split(Document(id="d", source="s", text="short text", metadata={}))
    assert len(out) == 1
    assert out[0].text == "short text"
    assert out[0].chunk_index == 0
    assert out[0].doc_id == "d"
    assert out[0].id == "d::chunk_0"


def test_split_long_document_produces_multiple_chunks() -> None:
    s = RecursiveTextSplitter(ChunkConfig(chunk_size=50, chunk_overlap=10))
    text = "para1.\n\npara2 is a longer paragraph than the first.\n\npara3 caps it off."
    out = s.split(Document(id="d", source="s", text=text, metadata={}))
    assert len(out) >= 2
    for c in out:
        assert len(c.text) <= 70  # chunk_size + overlap headroom
    indices = [c.chunk_index for c in out]
    assert indices == sorted(indices)


def test_split_metadata_propagates_to_chunks() -> None:
    s = RecursiveTextSplitter(ChunkConfig(chunk_size=100, chunk_overlap=0))
    doc = Document(
        id="d",
        source="s",
        text="x" * 50,
        metadata={"ticker": "AAPL", "fiscal_year": "2024"},
    )
    out = s.split(doc)
    assert out[0].metadata["ticker"] == "AAPL"
    assert out[0].metadata["fiscal_year"] == "2024"


def test_split_chunk_offsets_index_back_into_source() -> None:
    """char_start/char_end let citations point at the exact original-text slice."""
    s = RecursiveTextSplitter(ChunkConfig(chunk_size=20, chunk_overlap=0))
    text = "abcdefghijklmnop12345678901234567890"
    out = s.split(Document(id="d", source="s", text=text, metadata={}))
    for c in out:
        # The chunk text must be a substring of the source text starting at char_start.
        assert text[c.char_start : c.char_end].startswith(c.text[:5])


def test_split_documents_rejects_empty_list() -> None:
    with pytest.raises(RagError) as info:
        split_documents([])
    assert info.value.code == "corpus_empty"


def test_split_documents_aggregates() -> None:
    docs = [
        Document(id="a", source="s", text="alpha text", metadata={}),
        Document(id="b", source="s", text="beta text", metadata={}),
    ]
    out = split_documents(docs, ChunkConfig(chunk_size=1024, chunk_overlap=0))
    doc_ids = {c.doc_id for c in out}
    assert doc_ids == {"a", "b"}
