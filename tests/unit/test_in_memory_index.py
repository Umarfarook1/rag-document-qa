from pathlib import Path

import numpy as np
import pytest

from rag_document_qa.embed.fake import FakeEmbedder
from rag_document_qa.errors import RagError
from rag_document_qa.index.memory import InMemoryVectorIndex
from rag_document_qa.protocols import VectorIndex
from rag_document_qa.types import Chunk, IndexMetadata


def _mk_chunk(i: int, text: str) -> Chunk:
    return Chunk(
        id=f"c{i}",
        doc_id="d",
        text=text,
        chunk_index=i,
        char_start=0,
        char_end=len(text),
        metadata={},
    )


def test_implements_protocol() -> None:
    assert isinstance(InMemoryVectorIndex(), VectorIndex)


def test_search_before_build_raises() -> None:
    idx = InMemoryVectorIndex()
    with pytest.raises(RagError) as info:
        idx.search(np.zeros(8, dtype=np.float32), k=3)
    assert info.value.code == "index_not_built"


def test_metadata_before_build_raises() -> None:
    idx = InMemoryVectorIndex()
    with pytest.raises(RagError) as info:
        _ = idx.metadata
    assert info.value.code == "index_not_built"


def test_build_and_search() -> None:
    e = FakeEmbedder(dim=16)
    chunks = [_mk_chunk(i, f"chunk text {i}") for i in range(5)]
    embeddings = e.encode([c.text for c in chunks])
    md = IndexMetadata(
        embedder_name=e.name,
        embedder_version=e.version,
        embedding_dim=e.dim,
        chunk_count=len(chunks),
        index_kind="memory",
    )
    idx = InMemoryVectorIndex()
    idx.build(chunks, embeddings, md)
    # Querying with one of the original texts should put it at rank 1.
    q = e.encode(["chunk text 2"])
    hits = idx.search(q, k=3)
    assert len(hits) == 3
    assert hits[0].chunk.id == "c2"
    assert hits[0].rank == 1
    assert hits[0].score > 0.9
    # Ranks are 1-based and dense
    assert [h.rank for h in hits] == [1, 2, 3]


def test_build_rejects_empty_chunks() -> None:
    idx = InMemoryVectorIndex()
    md = IndexMetadata(
        embedder_name="fake",
        embedder_version="1",
        embedding_dim=4,
        chunk_count=0,
        index_kind="memory",
    )
    with pytest.raises(RagError) as info:
        idx.build([], np.zeros((0, 4), dtype=np.float32), md)
    assert info.value.code == "corpus_empty"


def test_build_rejects_count_mismatch() -> None:
    idx = InMemoryVectorIndex()
    chunks = [_mk_chunk(0, "x")]
    md = IndexMetadata(
        embedder_name="fake",
        embedder_version="1",
        embedding_dim=4,
        chunk_count=1,
        index_kind="memory",
    )
    with pytest.raises(RagError) as info:
        idx.build(chunks, np.zeros((2, 4), dtype=np.float32), md)
    assert info.value.code == "invalid_corpus"


def test_build_rejects_dim_mismatch() -> None:
    idx = InMemoryVectorIndex()
    chunks = [_mk_chunk(0, "x")]
    md = IndexMetadata(
        embedder_name="fake",
        embedder_version="1",
        embedding_dim=4,
        chunk_count=1,
        index_kind="memory",
    )
    with pytest.raises(RagError) as info:
        idx.build(chunks, np.zeros((1, 8), dtype=np.float32), md)
    assert info.value.code == "embedding_dim_mismatch"


def test_search_rejects_dim_mismatch() -> None:
    e = FakeEmbedder(dim=16)
    chunks = [_mk_chunk(0, "x")]
    embeddings = e.encode([c.text for c in chunks])
    md = IndexMetadata(
        embedder_name=e.name,
        embedder_version=e.version,
        embedding_dim=e.dim,
        chunk_count=1,
        index_kind="memory",
    )
    idx = InMemoryVectorIndex()
    idx.build(chunks, embeddings, md)
    with pytest.raises(RagError) as info:
        idx.search(np.zeros(8, dtype=np.float32), k=1)
    assert info.value.code == "embedding_dim_mismatch"


def test_search_k_zero_returns_empty() -> None:
    e = FakeEmbedder(dim=8)
    chunks = [_mk_chunk(i, f"t{i}") for i in range(3)]
    md = IndexMetadata(
        embedder_name=e.name,
        embedder_version=e.version,
        embedding_dim=e.dim,
        chunk_count=3,
        index_kind="memory",
    )
    idx = InMemoryVectorIndex()
    idx.build(chunks, e.encode([c.text for c in chunks]), md)
    assert idx.search(e.encode(["q"]), k=0) == []


def test_search_k_larger_than_corpus_clamps() -> None:
    e = FakeEmbedder(dim=8)
    chunks = [_mk_chunk(i, f"t{i}") for i in range(3)]
    md = IndexMetadata(
        embedder_name=e.name,
        embedder_version=e.version,
        embedding_dim=e.dim,
        chunk_count=3,
        index_kind="memory",
    )
    idx = InMemoryVectorIndex()
    idx.build(chunks, e.encode([c.text for c in chunks]), md)
    hits = idx.search(e.encode(["q"]), k=99)
    assert len(hits) == 3


def test_persist_and_load_round_trip(tmp_path: Path) -> None:
    e = FakeEmbedder(dim=16)
    chunks = [_mk_chunk(i, f"chunk {i}") for i in range(4)]
    md = IndexMetadata(
        embedder_name=e.name,
        embedder_version=e.version,
        embedding_dim=e.dim,
        chunk_count=4,
        index_kind="memory",
    )
    idx = InMemoryVectorIndex()
    idx.build(chunks, e.encode([c.text for c in chunks]), md)
    idx.persist(tmp_path / "idx")

    reloaded = InMemoryVectorIndex.load(tmp_path / "idx")
    assert reloaded.metadata.embedder_name == e.name
    assert reloaded.metadata.chunk_count == 4
    hits = reloaded.search(e.encode(["chunk 2"]), k=2)
    assert hits[0].chunk.id == "c2"


def test_load_missing_path_raises() -> None:
    with pytest.raises(RagError) as info:
        InMemoryVectorIndex.load(Path("nonexistent_path"))
    assert info.value.code == "index_not_built"
