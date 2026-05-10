import pytest

from rag_document_qa.embed.fake import FakeEmbedder
from rag_document_qa.errors import RagError
from rag_document_qa.index.memory import InMemoryVectorIndex
from rag_document_qa.rerank import IdentityReranker, ReverseReranker
from rag_document_qa.retriever import Retriever, RetrieverConfig
from rag_document_qa.types import Chunk, IndexMetadata


def _build_index(
    embedder: FakeEmbedder, texts: list[str]
) -> tuple[InMemoryVectorIndex, list[Chunk]]:
    chunks = [
        Chunk(
            id=f"c{i}",
            doc_id="d",
            text=t,
            chunk_index=i,
            char_start=0,
            char_end=len(t),
            metadata={},
        )
        for i, t in enumerate(texts)
    ]
    embeddings = embedder.encode(texts)
    md = IndexMetadata(
        embedder_name=embedder.name,
        embedder_version=embedder.version,
        embedding_dim=embedder.dim,
        chunk_count=len(chunks),
        index_kind="memory",
    )
    idx = InMemoryVectorIndex()
    idx.build(chunks, embeddings, md)
    return idx, chunks


def test_retriever_returns_top_k_return_results() -> None:
    embedder = FakeEmbedder(dim=16)
    idx, _ = _build_index(embedder, [f"text {i}" for i in range(10)])
    r = Retriever(
        embedder=embedder,
        index=idx,
        config=RetrieverConfig(top_k_retrieve=10, top_k_return=3),
    )
    out = r.retrieve("query")
    assert len(out) == 3
    # Ranks should be 1-based and dense for the returned slice.
    assert [c.rank for c in out] == [1, 2, 3]


def test_retriever_query_finds_exact_match() -> None:
    embedder = FakeEmbedder(dim=16)
    idx, _ = _build_index(embedder, ["alpha beta", "gamma delta", "epsilon zeta"])
    r = Retriever(embedder=embedder, index=idx)
    out = r.retrieve("gamma delta")
    assert out[0].chunk.id == "c1"


def test_retriever_rejects_embedder_mismatch() -> None:
    embedder = FakeEmbedder(dim=16)
    idx, _ = _build_index(embedder, ["one"])

    class OtherEmbedder(FakeEmbedder):
        @property
        def name(self) -> str:
            return "other-embedder"

    with pytest.raises(RagError) as info:
        Retriever(embedder=OtherEmbedder(dim=16), index=idx)
    assert info.value.code == "embedder_mismatch"


def test_retriever_rejects_dim_mismatch_via_embedder_swap() -> None:
    embedder = FakeEmbedder(dim=16)
    idx, _ = _build_index(embedder, ["one"])

    class WrongDim(FakeEmbedder):
        @property
        def name(self) -> str:
            return embedder.name  # match name, mismatch dim

        @property
        def dim(self) -> int:
            return 32

    with pytest.raises(RagError) as info:
        Retriever(embedder=WrongDim(dim=32), index=idx)
    assert info.value.code == "embedding_dim_mismatch"


def test_retriever_with_identity_reranker_unchanged() -> None:
    embedder = FakeEmbedder(dim=16)
    idx, _ = _build_index(embedder, [f"t{i}" for i in range(6)])
    r_no = Retriever(embedder=embedder, index=idx, reranker=None)
    r_id = Retriever(embedder=embedder, index=idx, reranker=IdentityReranker())
    out_no = r_no.retrieve("query")
    out_id = r_id.retrieve("query")
    assert [c.chunk.id for c in out_no] == [c.chunk.id for c in out_id]


def test_retriever_with_reverse_reranker_inverts_order() -> None:
    embedder = FakeEmbedder(dim=16)
    idx, _ = _build_index(embedder, [f"t{i}" for i in range(6)])
    r_no = Retriever(
        embedder=embedder,
        index=idx,
        config=RetrieverConfig(top_k_retrieve=4, top_k_return=4),
    )
    r_rev = Retriever(
        embedder=embedder,
        index=idx,
        reranker=ReverseReranker(),
        config=RetrieverConfig(top_k_retrieve=4, top_k_return=4),
    )
    out_no = r_no.retrieve("query")
    out_rev = r_rev.retrieve("query")
    assert [c.chunk.id for c in out_rev] == list(reversed([c.chunk.id for c in out_no]))
