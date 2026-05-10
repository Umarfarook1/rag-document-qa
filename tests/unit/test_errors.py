import pytest

from rag_document_qa.errors import (
    ErrorCode,
    RagError,
)


def test_rag_error_minimal() -> None:
    err = RagError(code="corpus_empty", message="no documents loaded")
    assert err.code == "corpus_empty"
    assert err.message == "no documents loaded"
    assert err.details == {}


def test_rag_error_with_details() -> None:
    err = RagError(
        code="embedder_mismatch",
        message="index built with bge-small but query embedder is openai",
        details={"index_embedder": "bge-small", "query_embedder": "openai"},
    )
    assert err.details["index_embedder"] == "bge-small"


def test_rag_error_to_dict_shape() -> None:
    err = RagError(code="invalid_corpus", message="bad", details={"path": "/x"})
    d = err.to_dict()
    assert d == {"error": "invalid_corpus", "message": "bad", "details": {"path": "/x"}}


def test_rag_error_to_dict_omits_empty_details() -> None:
    err = RagError(code="unknown", message="something")
    d = err.to_dict()
    assert d == {"error": "unknown", "message": "something"}
    assert "details" not in d


def test_rag_error_is_an_exception() -> None:
    """RagError is raisable and catchable as a normal exception."""
    with pytest.raises(RagError) as info:
        raise RagError(code="unknown", message="boom")
    assert info.value.code == "unknown"


def test_error_code_literal_includes_known_codes() -> None:
    """Spot-check that the Literal type spans the codes we depend on elsewhere.

    These codes are part of the public contract for any agent or downstream caller
    catching RagError and switching on err.code, so they must stay stable.
    """
    expected = {
        "corpus_empty",
        "index_not_built",
        "embedding_dim_mismatch",
        "embedder_mismatch",
        "invalid_corpus",
        "loader_failed",
        "answer_generation_failed",
        "no_confident_answer",
        "rate_limited",
        "unknown",
    }
    # ErrorCode is a typing.Literal alias; the set of args is its second tuple element.
    import typing

    args = typing.get_args(ErrorCode)
    assert expected.issubset(set(args)), f"Missing codes: {expected - set(args)}"
