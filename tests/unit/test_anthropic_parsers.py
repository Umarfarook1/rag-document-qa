"""Tests for the citation/self-rated parsers in the Anthropic answer generator.

We exercise the parsers directly (no SDK call needed) to keep these tests fast
and run-anywhere.
"""

from rag_document_qa.answer.anthropic import (
    _parse_citations,
    _parse_self_rated,
)
from rag_document_qa.types import Chunk, RetrievedChunk


def _retrieved(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                id=f"id_{i}",
                doc_id="d",
                text="x",
                chunk_index=i,
                char_start=0,
                char_end=1,
                metadata={},
            ),
            score=0.9 - 0.05 * i,
            rank=i + 1,
        )
        for i in range(n)
    ]


def test_parse_citations_basic() -> None:
    text = "The number was X [chunk_1] and Y [chunk_3]."
    ids = _parse_citations(text, _retrieved(5))
    assert ids == ["id_0", "id_2"]


def test_parse_citations_dedupes_repeated_refs() -> None:
    text = "Used [chunk_1] and again [chunk_1] elsewhere [chunk_2]."
    ids = _parse_citations(text, _retrieved(3))
    assert ids == ["id_0", "id_1"]


def test_parse_citations_ignores_out_of_range_refs() -> None:
    """Hallucinated [chunk_99] when only 3 were retrieved should be dropped."""
    text = "From [chunk_2] and the imaginary [chunk_99]."
    ids = _parse_citations(text, _retrieved(3))
    assert ids == ["id_1"]


def test_parse_citations_empty_when_none_present() -> None:
    text = "Plain answer with no references."
    assert _parse_citations(text, _retrieved(3)) == []


def test_parse_citations_case_insensitive() -> None:
    text = "Cite [Chunk_2]."
    ids = _parse_citations(text, _retrieved(3))
    assert ids == ["id_1"]


def test_parse_self_rated_takes_last_match() -> None:
    text = "This is the answer.\nConfidence: 8/10"
    assert _parse_self_rated(text) == 8.0


def test_parse_self_rated_with_decimal() -> None:
    text = "Confidence: 7.5"
    assert _parse_self_rated(text) == 7.5


def test_parse_self_rated_missing_returns_none() -> None:
    assert _parse_self_rated("nothing here") is None


def test_parse_self_rated_takes_last_when_multiple() -> None:
    text = "Confidence: 3 .. but actually Confidence: 9"
    assert _parse_self_rated(text) == 9.0
