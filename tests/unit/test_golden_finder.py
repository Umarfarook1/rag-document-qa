import json
from pathlib import Path

import pytest

from rag_document_qa.errors import RagError
from rag_document_qa.evals.golden_finder import load_finder_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_loads_query_evidence_naming(tmp_path: Path) -> None:
    p = tmp_path / "finder.jsonl"
    _write_jsonl(
        p,
        [
            {
                "query_id": "q1",
                "query": "What is X?",
                "evidence": ["passage A", "passage B"],
                "answer": "X is Y.",
                "source": {"ticker": "AAPL", "accession": "0001-2-3"},
            }
        ],
    )
    pairs = load_finder_jsonl(p)
    assert len(pairs) == 1
    assert pairs[0].question.id == "q1"
    assert pairs[0].question.text == "What is X?"
    assert pairs[0].gold_passages == ["passage A", "passage B"]
    # Doc id format: "{ticker_lower}-10k-{accession_with_dashes_stripped}"
    assert pairs[0].gold_doc_id == "aapl-10k-000123"
    # answer flows through metadata
    assert pairs[0].metadata["answer"] == "X is Y."


def test_loads_question_gold_passages_naming(tmp_path: Path) -> None:
    p = tmp_path / "finder.jsonl"
    _write_jsonl(
        p,
        [
            {
                "id": "q42",
                "question": "When did Y?",
                "gold_passages": ["evidence text"],
                "doc_id": "msft-10k-2023",
            }
        ],
    )
    pairs = load_finder_jsonl(p)
    assert pairs[0].question.id == "q42"
    assert pairs[0].gold_doc_id == "msft-10k-2023"


def test_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "finder.jsonl"
    p.write_text(
        '{"query": "Q1", "evidence": ["e"]}\n\n{"query": "Q2", "evidence": ["e"]}\n',
        encoding="utf-8",
    )
    assert len(load_finder_jsonl(p)) == 2


def test_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "finder.jsonl"
    p.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(RagError) as info:
        load_finder_jsonl(p)
    assert info.value.code == "invalid_corpus"


def test_missing_query_raises(tmp_path: Path) -> None:
    p = tmp_path / "finder.jsonl"
    _write_jsonl(p, [{"evidence": ["x"]}])
    with pytest.raises(RagError) as info:
        load_finder_jsonl(p)
    assert info.value.code == "invalid_corpus"


def test_evidence_not_list_raises(tmp_path: Path) -> None:
    p = tmp_path / "finder.jsonl"
    _write_jsonl(p, [{"query": "q", "evidence": "not a list"}])
    with pytest.raises(RagError) as info:
        load_finder_jsonl(p)
    assert info.value.code == "invalid_corpus"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RagError) as info:
        load_finder_jsonl(tmp_path / "nonexistent.jsonl")
    assert info.value.code == "invalid_corpus"


def test_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(RagError) as info:
        load_finder_jsonl(p)
    assert info.value.code == "corpus_empty"


def test_limit_truncates(tmp_path: Path) -> None:
    p = tmp_path / "finder.jsonl"
    _write_jsonl(
        p,
        [{"query": f"q{i}", "evidence": ["x"]} for i in range(10)],
    )
    pairs = load_finder_jsonl(p, limit=3)
    assert len(pairs) == 3
