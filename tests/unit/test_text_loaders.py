from pathlib import Path

import pytest

from rag_document_qa.errors import RagError
from rag_document_qa.loaders.text import MarkdownLoader, TextLoader
from rag_document_qa.protocols import DocLoader


def test_text_loader_implements_protocol() -> None:
    assert isinstance(TextLoader(), DocLoader)


def test_markdown_loader_implements_protocol() -> None:
    assert isinstance(MarkdownLoader(), DocLoader)


def test_text_loader_loads_file(tmp_path: Path) -> None:
    p = tmp_path / "hello.txt"
    p.write_text("Hello, world.", encoding="utf-8")
    docs = TextLoader().load(str(p))
    assert len(docs) == 1
    assert docs[0].text == "Hello, world."
    assert docs[0].id == "hello"
    assert docs[0].metadata["loader"] == "text"


def test_text_loader_missing_file_raises_loader_failed(tmp_path: Path) -> None:
    with pytest.raises(RagError) as info:
        TextLoader().load(str(tmp_path / "nonexistent.txt"))
    assert info.value.code == "loader_failed"


def test_text_loader_empty_file_returns_no_documents(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("   \n  ", encoding="utf-8")
    assert TextLoader().load(str(p)) == []


def test_markdown_loader_extracts_h1_as_title(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# My Title\n\nSome body text.", encoding="utf-8")
    docs = MarkdownLoader().load(str(p))
    assert docs[0].metadata["title"] == "My Title"


def test_markdown_loader_no_h1_falls_back_to_filename_stem(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("Just body text, no heading.", encoding="utf-8")
    docs = MarkdownLoader().load(str(p))
    assert docs[0].metadata["title"] == "doc"
