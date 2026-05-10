"""End-to-end CLI smoke: ingest text files with fakes, ask, run evals.

Uses --embedder fake + --index-kind memory so no external deps + no API keys.
Validates that the CLI wires every layer together correctly without an
embedding model download or paid API call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_document_qa.cli import main


def _write_corpus(tmp_path: Path) -> list[Path]:
    """Create three small text documents the index can chunk."""
    paths: list[Path] = []
    for i, text in enumerate(
        [
            "Apple reported R&D expenses of $30 billion in fiscal 2024.",
            "Microsoft's cloud revenue grew 28% year over year in fiscal 2024.",
            "Nvidia data center revenue tripled in 2024 driven by AI chip demand.",
        ]
    ):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


def test_cli_ingest_then_ask_no_llm_e2e(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = _write_corpus(tmp_path)
    index_dir = tmp_path / "index"
    rc = main(
        [
            "ingest",
            "--paths",
            ",".join(str(p) for p in docs),
            "--index-out",
            str(index_dir),
            "--embedder",
            "fake",
            "--index-kind",
            "memory",
        ]
    )
    assert rc == 0
    assert (index_dir / "index.json").exists()

    capsys.readouterr()  # drain stderr from ingest

    rc2 = main(
        [
            "ask",
            "Apple R&D expenses",
            "--index-in",
            str(index_dir),
            "--embedder",
            "fake",
            "--index-kind",
            "memory",
            "--top-k",
            "3",
            "--no-llm",
        ]
    )
    assert rc2 == 0
    out = capsys.readouterr().out
    # Three retrieved chunks printed with rank prefixes.
    assert "[1]" in out and "[2]" in out and "[3]" in out


def test_cli_evals_run_e2e_with_fakes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = _write_corpus(tmp_path)
    index_dir = tmp_path / "index"
    main(
        [
            "ingest",
            "--paths",
            ",".join(str(p) for p in docs),
            "--index-out",
            str(index_dir),
            "--embedder",
            "fake",
            "--index-kind",
            "memory",
        ]
    )
    capsys.readouterr()

    # FakeEmbedder is hash-based, so each gold passage maps to a unique vector.
    # If we put the EXACT chunk text in evidence, the retriever finds it.
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"q{i}",
                    "query": text,
                    "evidence": [text],
                    "doc_id": f"doc{i}",
                }
            )
            for i, text in enumerate(
                [
                    "Apple reported R&D expenses of $30 billion in fiscal 2024.",
                    "Microsoft's cloud revenue grew 28% year over year in fiscal 2024.",
                ]
            )
        ),
        encoding="utf-8",
    )

    report_path = tmp_path / "evals" / "report.json"
    rc = main(
        [
            "evals",
            "run",
            "--golden",
            str(golden),
            "--index-in",
            str(index_dir),
            "--embedder",
            "fake",
            "--index-kind",
            "memory",
            "--report",
            str(report_path),
        ]
    )
    assert rc == 0
    assert report_path.exists()
    assert (report_path.parent / "badge.json").exists()
    data = json.loads(report_path.read_text())
    assert "recall_at_5" in data
    assert data["total"] == 2
