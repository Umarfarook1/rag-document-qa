import json
from pathlib import Path

import pytest

from rag_document_qa.evals.report import write_badge, write_report
from rag_document_qa.evals.runner import EvalReport, PerQueryResult


def _sample_report(recall_at_5: float = 0.6) -> EvalReport:
    return EvalReport(
        total=10,
        recall_at_1=0.4,
        recall_at_5=recall_at_5,
        recall_at_10=0.8,
        mrr=0.5,
        ndcg_at_10=0.55,
        avg_latency_ms=120,
        errors=0,
        per_query=[
            PerQueryResult(
                question_id="q1",
                nl="?",
                gold_doc_id="d1",
                recall_at_1=1.0,
                recall_at_5=1.0,
                recall_at_10=1.0,
                mrr=1.0,
                ndcg_at_10=1.0,
                top_chunk_id="c1",
                top_score=0.95,
                latency_ms=100,
                error=None,
            )
        ],
    )


def test_write_report_creates_json(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    write_report(_sample_report(), out, retriever_name="bge-faiss")
    data = json.loads(out.read_text())
    assert data["retriever"] == "bge-faiss"
    assert data["total"] == 10
    assert data["recall_at_5"] == 0.6
    assert "per_query" in data
    assert "generated_at" in data


def test_write_report_creates_parent_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deep" / "r.json"
    write_report(_sample_report(), out, retriever_name="x")
    assert out.exists()


def test_write_badge_produces_shields_endpoint_json(tmp_path: Path) -> None:
    out = tmp_path / "badge.json"
    write_badge(_sample_report(), out, retriever_name="bge-faiss")
    data = json.loads(out.read_text())
    assert data["schemaVersion"] == 1
    assert "bge-faiss" in data["label"]
    assert "60%" in data["message"]
    assert data["color"] in {"red", "orange", "yellow", "green", "brightgreen"}


def test_badge_color_thresholds(tmp_path: Path) -> None:
    cases = [
        (0.10, "red"),
        (0.29, "red"),
        (0.30, "orange"),
        (0.49, "orange"),
        (0.50, "yellow"),
        (0.64, "yellow"),
        (0.65, "green"),
        (0.79, "green"),
        (0.80, "brightgreen"),
        (1.00, "brightgreen"),
    ]
    for value, expected in cases:
        out = tmp_path / f"b_{int(value * 100)}.json"
        write_badge(_sample_report(recall_at_5=value), out, retriever_name="r")
        data = json.loads(out.read_text())
        assert data["color"] == expected, f"value={value}: expected {expected}, got {data['color']}"


def test_write_badge_picks_alternate_metric(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    write_badge(_sample_report(), out, retriever_name="r", metric="mrr")
    data = json.loads(out.read_text())
    assert "mrr" in data["label"]
    # mrr is 0.5 -> 50% -> yellow band
    assert "50%" in data["message"]


def test_write_badge_unknown_metric_rejects(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    with pytest.raises(ValueError, match="unknown metric"):
        write_badge(_sample_report(), out, retriever_name="r", metric="bogus")
