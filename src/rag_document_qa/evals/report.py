"""JSON report + shields.io endpoint badge writer.

The badge tracks Recall@5 by default since it's the most common headline metric
for RAG retrieval. Color thresholds:
  <30%: red
  <50%: orange
  <65%: yellow
  <80%: green
  >=80%: brightgreen
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_document_qa.evals.runner import EvalReport


def write_report(report: EvalReport, out: Path, retriever_name: str) -> None:
    """Write the full eval report as JSON. Includes retriever id + UTC timestamp."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "retriever": retriever_name,
        "generated_at": datetime.now(UTC).isoformat(),
        **asdict(report),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))


def write_badge(
    report: EvalReport,
    out: Path,
    retriever_name: str,
    metric: str = "recall_at_5",
) -> None:
    """Write a shields.io endpoint JSON file (https://shields.io/badges/endpoint-badge)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    value = _get_metric(report, metric)
    pct = int(round(value * 100))
    payload = {
        "schemaVersion": 1,
        "label": f"{retriever_name} {metric}",
        "message": f"{pct}%",
        "color": _color_for(value),
    }
    out.write_text(json.dumps(payload))


def _get_metric(report: EvalReport, metric: str) -> float:
    if not hasattr(report, metric):
        raise ValueError(
            f"unknown metric {metric!r}; must be one of "
            f"recall_at_1, recall_at_5, recall_at_10, mrr, ndcg_at_10"
        )
    val = getattr(report, metric)
    return float(val)


def _color_for(value: float) -> str:
    if value < 0.30:
        return "red"
    if value < 0.50:
        return "orange"
    if value < 0.65:
        return "yellow"
    if value < 0.80:
        return "green"
    return "brightgreen"
