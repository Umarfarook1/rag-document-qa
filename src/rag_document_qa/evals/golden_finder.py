"""FinDER golden-pair loader.

FinDER (April 2026) ships ~5,703 expert-annotated query/evidence/answer triplets
on real SEC 10-K filings. Each triplet has at minimum:

  - query (str)
  - evidence (list[str])  -- supporting passages from the source 10-K
  - answer (str)
  - source (dict)         -- {ticker, fiscal_year, ...} identifying the filing

The dataset is distributed as a JSON Lines file. Layout names may evolve; the
loader accepts a couple of common synonyms (`question`/`query`,
`gold_passages`/`evidence`) so future schema tweaks don't break us silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_document_qa.errors import RagError
from rag_document_qa.types import GoldPair, Question


def load_finder_jsonl(path: Path, limit: int | None = None) -> list[GoldPair]:
    """Parse a FinDER JSONL file into our `GoldPair` shape.

    Tolerant of two field-naming conventions:
      - {"question": ..., "gold_passages": [...], "doc_id": "..."}
      - {"query":    ..., "evidence":      [...], "source": {"ticker": "..."}}
    """
    if not path.exists():
        raise RagError(
            code="invalid_corpus",
            message=f"FinDER jsonl not found at {path}",
            details={"path": str(path)},
        )

    pairs: list[GoldPair] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                raise RagError(
                    code="invalid_corpus",
                    message=f"FinDER jsonl line {lineno} is not valid JSON: {e}",
                    details={"path": str(path), "lineno": lineno},
                ) from e

            qtext = obj.get("query") or obj.get("question")
            evidence = obj.get("evidence") or obj.get("gold_passages") or []
            qid = str(obj.get("id") or obj.get("query_id") or f"finder_{lineno}")
            if not qtext:
                raise RagError(
                    code="invalid_corpus",
                    message=f"FinDER line {lineno} missing query/question",
                    details={"lineno": lineno},
                )
            if not isinstance(evidence, list):
                raise RagError(
                    code="invalid_corpus",
                    message=f"FinDER line {lineno} evidence must be a list",
                    details={"lineno": lineno},
                )
            doc_id = str(obj.get("doc_id") or _extract_doc_id_from_source(obj.get("source")))

            pairs.append(
                GoldPair(
                    question=Question(
                        id=qid, text=str(qtext), metadata={"finder_raw_keys": list(obj.keys())}
                    ),
                    gold_passages=[str(e) for e in evidence],
                    gold_doc_id=doc_id,
                    metadata={
                        "answer": str(obj.get("answer", "")),
                        "source": obj.get("source", {}),
                    },
                )
            )
            if limit is not None and len(pairs) >= limit:
                break

    if not pairs:
        raise RagError(
            code="corpus_empty",
            message=f"FinDER jsonl at {path} contained no usable rows",
        )
    return pairs


def _extract_doc_id_from_source(source: Any) -> str:
    """Map a FinDER source-dict to our doc_id convention (`{ticker}-10k-{accession}`).

    Falls back to ticker alone when accession is absent, or empty string when
    nothing is available. The eval harness still works without a doc_id (it
    just disables the doc-id prefilter in `passage_overlap_match`).
    """
    if not isinstance(source, dict):
        return ""
    ticker = str(source.get("ticker", "")).lower()
    accession = str(source.get("accession", "")).replace("-", "")
    if ticker and accession:
        return f"{ticker}-10k-{accession}"
    return ticker
