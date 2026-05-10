"""Real-EDGAR smoke test. Run with: pytest -m edgar

Hits SEC's free public endpoints to fetch the latest 10-K for AAPL. The
endpoint is rate-limited (10 req/s) and requires a User-Agent set via env var:

  SEC_USER_AGENT="Your Name your.email@example.com"

Cost: zero, but the test takes a few seconds and is gated so the default
`pytest` run skips it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_document_qa.loaders.edgar import EdgarLoader

pytestmark = pytest.mark.edgar


@pytest.fixture
def edgar(tmp_path: Path) -> EdgarLoader:
    if not os.environ.get("SEC_USER_AGENT"):
        pytest.skip("SEC_USER_AGENT env var not set")
    return EdgarLoader(cache_dir=tmp_path / "edgar_cache")


def test_fetch_aapl_10k(edgar: EdgarLoader) -> None:
    docs = edgar.load("AAPL")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["ticker"] == "AAPL"
    assert doc.metadata["filing_type"] == "10-K"
    assert "accession" in doc.metadata
    assert len(doc.text) > 1000  # 10-Ks are very long; this is just a sanity floor
