"""SEC EDGAR loader.

Fetches the most recent 10-K filing for a given ticker via SEC's free public
endpoints. Two-step lookup:

  1. ticker -> CIK via the official ticker-to-CIK map at
     https://www.sec.gov/files/company_tickers.json
  2. CIK -> filings list via data.sec.gov/submissions/CIK{cik}.json
  3. Pick the most recent 10-K, fetch its primary document.

EDGAR fair-use rules:
  - User-Agent header MUST identify you ("Name email@example.com").
  - Soft rate limit of 10 req/s. We sleep 0.12s between requests.

The fetched HTML is converted to plain text via a minimal regex strip; for
production use a proper HTML parser. The chunker downstream tolerates the
artifacts that survive.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from rag_document_qa.errors import RagError
from rag_document_qa.types import Document

EDGAR_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_DOC_URL_TMPL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary}"
)

DEFAULT_RATE_DELAY_S = 0.12  # ~8 req/s, comfortably under the 10/s ceiling
DEFAULT_TIMEOUT_S = 30.0

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class EdgarLoader:
    """Loads recent 10-K filings from SEC EDGAR by ticker."""

    def __init__(
        self,
        user_agent: str | None = None,
        rate_delay_s: float = DEFAULT_RATE_DELAY_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cache_dir: Path | None = None,
    ) -> None:
        ua = user_agent or os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise RagError(
                code="loader_failed",
                message=(
                    "EDGAR requires a User-Agent identifying the caller. Pass "
                    "user_agent= or set SEC_USER_AGENT env var (format: 'Name email@example.com')."
                ),
            )
        self._headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self._rate_delay_s = rate_delay_s
        self._timeout_s = timeout_s
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._ticker_map: dict[str, str] | None = None  # ticker -> 10-digit CIK

    @property
    def name(self) -> str:
        return "edgar"

    def load(self, source: str) -> list[Document]:
        """`source` is a ticker symbol (e.g. 'AAPL'). Returns the latest 10-K."""
        ticker = source.strip().upper()
        if not ticker:
            return []
        cik = self._lookup_cik(ticker)
        latest = self._latest_10k(cik)
        if latest is None:
            return []
        accession_nodash, primary = latest
        cik_int = str(int(cik))  # strip leading zeros
        doc_url = EDGAR_DOC_URL_TMPL.format(
            cik_int=cik_int,
            accession_nodash=accession_nodash,
            primary=primary,
        )
        html = self._fetch_text(doc_url)
        plain = _strip_html(html)
        if not plain.strip():
            return []
        doc_id = f"{ticker.lower()}-10k-{accession_nodash}"
        return [
            Document(
                id=doc_id,
                source=doc_url,
                text=plain,
                metadata={
                    "loader": self.name,
                    "ticker": ticker,
                    "cik": cik,
                    "accession": accession_nodash,
                    "filing_type": "10-K",
                },
            )
        ]

    def _lookup_cik(self, ticker: str) -> str:
        if self._ticker_map is None:
            payload = self._fetch_json(EDGAR_TICKER_MAP_URL)
            mapping: dict[str, str] = {}
            # The endpoint returns {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
            for entry in payload.values():
                t = str(entry.get("ticker", "")).upper()
                cik_str = str(entry.get("cik_str", "")).zfill(10)
                if t and cik_str:
                    mapping[t] = cik_str
            self._ticker_map = mapping
        cik = self._ticker_map.get(ticker)
        if not cik:
            raise RagError(
                code="loader_failed",
                message=f"ticker {ticker!r} not found in EDGAR ticker map",
                details={"ticker": ticker},
            )
        return cik

    def _latest_10k(self, cik: str) -> tuple[str, str] | None:
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        payload = self._fetch_json(url)
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primaries = recent.get("primaryDocument", [])
        for form, accession, primary in zip(forms, accessions, primaries, strict=True):
            if form == "10-K":
                return accession.replace("-", ""), primary
        return None

    def _fetch_json(self, url: str) -> dict[str, Any]:
        result: Any = self._cached_get(url, parse_json=True)
        if not isinstance(result, dict):
            raise RagError(
                code="loader_failed",
                message=f"expected JSON object from {url}, got {type(result).__name__}",
            )
        return result

    def _fetch_text(self, url: str) -> str:
        result: Any = self._cached_get(url, parse_json=False)
        return str(result)

    def _cached_get(self, url: str, parse_json: bool) -> Any:
        cached_payload = self._cache_get(url)
        if cached_payload is not None:
            if parse_json:
                import json

                return json.loads(cached_payload)
            return cached_payload

        time.sleep(self._rate_delay_s)
        try:
            with httpx.Client(timeout=self._timeout_s, headers=self._headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
                body = resp.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RagError(
                    code="rate_limited",
                    message=f"EDGAR rate-limited for {url}",
                ) from e
            raise RagError(
                code="loader_failed",
                message=f"EDGAR HTTP {e.response.status_code} for {url}",
            ) from e
        except httpx.HTTPError as e:
            raise RagError(
                code="loader_failed",
                message=f"EDGAR fetch failed for {url}: {e}",
            ) from e

        self._cache_put(url, body)
        if parse_json:
            import json

            return json.loads(body)
        return body

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        # Map URL to a filesystem-safe filename
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", url)[-160:]
        return self._cache_dir / safe

    def _cache_get(self, url: str) -> str | None:
        p = self._cache_path(url)
        if p is None or not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    def _cache_put(self, url: str, body: str) -> None:
        p = self._cache_path(url)
        if p is None:
            return
        p.write_text(body, encoding="utf-8")


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Good enough for chunking; not pretending to handle HTML semantics. For
    production-quality 10-K extraction (preserving tables, headings), use a
    real parser like lxml or trafilatura.
    """
    no_tags = _HTML_TAG_RE.sub(" ", html)
    return _WS_RE.sub(" ", no_tags).strip()
