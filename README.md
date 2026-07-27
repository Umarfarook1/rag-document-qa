<div align="center">

# rag-document-qa

**A retrieval-augmented Q&A system built around a retriever eval harness.**

[![retrieval eval](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Umarfarook1/rag-document-qa/main/evals/badge.json)](#eval-harness)
[![CI](https://github.com/Umarfarook1/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Umarfarook1/rag-document-qa/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Retrieval over arbitrary document corpora &middot; citation-grounded answers &middot; retriever eval harness &middot; v0.0.1 (in-development)

</div>

---

## Design

Most "LangChain RAG quickstart" projects are unmeasured. They demo on three documents, ship, and silently break on real corpora the moment the right passage doesn't make it into the top-K. This repo is built the other way round: the retrieval eval is a component with its own CLI, metrics and report format rather than a script someone runs once. By that standard it is not finished, because the harness has not been pointed at a benchmark yet.

Five design choices:

1. **The retriever has its own eval harness.** It computes Recall@1, Recall@5, Recall@10, MRR and nDCG@10, covered by unit tests plus a CLI end-to-end test that runs ingest, ask and evals on fake embeddings. It has not been run against [FinDER](https://arxiv.org/html/2504.15800v1) (5,703 expert-annotated query/evidence/answer triplets on real SEC 10-K filings) yet. The weekly workflow skips because the `SEC_USER_AGENT` secret and `evals/golden_finder.jsonl` are not set up, so the badge above reads `pending`.
2. **The vector store is a `Protocol`, not a hardcoded provider.** Two backends implement it today: a numpy in-memory index and FAISS `IndexFlatIP` over L2-normalised vectors. A Chroma or Pinecone backend would be a new file implementing the same five members. No cross-backend comparison has been run.
3. **Reranking sits behind a `Protocol`.** `rerank.py` ships an identity pass-through and a reverse implementation used in tests. No cross-encoder reranker is implemented, and the eval CLI builds its retriever without one, so there is no measured lift to report.
4. **Hallucination guards are first-class.** If top-1 cosine is below threshold OR the model's self-rated confidence is below 6/10, the system explicitly returns "I don't know, here's what looked closest" rather than confabulating.
5. **Citations are forced.** The answer prompt requires `[chunk_3, chunk_7]`-style references, and only ids that match a retrieved chunk are accepted. An uncited answer is retried once with a stricter follow-up; if it comes back uncited again, the result is returned unconfident with reason `no_citations_emitted`.

## Architecture

```
                                 ┌─────────────────────────────┐
  Document corpus                │  rag-document-qa            │
  (PDF / MD / TXT / EDGAR)       │                             │
            │                    │  ┌───────────────────────┐  │
            ▼                    │  │ DocLoader Protocol    │  │
  ┌─────────────────┐            │  │ Chunker (recursive)   │  │
  │ Loader+Chunker  │ ─────────►│  │ Embedder Protocol     │  │
  └─────────────────┘            │  │ VectorIndex Protocol  │  │
                                 │  │ Retriever (top-K)     │  │
  User question ──────────────► │  │ Reranker Protocol *   │  │
                                 │  │ AnswerGenerator       │  │
                                 │  │ Confidence gate       │  │
                                 │  └───────────────────────┘  │
                                 │                             │
                                 │  Eval Harness               │
                                 │  ─ Recall@K, MRR, nDCG      │
                                 │  ─ no benchmark run yet     │
                                 │  ─ writes badge.json        │
                                 └─────────────────────────────┘
                                       * optional
```

Every external dependency sits behind a `Protocol`. Tests run against in-memory fakes; CI exercises the full pipeline without an embedding model download or a paid API call.

## Data &middot; FinDER + SEC EDGAR

Intended eval set: [FinDER](https://arxiv.org/html/2504.15800v1) (April 2026, expert-annotated). `evals/golden_finder.py` parses its JSONL into the harness's `GoldPair` shape, but no golden file is committed, so the harness has no data to run on here.

Intended corpus: real 10-K filings pulled from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K) via the official API. `loaders/edgar.py` implements the client (CIK lookup, latest 10-K fetch, on-disk cache) and `tests/integration/test_edgar_smoke.py` exercises it against the live API under `pytest -m edgar`, which the default test run deselects. Both sources are free and official, no scraping.

## Quickstart

Not on PyPI. Install from source.

```bash
git clone https://github.com/Umarfarook1/rag-document-qa
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `fake` embedder and the `memory` index need no model download, no network and no API key, so this path runs on that install alone:

```bash
rag-document-qa ingest --paths notes.txt --index-out .cache/index \
  --embedder fake --index-kind memory
rag-document-qa ask "your question" --index-in .cache/index \
  --embedder fake --index-kind memory --no-llm
```

Real embeddings, EDGAR and answer generation need the extras and two secrets:

```bash
pip install -e ".[dev,all]"
cp .env.example .env && $EDITOR .env    # ANTHROPIC_API_KEY, SEC_USER_AGENT

rag-document-qa ingest --tickers AAPL,MSFT,NVDA
rag-document-qa ask "What were Apple's R&D expenses in fiscal 2024?"
```

`--golden` takes a path to a JSONL file in FinDER's shape, one `{id, query, evidence, doc_id}` object per line. No golden file ships with the repo, so you have to supply one:

```bash
rag-document-qa evals run --golden evals/golden_finder.jsonl --report evals/latest.json
```

## Eval harness

For each `(question, gold_passage)` pair in the golden file:

1. Embed the question with the same embedder used to ingest the corpus.
2. Retrieve top-K chunks from the vector index (the eval CLI retrieves 20, returns 10).
3. If the `Retriever` was built with a `Reranker`, rerank. The eval CLI passes none.
4. Score: was a gold passage in top-K? At what rank?

Aggregate: **Recall@1**, **Recall@5**, **Recall@10**, **MRR**, **nDCG@10**. The shields.io badge tracks Recall@5 by default and currently reads `pending`, because the harness has not been run on a benchmark.

| Comparator concern | Semantics |
|---|---|
| Retrieved chunk overlap with gold passage | longest common substring as a fraction of the shorter string, 0.5 or above = match. Containment in either direction is the 1.0 case. |
| Multiple gold passages per query | binary hit: the query scores 1.0 if any gold passage matches a retrieved chunk, 0.0 otherwise |
| Embedder version mismatch | hard fail at query time (mismatch detected via index metadata stamp) |

## Repo structure

```
rag-document-qa/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── src/rag_document_qa/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── types.py
│   ├── errors.py
│   ├── protocols.py
│   ├── chunking.py
│   ├── retriever.py
│   ├── rerank.py
│   ├── confidence.py
│   ├── embed/
│   │   ├── fake.py
│   │   └── sentence_transformers.py
│   ├── index/
│   │   ├── memory.py
│   │   └── faiss.py
│   ├── loaders/
│   │   ├── text.py
│   │   ├── pdf.py
│   │   └── edgar.py
│   ├── answer/
│   │   ├── fake.py
│   │   └── anthropic.py
│   └── evals/
│       ├── metrics.py
│       ├── runner.py
│       ├── report.py
│       └── golden_finder.py
├── evals/
│   └── badge.json
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── .github/workflows/
    ├── ci.yml
    └── evals.yml
```

## Status

**v0.0.1, in-development.** Landed: the Protocols, the fakes, and the real implementations behind them (BGE via sentence-transformers, FAISS, Anthropic, EDGAR), the chunker, the confidence gate, the metrics and the eval runner. 126 unit tests pass with no model download, no network and no API key. CI runs ruff, ruff format, mypy strict and pytest on 3.11 and 3.12.

Not landed: no golden file is committed, so the eval harness has never been run on a benchmark and the badge reads `pending`. No cross-encoder reranker. The weekly eval workflow is wired but exits early without the `SEC_USER_AGENT` secret.

This repo's predecessor lived on an older personal account and is being rebuilt cleanly here (mypy strict, Protocol-based seams, structured errors, tests first). Track progress on the [issues board](https://github.com/Umarfarook1/rag-document-qa/issues).

## License

MIT, see [`LICENSE`](LICENSE).

## Author

**Umarfarook Gurramkonda** &middot; AI Engineer
[GitHub](https://github.com/Umarfarook1) &middot; [Portfolio](https://umarfarook-ai.vercel.app)
