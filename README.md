<div align="center">

# rag-document-qa

**A retrieval-augmented Q&A system whose retriever is measured, not assumed.**

[![accuracy](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Umarfarook1/rag-document-qa/main/evals/badge.json)](#eval-harness)
[![CI](https://github.com/Umarfarook1/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Umarfarook1/rag-document-qa/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Retrieval over arbitrary document corpora &middot; citation-grounded answers &middot; eval-harness-tested retrieval &middot; v0.0.1 (in-development)

</div>

---

## What's different

Most "LangChain RAG quickstart" projects are unmeasured. They demo on three documents, ship, and silently break on real corpora the moment the right passage doesn't make it into the top-K. This one is built around the opposite assumption: **a RAG system without a retriever eval is not a system, it's a demo.**

Five concrete differences:

1. **The retriever has its own eval harness.** Every CI commit reports Recall@1, Recall@5, MRR, and nDCG against [FinDER](https://arxiv.org/html/2504.15800v1) (5,703 expert-annotated query/evidence/answer triplets on real SEC 10-K filings). The accuracy badge above is the live number.
2. **The vector store is a `Protocol`, not a hardcoded provider.** Switching from FAISS to Chroma to Pinecone is a one-file change. The eval harness then tells you exactly what each backend buys you.
3. **Reranking is optional but measured.** Cross-encoder rerank is off by default; turn it on and the harness shows the lift in numbers, not vibes.
4. **Hallucination guards are first-class.** If top-1 cosine is below threshold OR the model's self-rated confidence is below 6/10, the system explicitly returns "I don't know, here's what looked closest" rather than confabulating.
5. **Citations are forced.** The answer prompt requires `[chunk_3, chunk_7]`-style references. Answers without citations are rejected and re-prompted. Output is auditable.

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
                                 │  ─ runs in CI               │
                                 │  ─ writes badge.json        │
                                 └─────────────────────────────┘
                                       * optional
```

Every external dependency sits behind a `Protocol`. Tests run against in-memory fakes; CI exercises the full pipeline without an embedding model download or a paid API call.

## Data &middot; FinDER + SEC EDGAR

Production eval set: [FinDER](https://arxiv.org/html/2504.15800v1) (April 2026, expert-annotated). Production corpus: real 10-K filings pulled from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K) via the official API. Both free, official, no scraping.

## Quickstart (planned, in-development)

```bash
# clone + install (in-dev; expect the wheel on PyPI later)
git clone https://github.com/Umarfarook1/rag-document-qa
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"

# point at a corpus + your Anthropic key
cp .env.example .env && $EDITOR .env

# build the EDGAR corpus + index for a few tickers
rag-document-qa ingest --tickers AAPL,MSFT,NVDA

# ask
rag-document-qa ask "What were Apple's R&D expenses in fiscal 2024?"

# run the retrieval eval against FinDER
rag-document-qa evals run --golden finder --report evals/latest.json
```

## Eval harness

For each `(question, gold_passage)` pair in FinDER:

1. Embed the question with the same embedder used to ingest the corpus.
2. Retrieve top-K chunks from the vector index.
3. Optionally rerank with a cross-encoder.
4. Score: was the gold passage in top-K? At what rank?

Aggregate: **Recall@1**, **Recall@5**, **Recall@10**, **MRR**, **nDCG@10**. The shields.io badge tracks Recall@5 by default.

| Comparator concern | Semantics |
|---|---|
| Retrieved chunk overlap with gold passage | character-span overlap above 0.8 = match |
| Multiple gold passages per query | recall computed over the set; partial credit per match |
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
│   ├── confidence.py
│   ├── embed/
│   │   ├── fake.py
│   │   └── sentence_transformers.py
│   ├── index/
│   │   ├── memory.py
│   │   └── faiss.py
│   ├── loaders/
│   │   ├── text.py
│   │   ├── markdown.py
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
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── .github/workflows/
    ├── ci.yml
    └── evals.yml
```

## Status

**v0.0.1, in-development.** Scaffolding + protocols + fakes landing first. Real impls (BGE/FAISS, Anthropic, EDGAR) follow. CI green and FinDER badge live before the first tagged release.

This repo's predecessor lived on an older personal account and is being rebuilt cleanly here with senior-engineer standards (mypy strict, Protocol-based seams, eval harness from day 1, structured errors, TDD). Track progress on the [issues board](https://github.com/Umarfarook1/rag-document-qa/issues).

## License

MIT, see [`LICENSE`](LICENSE).

## Author

**Umarfarook Gurramkonda** &middot; AI Engineer
[GitHub](https://github.com/Umarfarook1) &middot; [Portfolio](https://umarfarook-ai.vercel.app)
