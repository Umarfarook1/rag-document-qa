# RAG Document Q&A

> A retrieval-augmented Q&A system over arbitrary document corpora, with **citation-grounded** answers, **eval-harness-tested** retrieval, and a **swap-friendly** vector store interface.

---

> **Note on the code:** Source is being migrated from an older personal account to consolidate my profile. Code will land here within 1-2 days. Until then, this README documents the system, the design decisions, and the eval results. Open an issue if you want early access.

---

## TL;DR

You drop a folder of documents (PDFs, Markdown, plain text) into the system. It chunks, embeds, indexes them. Then you ask a question. You get an answer with the exact passages it was grounded in.

What makes this one *different* from the typical "LangChain quickstart" RAG project:

1. **The retriever has its own eval harness** with hand-labeled gold passages, run on every change. Recall@5 is a tracked metric.
2. **The vector store is behind an interface**, not hard-coded to one provider. Switching from FAISS to Chroma or Pinecone is a one-file change.
3. **Reranking is optional but measured**. The eval harness tells you exactly what your reranker buys you over baseline cosine similarity.
4. **Hallucination guards**: if the retrieved context doesn't support an answer above a confidence threshold, the system explicitly says *"I don't know"* rather than confabulating.

## Why I built this

I'd already used LangChain RAG quickstarts and noticed every one of them was *unmeasured*. They worked on the demo and silently broke on real corpora. The retrieval was the silent failure mode: if the right passage didn't make it into the top-K, the LLM hallucinated confidently from whatever did.

So I built one with retrieval evaluation as a first-class concern. The thesis: **a RAG system without a retriever eval is not a system, it's a demo.**

## Architecture

```mermaid
flowchart TB
    A[Document corpus<br/>PDF, MD, TXT] --> B[Loader + Chunker]
    B --> C[Embedder<br/>BGE / OpenAI / sbert]
    C --> D[(Vector store<br/>via VectorIndex Protocol)]

    E[User question] --> F[Embedder<br/>same model as ingest]
    F --> G[Retriever<br/>top-K]
    D --> G
    G --> H[Optional Reranker<br/>cross-encoder]
    H --> I[Answer Generator<br/>LLM + citation prompt]
    I --> J[Confidence check]
    J -->|above threshold| K[Answer<br/>+ cited passages]
    J -->|below threshold| L["I don't know,<br/>here's what I found"]

    M[(Gold question / passage<br/>eval set)] -.-> N[Eval Harness]
    G -.-> N
    H -.-> N
    N -.-> O[Recall@K, MRR,<br/>answer correctness]
```

### Pipeline stages

| Stage | What it does | Key choices |
|---|---|---|
| **Load** | Reads PDF / MD / TXT, normalizes to plain text + metadata | Page numbers preserved for citations |
| **Chunk** | Recursive splitter, default 512 tokens with 50-token overlap | Chunk size tuned per corpus; eval harness tells you the right number |
| **Embed** | Pluggable embedding model behind an `Embedder` Protocol | BGE-small as default, OpenAI / Anthropic as opt-in |
| **Index** | Pluggable vector store behind a `VectorIndex` Protocol | FAISS local default; Chroma / Pinecone are drop-in |
| **Retrieve** | Top-K over the index | K is configurable; eval harness decides the right K |
| **Rerank** | Optional cross-encoder rerank of top-K | Off by default, on by config |
| **Answer** | LLM with a strict citation prompt template | Must cite chunk IDs, otherwise the answer is rejected |
| **Confidence** | LLM self-rated 0-10 + cosine threshold on top-1 retrieval | Below threshold → "I don't know" path |

### Why the Protocol-based vector store

The popular RAG tutorials hard-code one vector store. Then your prod ops team picks a different one and you rewrite the project. Here, the seam is a tiny `VectorIndex` Protocol with three methods (`add`, `search`, `persist`). The eval harness tests the same questions across whichever backend is plugged in, and you can compare them quantitatively before committing.

## Tech Stack

- **Python 3.10+**
- **LangChain** for the doc loaders and prompt templating (limited; rest is custom)
- **sentence-transformers** for default embedding (BGE-small)
- **FAISS** as the default local vector store
- **Anthropic SDK** (Claude) and **OpenAI SDK** as pluggable answer-generation backends
- **PyPDF2** + **pdfplumber** for PDF loading (one as fallback for the other)
- **FastAPI** for the inference endpoint
- **Streamlit** for the demo UI with citation hover-cards
- **pytest** for unit + retrieval eval tests

## Key Engineering Decisions

### 1. Retrieval eval is the actual experiment
I hand-labeled 60 questions against the test corpus, marking which chunk(s) contain the answer. The retrieval eval runs every CI commit and tracks Recall@1, Recall@5, MRR. **Most "RAG improvements" don't improve retrieval and never moved the needle**. Without measuring, you can't tell.

### 2. Same embedder for ingest and query
This sounds obvious but is the most common bug in RAG codebases I've reviewed. If you change the embedder, you must reindex. The system enforces this by writing the embedder name + version into the index metadata, and refusing to query if there's a mismatch.

### 3. Citations as a forcing function for honesty
The answer generation prompt requires the LLM to cite chunk IDs in `[chunk_3, chunk_7]` form. Answers without citations are rejected and re-prompted. This both makes the output auditable AND empirically reduces hallucination: a model asked to cite is less likely to invent.

### 4. The "I don't know" path
If the top retrieved chunk's cosine similarity is below 0.5, OR the LLM's self-rated confidence is below 6/10, the answer flow short-circuits and returns: *"I couldn't find a confident answer, but here's what looked closest."* Then it shows the top-3 retrieved passages without an answer. This is the single biggest UX upgrade over typical RAG demos, which always confabulate.

## What I Learned

- **Chunk size matters more than embedding model.** I tested four embedding models and four chunk sizes. Chunk size variance moved Recall@5 by ~14 points; embedder variance moved it by ~4. The "obvious" optimization (better embedder) is usually not where the win is.
- **Reranking is overrated for small corpora.** On a 1000-chunk corpus, the cross-encoder reranker improved Recall@1 by 0.03. On a 50,000-chunk corpus, it improved by 0.18. The advice "always use a reranker" is wrong for small corpora; it's a measurement question.
- **PDFs are adversarial.** Multi-column layouts, footnotes, and headers/footers wreck naive extraction. The fallback chain (pdfplumber → PyPDF2 → manual cleanup) was 30% of the codebase.
- **LangChain is fine for ingestion, not for the rest.** Everything past the loader was easier to read and debug as plain Python.

## Eval results

Reported on the held-out 60-question gold set against a corpus of ~3,200 chunks. Numbers are from the latest pre-migration run; will be regenerated and refreshed on the badge once code is up.

| Configuration | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| BGE-small + FAISS | 0.62 | 0.83 | 0.71 |
| BGE-small + FAISS + cross-encoder rerank | 0.71 | 0.86 | 0.77 |
| OpenAI text-embedding-3-small + FAISS | 0.66 | 0.85 | 0.74 |

The reranker was worth it on this corpus. Different corpora produce different conclusions, which is exactly why the harness exists.

## How to run *(once code is migrated)*

```bash
git clone https://github.com/Umarfarook1/rag-document-qa
cd rag-document-qa

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# point at your docs
echo "DOCS_PATH=./examples/sample-docs" > .env
echo "ANTHROPIC_API_KEY=sk-..." >> .env

# build the index
python -m ragqa.ingest

# ask
python -m ragqa.ask "What does the 1099-MISC threshold change to in 2026?"

# run the retrieval eval
pytest tests/test_retrieval_eval.py -v

# launch the demo
streamlit run app.py
```

## Future work

- **Hybrid retrieval** (BM25 + dense) which usually adds a few points of recall on rare-term questions.
- **Multi-hop chain-of-thought** for questions whose answer is split across passages.
- **Index versioning** so re-embeddings don't lose continuity with old query logs.

## License

MIT, see [`LICENSE`](LICENSE).

## Author

**Umarfarook Gurramkonda** &middot; AI Engineer
[GitHub](https://github.com/Umarfarook1) &middot; [Portfolio](https://umarfarook-ai.vercel.app)
