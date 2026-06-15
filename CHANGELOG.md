# Changelog

Recent history (generated 2026-06-16):

- bb128e2 docs: add Roadmap section to README
- 8bc0d0d docs: add Usage section to README
- 9124208 docs: add Installation section to README
- 5fda33c docs: add CONTRIBUTING guide
- 1fbefa0 cli: ingest/ask/evals subcommands plus CI and weekly retrieval-eval workflows
- a507932 loader: SEC EDGAR client (CIK lookup, latest 10-K fetch, on-disk cache)
- d96c50a evals: Recall@K/MRR/nDCG metrics, runner, JSON+badge writer, FinDER loader
- 8cb2cf0 answer: Anthropic generator with mandatory citations and dual-gate confidence
- c595911 retriever: orchestrator with embedder/index validation and optional reranker
- ba320fc real-impls: BAAI/bge-small-en-v1.5 embedder and FAISS-cpu IndexFlatIP
- c1d6cba loaders: recursive text splitter, text/markdown loaders, pdf with pdfplumber+pypdf2 fallback
- adb9ddf index: in-memory cosine index with persistence and load round-trip
- f95052a gitignore: scope index/corpora ignore rules to repo root only
- 1cce48d core: domain dataclasses, structured RagError with stable codes, Protocol seams
- 6f0c552 scaffold: pyproject (mypy strict, ruff, pytest), README rewrite, env example
- 7ef9f54 init
