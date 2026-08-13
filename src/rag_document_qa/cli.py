"""Command-line interface for rag-document-qa.

Three subcommands:

  ingest     Build a vector index from a corpus (EDGAR tickers, file paths, etc.)
  ask        Run a single question through the full pipeline
  evals run  Score the retriever on a golden set; write JSON report + badge

The CLI is the only place that wires real implementations together. The library
itself stays Protocol-based so callers can swap pieces.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from rag_document_qa.types import Document


def main(argv: list[str] | None = None) -> int:
    """Main."""
    parser = argparse.ArgumentParser(prog="rag-document-qa")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Build a vector index from a corpus.")
    p_ingest.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of SEC tickers (e.g. AAPL,MSFT). "
        "Each fetches the latest 10-K from EDGAR.",
    )
    p_ingest.add_argument(
        "--paths",
        type=str,
        default=None,
        help="Comma-separated list of local file paths (txt, md, pdf).",
    )
    p_ingest.add_argument(
        "--index-out",
        type=Path,
        default=Path(".cache/index"),
        help="Where to persist the vector index.",
    )
    p_ingest.add_argument(
        "--embedder",
        choices=["fake", "bge"],
        default="bge",
        help="Embedder backend: `fake` (no model download) or `bge` (BAAI/bge-small-en-v1.5).",
    )
    p_ingest.add_argument(
        "--index-kind",
        choices=["memory", "faiss"],
        default="faiss",
        help="Vector store: in-memory numpy or FAISS.",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a single question against an indexed corpus.")
    p_ask.add_argument("question", type=str, help="The natural-language question.")
    p_ask.add_argument(
        "--index-in",
        type=Path,
        default=Path(".cache/index"),
        help="Path to a previously persisted index.",
    )
    p_ask.add_argument(
        "--embedder",
        choices=["fake", "bge"],
        default="bge",
    )
    p_ask.add_argument(
        "--index-kind",
        choices=["memory", "faiss"],
        default="faiss",
    )
    p_ask.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks returned by the retriever.",
    )
    p_ask.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the answer generator; print just the retrieved chunks.",
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_evals = sub.add_parser("evals", help="Run the retrieval eval harness.")
    p_evals_sub = p_evals.add_subparsers(dest="evals_cmd")
    p_evals_run = p_evals_sub.add_parser("run", help="Run a retrieval eval over a golden set.")
    p_evals_run.add_argument(
        "--golden",
        type=Path,
        required=True,
        help="Path to the golden file (FinDER JSONL).",
    )
    p_evals_run.add_argument(
        "--index-in",
        type=Path,
        default=Path(".cache/index"),
        help="Path to a previously persisted index.",
    )
    p_evals_run.add_argument(
        "--embedder",
        choices=["fake", "bge"],
        default="bge",
    )
    p_evals_run.add_argument(
        "--index-kind",
        choices=["memory", "faiss"],
        default="faiss",
    )
    p_evals_run.add_argument(
        "--report",
        type=Path,
        default=Path("evals/last_report.json"),
        help="Where to write the JSON report.",
    )
    p_evals_run.add_argument(
        "--badge-metric",
        type=str,
        default="recall_at_5",
        help="Which metric to put on the shields.io badge.",
    )
    p_evals_run.add_argument("--limit", type=int, default=None, help="Run only N pairs.")
    p_evals_run.set_defaults(func=_cmd_evals_run)
    p_evals.set_defaults(func=_cmd_evals_help)

    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = getattr(args, "func", _cmd_help)
    return func(args)


def _cmd_help(_args: argparse.Namespace) -> int:
    """Cmd help."""
    print(
        "usage: rag-document-qa {ingest,ask,evals} ...\nTry `rag-document-qa --help`.",
        file=sys.stderr,
    )
    return 1


def _cmd_evals_help(_args: argparse.Namespace) -> int:
    """Cmd evals help."""
    print(
        "usage: rag-document-qa evals run --golden PATH ...\n"
        "Try `rag-document-qa evals run --help`.",
        file=sys.stderr,
    )
    return 1


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Cmd ingest."""
    _load_dotenv()
    from rag_document_qa.chunking import split_documents
    from rag_document_qa.types import IndexMetadata

    docs = _gather_documents(args.tickers, args.paths)
    if not docs:
        print("error: no documents to ingest (use --tickers or --paths)", file=sys.stderr)
        return 2

    chunks = split_documents(docs)
    embedder = _build_embedder(args.embedder)
    embeddings = embedder.encode([c.text for c in chunks])
    md = IndexMetadata(
        embedder_name=embedder.name,
        embedder_version=embedder.version,
        embedding_dim=embedder.dim,
        chunk_count=len(chunks),
        index_kind=args.index_kind,
    )
    index = _build_index(args.index_kind)
    index.build(chunks, embeddings, md)
    index.persist(args.index_out)
    print(
        f"Indexed {len(docs)} document(s) -> {len(chunks)} chunks at {args.index_out.resolve()}",
        file=sys.stderr,
    )
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Cmd ask."""
    _load_dotenv()
    from rag_document_qa.retriever import Retriever, RetrieverConfig
    from rag_document_qa.types import Question

    embedder = _build_embedder(args.embedder)
    index = _load_index(args.index_kind, args.index_in)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        config=RetrieverConfig(top_k_retrieve=max(args.top_k * 4, 20), top_k_return=args.top_k),
    )
    retrieved = retriever.retrieve(args.question)

    if args.no_llm:
        for r in retrieved:
            print(f"[{r.rank}] ({r.score:.3f}) {r.chunk.id}")
            print(r.chunk.text[:280])
            print()
        return 0

    from rag_document_qa.answer.anthropic import AnthropicAnswerGenerator

    generator = AnthropicAnswerGenerator()
    result = generator.generate(Question(id="cli", text=args.question, metadata={}), retrieved)
    print(result.text)
    print()
    if result.cited_chunk_ids:
        print(f"Citations: {', '.join(result.cited_chunk_ids)}", file=sys.stderr)
    if not result.confident and result.no_answer_reason:
        print(f"(unconfident: {result.no_answer_reason})", file=sys.stderr)
    return 0 if result.confident else 1


def _cmd_evals_run(args: argparse.Namespace) -> int:
    """Cmd evals run."""
    _load_dotenv()
    from rag_document_qa.evals.golden_finder import load_finder_jsonl
    from rag_document_qa.evals.report import write_badge, write_report
    from rag_document_qa.evals.runner import run_retrieval_eval
    from rag_document_qa.retriever import Retriever, RetrieverConfig

    pairs = load_finder_jsonl(args.golden, limit=args.limit)
    embedder = _build_embedder(args.embedder)
    index = _load_index(args.index_kind, args.index_in)
    retriever = Retriever(
        embedder=embedder,
        index=index,
        config=RetrieverConfig(top_k_retrieve=20, top_k_return=10),
    )
    retriever_name = f"{embedder.name}+{args.index_kind}"

    print(
        f"Running retrieval eval: pairs={len(pairs)} retriever={retriever_name}",
        file=sys.stderr,
    )
    report = run_retrieval_eval(pairs, retriever.retrieve)

    print(
        f"\nResults: R@1={report.recall_at_1:.1%} R@5={report.recall_at_5:.1%} "
        f"R@10={report.recall_at_10:.1%} MRR={report.mrr:.3f} "
        f"nDCG@10={report.ndcg_at_10:.3f} errors={report.errors}",
        file=sys.stderr,
    )

    write_report(report, args.report, retriever_name=retriever_name)
    print(f"Wrote report to {args.report.resolve()}", file=sys.stderr)
    badge_path = args.report.parent / "badge.json"
    write_badge(report, badge_path, retriever_name=retriever_name, metric=args.badge_metric)
    print(f"Wrote badge to {badge_path.resolve()}", file=sys.stderr)
    return 0 if report.recall_at_5 > 0 else 1


# ---- Helpers ----


def _load_dotenv() -> None:
    """Load dotenv."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _gather_documents(tickers: str | None, paths: str | None) -> list[Document]:
    """Gather documents."""
    from rag_document_qa.loaders.edgar import EdgarLoader
    from rag_document_qa.loaders.pdf import PDFLoader
    from rag_document_qa.loaders.text import MarkdownLoader, TextLoader

    docs: list[Document] = []
    if tickers:
        loader = EdgarLoader()
        for t in [t.strip() for t in tickers.split(",") if t.strip()]:
            docs.extend(loader.load(t))
    if paths:
        for raw in [p.strip() for p in paths.split(",") if p.strip()]:
            p = Path(raw)
            if p.suffix.lower() == ".pdf":
                docs.extend(PDFLoader().load(str(p)))
            elif p.suffix.lower() in {".md", ".markdown"}:
                docs.extend(MarkdownLoader().load(str(p)))
            else:
                docs.extend(TextLoader().load(str(p)))
    return docs


def _build_embedder(kind: str):  # type: ignore[no-untyped-def]
    """Build embedder."""
    if kind == "fake":
        from rag_document_qa.embed.fake import FakeEmbedder

        return FakeEmbedder(dim=384)
    from rag_document_qa.embed.sentence_transformers import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder()


def _build_index(kind: str):  # type: ignore[no-untyped-def]
    if kind == "memory":
        from rag_document_qa.index.memory import InMemoryVectorIndex

        return InMemoryVectorIndex()
    from rag_document_qa.index.faiss import FaissVectorIndex

    return FaissVectorIndex()


def _load_index(kind: str, path: Path):  # type: ignore[no-untyped-def]
    if kind == "memory":
        from rag_document_qa.index.memory import InMemoryVectorIndex

        return InMemoryVectorIndex.load(path)
    from rag_document_qa.index.faiss import FaissVectorIndex

    return FaissVectorIndex.load(path)
