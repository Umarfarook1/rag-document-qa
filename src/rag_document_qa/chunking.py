"""Recursive text splitter that produces Chunks from a Document.

Strategy: split first on the longest separator that yields fragments smaller
than the target size, then recurse on the shorter ones. This keeps semantic
units (paragraphs, sentences) intact when possible and only resorts to character
splits as a last resort.

Token-aware sizing is approximate via character count (4 chars per token is a
common heuristic). For exact tokenisation, plug in a real tokenizer at the
caller layer; this module stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_document_qa.errors import RagError
from rag_document_qa.types import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Knobs for the splitter.

    `chunk_size` is in characters by default (token-approximate). `chunk_overlap`
    is the number of characters reused between adjacent chunks; helps queries
    that span chunk boundaries.
    """

    chunk_size: int = 1024
    chunk_overlap: int = 128
    separators: tuple[str, ...] = DEFAULT_SEPARATORS


class RecursiveTextSplitter:
    """Recursive Text Splitter."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """Init."""
        self._cfg = config or ChunkConfig()
        if self._cfg.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self._cfg.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self._cfg.chunk_overlap >= self._cfg.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []
        pieces = self._split_recursive(document.text, list(self._cfg.separators))
        merged = self._merge_with_overlap(pieces)
        # Map each merged piece back to its position in the source text.
        chunks: list[Chunk] = []
        cursor = 0
        for i, piece in enumerate(merged):
            start = document.text.find(piece, cursor)
            if start < 0:
                # Defensive: if find fails (overlap caused duplication), fall back
                # to advancing cursor by the previous chunk's length.
                start = cursor
            end = start + len(piece)
            chunks.append(
                Chunk(
                    id=f"{document.id}::chunk_{i}",
                    doc_id=document.id,
                    text=piece,
                    chunk_index=i,
                    char_start=start,
                    char_end=end,
                    metadata=dict(document.metadata),
                )
            )
            # Advance cursor to (end - overlap) so subsequent .find() locates
            # the next chunk's start correctly.
            cursor = max(start + 1, end - self._cfg.chunk_overlap)
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split until every piece fits under chunk_size."""
        if len(text) <= self._cfg.chunk_size:
            return [text] if text else []

        # Find the first separator that actually appears in this text.
        sep = ""
        remaining: list[str] = list(separators)
        while remaining:
            candidate = remaining.pop(0)
            if candidate == "" or candidate in text:
                sep = candidate
                break

        if sep == "":
            # Last resort: hard-split by character.
            return [
                text[i : i + self._cfg.chunk_size]
                for i in range(0, len(text), self._cfg.chunk_size)
            ]

        out: list[str] = []
        for piece in text.split(sep):
            if not piece:
                continue
            piece_with_sep = piece + sep if sep else piece
            if len(piece_with_sep) <= self._cfg.chunk_size:
                out.append(piece_with_sep)
            else:
                out.extend(self._split_recursive(piece_with_sep, remaining))
        return [p for p in out if p]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedy-merge small pieces into chunks of <= chunk_size, with overlap."""
        if not pieces:
            return []
        merged: list[str] = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= self._cfg.chunk_size:
                current = current + piece
            else:
                merged.append(current)
                # Pull tail of `current` (up to overlap chars) onto `piece` so
                # adjacent chunks share context across the boundary.
                tail = current[-self._cfg.chunk_overlap :] if self._cfg.chunk_overlap else ""
                current = tail + piece
        if current:
            merged.append(current)
        return merged


# ---- Loader registry ----


def split_documents(
    documents: list[Document],
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Convenience: chunk a list of documents in one call."""
    if not documents:
        raise RagError(
            code="corpus_empty",
            message="split_documents() called with zero documents",
        )
    splitter = RecursiveTextSplitter(config)
    out: list[Chunk] = []
    for doc in documents:
        out.extend(splitter.split(doc))
    return out
