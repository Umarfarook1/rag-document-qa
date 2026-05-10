"""Plain-text + Markdown loaders.

Both produce a single `Document` per input file. Markdown loader preserves
the H1 heading (if any) into `metadata["title"]` so it can be cited.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_document_qa.errors import RagError
from rag_document_qa.types import Document

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class TextLoader:
    @property
    def name(self) -> str:
        return "text"

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise RagError(
                code="loader_failed",
                message=f"text file not found: {source}",
                details={"source": source},
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise RagError(
                code="loader_failed",
                message=f"failed to read {source}: {e}",
                details={"source": source},
            ) from e
        if not text.strip():
            return []
        return [
            Document(
                id=path.stem,
                source=str(path),
                text=text,
                metadata={"loader": self.name, "ext": path.suffix},
            )
        ]


class MarkdownLoader:
    @property
    def name(self) -> str:
        return "markdown"

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise RagError(
                code="loader_failed",
                message=f"markdown file not found: {source}",
                details={"source": source},
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise RagError(
                code="loader_failed",
                message=f"failed to read {source}: {e}",
                details={"source": source},
            ) from e
        if not text.strip():
            return []
        h1 = _H1_RE.search(text)
        title = h1.group(1).strip() if h1 else path.stem
        return [
            Document(
                id=path.stem,
                source=str(path),
                text=text,
                metadata={"loader": self.name, "ext": path.suffix, "title": title},
            )
        ]
