"""PDF loader with pdfplumber primary + PyPDF2 fallback.

Both libraries are optional; if neither is installed, `load()` raises
`RagError(loader_failed)` with a hint. This keeps the core package usable for
text-only flows without forcing every consumer to install heavy PDF deps.

Real-world PDFs are adversarial (multi-column, footnotes, headers, scanned
pages). The two-library fallback chain catches the long tail of weird inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_document_qa.errors import RagError
from rag_document_qa.types import Document


class PDFLoader:
    @property
    def name(self) -> str:
        return "pdf"

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise RagError(
                code="loader_failed",
                message=f"pdf file not found: {source}",
                details={"source": source},
            )

        text, page_count, backend = self._extract(path)
        if not text.strip():
            return []
        return [
            Document(
                id=path.stem,
                source=str(path),
                text=text,
                metadata={
                    "loader": self.name,
                    "ext": ".pdf",
                    "pages": page_count,
                    "backend": backend,
                },
            )
        ]

    def _extract(self, path: Path) -> tuple[str, int, str]:
        # Try pdfplumber first.
        plumber_mod: Any = None
        try:
            import pdfplumber  # noqa: F401

            plumber_mod = pdfplumber
        except ImportError:
            plumber_mod = None

        if plumber_mod is not None:
            try:
                pages_text: list[str] = []
                pdf_obj: Any
                with plumber_mod.open(str(path)) as pdf_obj:
                    for page in pdf_obj.pages:
                        page_text = page.extract_text() or ""
                        pages_text.append(page_text)
                return "\n\n".join(pages_text), len(pages_text), "pdfplumber"
            except Exception:
                # Fall through to PyPDF2.
                pass

        reader_cls: Any = None
        try:
            from PyPDF2 import PdfReader

            reader_cls = PdfReader
        except ImportError:
            reader_cls = None

        if reader_cls is not None:
            try:
                reader: Any = reader_cls(str(path))
                pages_text = [(p.extract_text() or "") for p in reader.pages]
                return "\n\n".join(pages_text), len(pages_text), "pypdf2"
            except Exception as e:
                raise RagError(
                    code="loader_failed",
                    message=f"PyPDF2 failed to read {path}: {e}",
                    details={"source": str(path)},
                ) from e

        raise RagError(
            code="loader_failed",
            message=(
                "no PDF backend available; install pdfplumber or PyPDF2 "
                "(`pip install rag-document-qa[loaders]`)"
            ),
            details={"source": str(path)},
        )
