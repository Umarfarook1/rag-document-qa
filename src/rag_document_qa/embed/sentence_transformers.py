"""SentenceTransformers-backed embedder.

Default model is `BAAI/bge-small-en-v1.5` (384-dim, fast, MIT-licensed). Other
models from the sentence-transformers hub work as long as they expose
`encode()` returning float arrays.

This module imports `sentence_transformers` lazily so the package can still be
imported in environments that haven't installed the `[embed]` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rag_document_qa.errors import RagError


class SentenceTransformersEmbedder:
    """Wraps a sentence-transformers model into the `Embedder` Protocol."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RagError(
                code="loader_failed",
                message=(
                    "sentence-transformers not installed; run `pip install rag-document-qa[embed]`"
                ),
            ) from e

        self._model_name = model_name or self.DEFAULT_MODEL
        self._batch_size = batch_size
        self._model: Any = SentenceTransformer(self._model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        # SentenceTransformers doesn't expose a version per model; fall back to
        # the model name's tail (e.g. "v1.5" for bge-small-en-v1.5).
        tail = self._model_name.rsplit("-", 1)[-1]
        return tail if tail else "unknown"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)
