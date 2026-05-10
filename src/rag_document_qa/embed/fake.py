"""Hash-based deterministic embedder for tests + CI.

No model download, no GPU, no network. The vectors are not semantically
meaningful, but they're deterministic and well-distributed enough that tests
which depend on "the same text gets the same vector" and "different texts get
different vectors" pass reliably.

Senior-engineer note: this is a TEST DOUBLE, not a baseline. Don't compare
recall numbers between FakeEmbedder and a real embedder; they live in
different vector spaces.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


class FakeEmbedder:
    """Deterministic hash-derived embedder.

    Produces L2-normalised float32 vectors of `dim` dimensions. Same text in
    always produces the same vector. Different texts produce different vectors
    with high probability (the 32-byte sha256 seed is wide enough).
    """

    def __init__(self, dim: int = 16) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim

    @property
    def name(self) -> str:
        return "fake-hash-embedder"

    @property
    def version(self) -> str:
        return "1"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows: list[NDArray[np.float32]] = []
        for text in texts:
            seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            # Seed a numpy RNG from the first 4 bytes; gives reproducible draws
            # without depending on global state.
            seed = int.from_bytes(seed_bytes[:4], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            rows.append(v)
        return np.vstack(rows).astype(np.float32)
