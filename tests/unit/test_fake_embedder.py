import numpy as np
import pytest

from rag_document_qa.embed.fake import FakeEmbedder
from rag_document_qa.protocols import Embedder


def test_implements_protocol() -> None:
    e = FakeEmbedder(dim=8)
    assert isinstance(e, Embedder)


def test_name_version_dim() -> None:
    e = FakeEmbedder(dim=32)
    assert e.name == "fake-hash-embedder"
    assert e.version == "1"
    assert e.dim == 32


def test_encode_returns_correct_shape() -> None:
    e = FakeEmbedder(dim=16)
    out = e.encode(["a", "b", "c"])
    assert out.shape == (3, 16)
    assert out.dtype == np.float32


def test_encode_empty_returns_empty_2d() -> None:
    e = FakeEmbedder(dim=16)
    out = e.encode([])
    assert out.shape == (0, 16)


def test_encode_is_deterministic() -> None:
    e = FakeEmbedder(dim=16)
    a = e.encode(["hello world"])
    b = e.encode(["hello world"])
    np.testing.assert_array_equal(a, b)


def test_different_texts_get_different_vectors() -> None:
    e = FakeEmbedder(dim=16)
    out = e.encode(["alpha", "beta"])
    # Cosine should be < 1.0 for different inputs (very high probability with sha256 seed).
    cos = float(out[0] @ out[1])
    assert cos < 0.99


def test_encoded_vectors_are_unit_norm() -> None:
    e = FakeEmbedder(dim=16)
    out = e.encode(["a", "b", "c"])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_zero_dim_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        FakeEmbedder(dim=0)


def test_negative_dim_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        FakeEmbedder(dim=-3)
