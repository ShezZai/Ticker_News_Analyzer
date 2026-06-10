import numpy as np
import pytest

from ticker_news.embedding.embedder import (
    MAX_INPUT_TOKENS,
    build_text,
    embed_query,
    embed_texts,
    truncate_tokens,
)


class FakeEmbeddings:
    """Stands in for OpenAIEmbeddings: returns a constant unit vector per text."""

    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.seen.append(text)
        return [0.0, 1.0, 0.0]


def test_truncate_passes_short_text_through():
    assert truncate_tokens("hello world") == "hello world"


def test_truncate_empty_becomes_single_space():
    assert truncate_tokens("") == " "
    assert truncate_tokens("   ") == " "


def test_truncate_caps_long_text():
    long = "word " * (MAX_INPUT_TOKENS * 2)
    out = truncate_tokens(long)
    assert len(out) < len(long)
    assert len(out.split()) <= MAX_INPUT_TOKENS


def test_build_text_joins_title_and_content():
    assert build_text("Title", "Body") == "Title\n\nBody"
    assert build_text(None, "Body") == "Body"
    assert build_text("Title", None) == "Title"
    assert build_text(None, None) == ""


def test_embed_texts_returns_float32_arrays_aligned():
    fake = FakeEmbeddings()
    out = embed_texts(["a", "", "c"], embeddings=fake)
    assert len(out) == 3
    assert all(isinstance(v, np.ndarray) and v.dtype == np.float32 for v in out)
    assert fake.seen[1] == " "  # empty input replaced, alignment preserved


def test_embed_query_rejects_empty():
    with pytest.raises(ValueError):
        embed_query("   ", embeddings=FakeEmbeddings())


def test_embed_query_returns_vector():
    v = embed_query("nvidia datacenter", embeddings=FakeEmbeddings())
    assert isinstance(v, np.ndarray) and v.dtype == np.float32
