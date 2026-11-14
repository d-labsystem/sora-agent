"""Shared test setup: no test ever downloads or runs the real embedding model."""

import re
import zlib

import numpy as np
import pytest

# Words that mean the same thing to the fake model, so tests can check
# meaning-based recall without the real one.
CONCEPTS = {"snickers": "peanut", "peanuts": "peanut", "nut": "peanut", "nuts": "peanut",
            "boss": "manager", "manager": "manager", "sibling": "sister", "sister": "sister"}


class FakeEmbedder:
    """Bag of concepts hashed into 64 dimensions, then normalised."""
    model_name = "fake-embedder"

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def _vector(self, text):
        v = np.zeros(64, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            word = CONCEPTS.get(word, word)
            v[zlib.crc32(word.encode()) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def documents(self, texts):
        self.calls += 1
        return np.array([self._vector(t) for t in texts])

    def query(self, text):
        return self._vector(text)


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    import harness.app
    monkeypatch.setattr(harness.app, "Embedder", FakeEmbedder)
