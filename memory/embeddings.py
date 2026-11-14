"""Embeddings: text turned into vectors, so memory can be searched by meaning.

An embedding model maps a sentence to a point in a 384-dimensional space,
placed so that sentences that mean similar things land close together.
"Can I eat a Snickers bar?" and "Aryan is allergic to peanuts" share no
words, but their vectors point in similar directions, which word search can
never see. Closeness is measured with cosine similarity: the dot product of
two length-normalised vectors, from -1 (opposite) to 1 (same direction).

The model (BAAI/bge-small-en-v1.5, via fastembed) runs locally on the CPU
through ONNX: no API key, no per-call cost, nothing leaves the machine. It
is downloaded once (~70 MB) into the models folder and loaded lazily, on
the first search or fact that needs it, so starting sora stays fast.

bge models embed questions and documents slightly differently ("query" vs
"passage"), which is why there are two methods.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class Embedder:
    def __init__(self, cache_dir: Path, model: str = DEFAULT_MODEL):
        self.cache_dir = cache_dir
        self.model_name = model
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding      # imported late: it's slow to import
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                log.info("loading embedding model %s", self.model_name)
                self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def documents(self, texts: list[str]) -> np.ndarray:
        """Vectors for things stored in memory (facts, episodes)."""
        return normalise(list(self._load().passage_embed(texts)))

    def query(self, text: str) -> np.ndarray:
        """The vector for a search."""
        return normalise(list(self._load().query_embed([text])))[0]
