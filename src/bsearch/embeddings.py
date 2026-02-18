from __future__ import annotations

import numpy as np


class Embedder:
    """Lazy-loaded sentence-transformer wrapper for generating embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings.

        Returns an array of shape (len(texts), dimensions).
        """
        if not texts:
            return np.array([], dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector."""
        return self.encode([text])[0]
