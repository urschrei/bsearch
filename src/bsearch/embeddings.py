from __future__ import annotations

import numpy as np


class Embedder:
    """Lazy-loaded sentence-transformer wrapper for generating embeddings."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", quiet: bool = False
    ) -> None:
        self.model_name = model_name
        self.quiet = quiet
        self._model = None

    def _load_model(self):
        if self.quiet:
            import logging as _logging
            import os

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TQDM_DISABLE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["SAFETENSORS_LOG_LEVEL"] = "error"
            for name in (
                "sentence_transformers",
                "huggingface_hub",
                "huggingface_hub.utils._http",
                "httpx",
                "safetensors",
            ):
                _logging.getLogger(name).setLevel(_logging.ERROR)

        from sentence_transformers import SentenceTransformer

        if self.quiet:
            import os
            import warnings

            # Redirect OS-level file descriptors to suppress C-level output
            # (e.g. safetensors LOAD REPORT)
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stdout = os.dup(1)
            old_stderr = os.dup(2)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._model = SentenceTransformer(self.model_name)
            finally:
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
                os.close(old_stdout)
                os.close(old_stderr)
        else:
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
        return self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=not self.quiet
        ).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector."""
        return self.encode([text])[0]
