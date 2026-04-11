from pathlib import Path

import numpy as np
import pytest

from bsearch.embeddings import Embedder


@pytest.fixture
def onnx_model_dir():
    """Check the ONNX model has been exported."""
    model_dir = Path.home() / ".cache" / "bsearch" / "all-MiniLM-L6-v2"
    if not (model_dir / "model.onnx").exists():
        pytest.skip("ONNX model not exported. Run `bsearch export-model` first.")
    return model_dir


class TestOnnxParity:
    def test_embedding_parity(self, onnx_model_dir):
        """Verify ONNX model produces same embeddings as SentenceTransformer."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        embedder = Embedder()

        test_texts = [
            "Hello world",
            "I love cats and dogs",
            "The stock market crashed today",
        ]

        python_embeddings = embedder.encode(test_texts)

        # Load ONNX model directly in Python for comparison
        session = ort.InferenceSession(str(onnx_model_dir / "model.onnx"))
        tokenizer = AutoTokenizer.from_pretrained(str(onnx_model_dir))

        for i, text in enumerate(test_texts):
            encoded = tokenizer(text, return_tensors="np")
            outputs = session.run(
                None,
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "token_type_ids": encoded["token_type_ids"],
                },
            )
            hidden_state = outputs[0]  # (1, seq_len, 384)
            attention_mask = encoded["attention_mask"].astype(np.float32)

            # Mean pool
            mask_expanded = np.expand_dims(attention_mask, -1)
            summed = np.sum(hidden_state * mask_expanded, axis=1)
            counts = np.sum(mask_expanded, axis=1)
            pooled = summed / counts

            # L2 normalise
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalised = pooled / norm

            np.testing.assert_allclose(
                normalised[0],
                python_embeddings[i],
                atol=1e-4,
                err_msg=f"Embedding mismatch for text: {text}",
            )
