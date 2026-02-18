import numpy as np

from bsearch.embeddings import Embedder


class TestEmbedder:
    def test_encode_single(self):
        embedder = Embedder()
        result = embedder.encode_single("Hello world")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert result.dtype == np.float32

    def test_encode_batch(self):
        embedder = Embedder()
        texts = ["Hello world", "Goodbye world", "Testing embeddings"]
        result = embedder.encode(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_encode_empty(self):
        embedder = Embedder()
        result = embedder.encode([])
        assert len(result) == 0

    def test_similar_texts_closer(self):
        embedder = Embedder()
        cat_emb = embedder.encode_single("I love cats")
        dog_emb = embedder.encode_single("I love dogs")
        weather_emb = embedder.encode_single("The stock market crashed today")

        # Cosine similarity: cat/dog should be more similar than cat/weather
        cat_dog = np.dot(cat_emb, dog_emb) / (
            np.linalg.norm(cat_emb) * np.linalg.norm(dog_emb)
        )
        cat_weather = np.dot(cat_emb, weather_emb) / (
            np.linalg.norm(cat_emb) * np.linalg.norm(weather_emb)
        )
        assert cat_dog > cat_weather

    def test_lazy_loading(self):
        embedder = Embedder()
        assert embedder._model is None
        embedder.encode_single("trigger loading")
        assert embedder._model is not None
