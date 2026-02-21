from datetime import datetime

import numpy as np
import pytest

from bsearch.db import Database
from bsearch.models import Post


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db = Database(tmp_path / "test.db")
    yield db
    db.close()


def _make_post(uri="at://did:plc:abc/app.bsky.feed.post/123", **kwargs):
    defaults = {
        "uri": uri,
        "cid": "bafyreiabc",
        "author_did": "did:plc:abc",
        "author_handle": "test.bsky.social",
        "text": "Hello world",
        "created_at": datetime(2025, 1, 1, 12, 0, 0),
        "source": "own_post",
    }
    defaults.update(kwargs)
    return Post(**defaults)


class TestInsertPost:
    def test_insert_returns_id(self, db):
        post = _make_post()
        result = db.insert_post(post)
        assert result is not None
        assert result > 0

    def test_duplicate_uri_returns_none(self, db):
        post = _make_post()
        db.insert_post(post)
        result = db.insert_post(post)
        assert result is None

    def test_different_uris_insert_separately(self, db):
        id1 = db.insert_post(_make_post(uri="at://did:plc:abc/app.bsky.feed.post/1"))
        id2 = db.insert_post(_make_post(uri="at://did:plc:abc/app.bsky.feed.post/2"))
        assert id1 != id2


class TestEmbeddings:
    def test_posts_without_embeddings(self, db):
        db.insert_post(_make_post())
        pending = db.get_posts_without_embeddings()
        assert len(pending) == 1
        assert pending[0][1] == "Hello world"

    def test_store_and_mark_embedded(self, db):
        post_id = db.insert_post(_make_post())
        embedding = np.random.randn(384).astype(np.float32)
        db.store_embeddings([(post_id, embedding)])
        pending = db.get_posts_without_embeddings()
        assert len(pending) == 0


class TestSearch:
    def test_knn_search(self, db):
        # Insert two posts with different embeddings
        id1 = db.insert_post(_make_post(uri="at://a/1", text="cats are great"))
        id2 = db.insert_post(_make_post(uri="at://a/2", text="dogs are wonderful"))
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        db.store_embeddings([(id1, emb1), (id2, emb2)])

        # Search with an embedding close to emb1
        results = db.search(emb1, limit=2)
        assert len(results) == 2
        assert results[0]["id"] == id1  # closest match

    def test_source_filter(self, db):
        id1 = db.insert_post(
            _make_post(uri="at://a/1", text="own post", source="own_post")
        )
        id2 = db.insert_post(
            _make_post(uri="at://a/2", text="liked post", source="like")
        )
        emb = np.random.randn(384).astype(np.float32)
        db.store_embeddings([(id1, emb), (id2, emb)])

        results = db.search(emb, limit=10, source_filter="like")
        assert all(r["source"] == "like" for r in results)


class TestCursor:
    def test_cursor_none_by_default(self, db):
        assert db.get_cursor() is None

    def test_set_and_get_cursor(self, db):
        db.set_cursor(1234567890)
        assert db.get_cursor() == 1234567890

    def test_cursor_overwrites(self, db):
        db.set_cursor(100)
        db.set_cursor(200)
        assert db.get_cursor() == 200


class TestFTSSearch:
    def test_fts_finds_matching_text(self, db):
        db.insert_post(_make_post(uri="at://a/1", text="the cat sat on the mat"))
        db.insert_post(_make_post(uri="at://a/2", text="dogs playing in the park"))

        results = db.search_fts("cat", limit=10)
        assert len(results) == 1
        assert "cat" in results[0]["text"]

    def test_fts_no_results(self, db):
        db.insert_post(_make_post(uri="at://a/1", text="hello world"))
        results = db.search_fts("nonexistent", limit=10)
        assert len(results) == 0

    def test_fts_source_filter(self, db):
        db.insert_post(
            _make_post(uri="at://a/1", text="cats are great", source="own_post")
        )
        db.insert_post(
            _make_post(uri="at://a/2", text="cats are wonderful", source="like")
        )

        results = db.search_fts("cats", limit=10, source_filter="own_post")
        assert len(results) == 1
        assert results[0]["source"] == "own_post"

    def test_fts_porter_stemming(self, db):
        db.insert_post(_make_post(uri="at://a/1", text="running quickly"))
        results = db.search_fts("run", limit=10)
        assert len(results) == 1

    def test_fts_empty_query_returns_empty(self, db):
        db.insert_post(_make_post(uri="at://a/1", text="hello world"))
        assert db.search_fts("", limit=10) == []
        assert db.search_fts("   ", limit=10) == []

    def test_fts_special_characters_do_not_crash(self, db):
        db.insert_post(_make_post(uri="at://a/1", text="hello world"))
        # Special characters that would break FTS5 syntax should not raise
        results = db.search_fts("hello (world", limit=10)
        assert isinstance(results, list)

    def test_fts_multiple_results_ranked(self, db):
        db.insert_post(_make_post(uri="at://a/1", text="python programming language"))
        db.insert_post(
            _make_post(uri="at://a/2", text="python snake python reptile python")
        )
        db.insert_post(_make_post(uri="at://a/3", text="java programming"))

        results = db.search_fts("python", limit=10)
        assert len(results) == 2
        # Post with more occurrences of "python" should rank higher
        assert results[0]["uri"] == "at://a/2"


class TestHybridSearch:
    def test_hybrid_returns_results_from_both(self, db):
        id1 = db.insert_post(
            _make_post(uri="at://a/1", text="learning python programming")
        )
        id2 = db.insert_post(
            _make_post(uri="at://a/2", text="machine learning is fascinating")
        )

        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        db.store_embeddings([(id1, emb1), (id2, emb2)])

        results = db.search_hybrid("python", emb1, limit=10)
        assert len(results) >= 1
        # Post 1 matches both keyword ("python") and vector (emb1)
        assert results[0]["id"] == id1
        assert results[0]["match_type"] == "keyword+semantic"

    def test_hybrid_fts_only_fallback(self, db):
        db.insert_post(_make_post(uri="at://a/1", text="specific keyword here"))
        results = db.search_hybrid("keyword", query_embedding=None, limit=10)
        assert len(results) == 1
        assert results[0]["match_type"] == "keyword"

    def test_hybrid_vector_only_fallback(self, db):
        id1 = db.insert_post(_make_post(uri="at://a/1", text="hello world"))
        emb1 = np.random.randn(384).astype(np.float32)
        db.store_embeddings([(id1, emb1)])

        results = db.search_hybrid("zzzznonexistent", emb1, limit=10)
        assert len(results) >= 1

    def test_hybrid_deduplication(self, db):
        id1 = db.insert_post(_make_post(uri="at://a/1", text="unique test post"))
        emb1 = np.random.randn(384).astype(np.float32)
        db.store_embeddings([(id1, emb1)])

        results = db.search_hybrid("unique test", emb1, limit=10)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_hybrid_source_filter(self, db):
        id1 = db.insert_post(
            _make_post(uri="at://a/1", text="cats everywhere", source="own_post")
        )
        id2 = db.insert_post(
            _make_post(uri="at://a/2", text="cats are nice", source="like")
        )
        emb = np.random.randn(384).astype(np.float32)
        db.store_embeddings([(id1, emb), (id2, emb)])

        results = db.search_hybrid("cats", emb, limit=10, source_filter="own_post")
        assert all(r["source"] == "own_post" for r in results)

    def test_hybrid_empty_db(self, db):
        emb = np.random.randn(384).astype(np.float32)
        results = db.search_hybrid("anything", emb, limit=10)
        assert results == []


class TestStats:
    def test_empty_stats(self, db):
        stats = db.get_stats()
        assert stats["total_posts"] == 0
        assert stats["with_embeddings"] == 0
        assert stats["by_source"] == {}

    def test_stats_with_data(self, db):
        db.insert_post(_make_post(uri="at://a/1", source="own_post"))
        db.insert_post(_make_post(uri="at://a/2", source="like"))
        stats = db.get_stats()
        assert stats["total_posts"] == 2
        assert stats["by_source"]["own_post"] == 1
        assert stats["by_source"]["like"] == 1
