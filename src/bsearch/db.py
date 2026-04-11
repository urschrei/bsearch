from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sqlite_vec

if TYPE_CHECKING:
    from bsearch.models import Post

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT UNIQUE NOT NULL,
    cid TEXT NOT NULL,
    author_did TEXT NOT NULL,
    author_handle TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    has_embedding INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_posts_source ON posts(source);
CREATE INDEX IF NOT EXISTS idx_posts_has_embedding ON posts(has_embedding);
"""

VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_posts USING vec0(
    embedding float[384]
);
"""

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_posts USING fts5(
    text,
    content=posts,
    content_rowid=id,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
    INSERT INTO fts_posts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS posts_ad AFTER DELETE ON posts BEGIN
    INSERT INTO fts_posts(fts_posts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS posts_au AFTER UPDATE ON posts BEGIN
    INSERT INTO fts_posts(fts_posts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO fts_posts(rowid, text) VALUES (new.id, new.text);
END;
"""


class Database:
    """SQLite + sqlite-vec database layer."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self.conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA temp_store=memory")
        return conn

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(VEC_TABLE_SQL)
        self.conn.executescript(FTS_SCHEMA_SQL)
        self._ensure_fts_populated()
        self.conn.commit()

    def _ensure_fts_populated(self) -> None:
        """Rebuild the FTS index if it has never been initialised.

        For external-content FTS5 tables, SELECT count(*) reads from the
        content table rather than the index, so we cannot compare counts.
        Instead we track a metadata flag that is set after the first rebuild.
        """
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'fts_initialized'"
        ).fetchone()
        if row is not None:
            return
        posts_count = self.conn.execute("SELECT count(*) FROM posts").fetchone()[0]
        if posts_count > 0:
            logger.info("Rebuilding FTS index for %d posts...", posts_count)
            self.conn.execute("INSERT INTO fts_posts(fts_posts) VALUES('rebuild')")
            logger.info("FTS index rebuilt.")
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('fts_initialized', '1')"
        )

    def insert_post(self, post: Post) -> int | None:
        """Insert a post, returning its id. Returns None if it already exists."""
        try:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO posts
                    (uri, cid, author_did, author_handle, text, created_at, source, indexed_at, has_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    post.uri,
                    post.cid,
                    post.author_did,
                    post.author_handle,
                    post.text,
                    post.created_at.isoformat(),
                    post.source,
                    post.indexed_at.isoformat(),
                    int(post.has_embedding),
                ),
            )
            self.conn.commit()
            if cursor.rowcount == 0:
                return None
            return cursor.lastrowid
        except sqlite3.Error:
            self.conn.rollback()
            raise

    def get_posts_without_embeddings(self, limit: int = 100) -> list[tuple[int, str]]:
        """Return (id, text) pairs for posts that need embeddings."""
        rows = self.conn.execute(
            "SELECT id, text FROM posts WHERE has_embedding = 0 LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row["id"], row["text"]) for row in rows]

    def store_embeddings(self, embeddings: list[tuple[int, np.ndarray]]) -> None:
        """Store embeddings for the given post ids and mark them as embedded."""
        for post_id, embedding in embeddings:
            serialised = embedding.astype(np.float32).tobytes()
            self.conn.execute(
                "INSERT INTO vec_posts (rowid, embedding) VALUES (?, ?)",
                (post_id, serialised),
            )
            self.conn.execute(
                "UPDATE posts SET has_embedding = 1 WHERE id = ?",
                (post_id,),
            )
        self.conn.commit()

    def get_cursor(self) -> int | None:
        """Get the stored Jetstream cursor (microseconds timestamp)."""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'cursor'"
        ).fetchone()
        if row:
            return int(row["value"])
        return None

    def set_cursor(self, cursor: int) -> None:
        """Store the Jetstream cursor."""
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('cursor', ?)",
            (str(cursor),),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Return database statistics."""
        total = self.conn.execute("SELECT COUNT(*) AS c FROM posts").fetchone()["c"]
        with_embeddings = self.conn.execute(
            "SELECT COUNT(*) AS c FROM posts WHERE has_embedding = 1"
        ).fetchone()["c"]
        by_source = {}
        for row in self.conn.execute(
            "SELECT source, COUNT(*) AS c FROM posts GROUP BY source"
        ).fetchall():
            by_source[row["source"]] = row["c"]
        cursor = self.get_cursor()
        return {
            "total_posts": total,
            "with_embeddings": with_embeddings,
            "without_embeddings": total - with_embeddings,
            "by_source": by_source,
            "cursor": cursor,
        }

    def vacuum(self) -> None:
        """Reclaim unused space and defragment the database file."""
        self.conn.execute("VACUUM")

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
