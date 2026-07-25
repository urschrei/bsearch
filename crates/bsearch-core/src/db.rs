use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::Once;

use anyhow::Context;
use anyhow::Result;
use rusqlite::Connection;
use rusqlite::OpenFlags;
use rusqlite::OptionalExtension;

use crate::models::Post;

/// Schema statements, kept byte-for-byte in step with `src/bsearch/db.py`.
/// Both the daemon and the Python CLI may be the first to touch a fresh
/// database, so they must agree on exactly what they create.
const SCHEMA_SQL: &str = "
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
";

const VEC_TABLE_SQL: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS vec_posts USING vec0(
    embedding float[384]
);
";

const FTS_SCHEMA_SQL: &str = "
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
";

/// Describes which retrieval method(s) contributed to a search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    Keyword,
    Semantic,
    KeywordAndSemantic,
}

impl fmt::Display for MatchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keyword => write!(f, "keyword"),
            Self::Semantic => write!(f, "semantic"),
            Self::KeywordAndSemantic => write!(f, "keyword+semantic"),
        }
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub id: i64,
    pub uri: String,
    pub cid: String,
    pub author_did: String,
    pub author_handle: String,
    pub text: String,
    pub created_at: String,
    pub source: String,
    pub indexed_at: String,
    pub distance: Option<f64>,
    pub bm25_rank: Option<f64>,
    pub rrf_score: Option<f64>,
    pub match_type: Option<MatchType>,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open the database read-only, for search.
    pub fn open(path: &Path) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        Ok(Self { conn })
    }

    /// Open the database read-write and ensure the schema exists, for ingest.
    ///
    /// The pragmas mirror `Database._connect` in `src/bsearch/db.py` so that
    /// the daemon and the Python CLI behave identically against the same file.
    pub fn open_read_write(path: &Path) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "temp_store", "memory")?;
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;

        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn
            .execute_batch(SCHEMA_SQL)
            .context("failed to create base schema")?;
        self.conn
            .execute_batch(VEC_TABLE_SQL)
            .context("failed to create vec_posts table")?;
        self.conn
            .execute_batch(FTS_SCHEMA_SQL)
            .context("failed to create FTS schema")?;
        self.ensure_fts_populated()?;
        Ok(())
    }

    /// Port of `Database._ensure_fts_populated` in `src/bsearch/db.py`.
    ///
    /// For an external-content FTS5 table, `count(*)` reads through to the
    /// content table rather than the index, so the two cannot be compared.
    /// Python instead records a flag once the index is known to be good; we
    /// honour the same flag, both so a Rust-created database does not provoke
    /// a needless rebuild the next time the Python CLI opens it, and so we
    /// still repair an older database that predates the flag.
    fn ensure_fts_populated(&self) -> Result<()> {
        let initialised: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'fts_initialized'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if initialised.is_some() {
            return Ok(());
        }

        let posts_count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM posts", [], |row| row.get(0))?;
        if posts_count > 0 {
            self.conn
                .execute("INSERT INTO fts_posts(fts_posts) VALUES('rebuild')", [])
                .context("failed to rebuild FTS index")?;
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('fts_initialized', '1')",
            [],
        )?;
        Ok(())
    }

    /// Insert a post, returning its rowid, or `None` if the URI already exists.
    ///
    /// The FTS index is maintained by the `posts_ai` trigger, so no separate
    /// write is needed here.
    pub fn insert_post(&self, post: &Post) -> Result<Option<i64>> {
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO posts
                 (uri, cid, author_did, author_handle, text, created_at, source, indexed_at, has_embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0)",
            rusqlite::params![
                post.uri,
                post.cid,
                post.author_did,
                post.author_handle,
                post.text,
                post.created_at,
                post.source,
                post.indexed_at,
            ],
        )?;

        if changed == 0 {
            return Ok(None);
        }
        Ok(Some(self.conn.last_insert_rowid()))
    }

    /// Return `(id, text)` for posts that still need an embedding.
    pub fn get_posts_without_embeddings(&self, limit: usize) -> Result<Vec<(i64, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, text FROM posts WHERE has_embedding = 0 LIMIT ?1")?;
        let rows = stmt.query_map([limit as i64], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Store embeddings and flag the corresponding posts as embedded.
    ///
    /// Runs as a single transaction so that `vec_posts` and `posts.has_embedding`
    /// cannot disagree if the process dies mid-batch.
    pub fn store_embeddings(&mut self, embeddings: &[(i64, [f32; 384])]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for (post_id, embedding) in embeddings {
            let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
            tx.execute(
                "INSERT INTO vec_posts (rowid, embedding) VALUES (?1, ?2)",
                rusqlite::params![post_id, bytes],
            )?;
            tx.execute(
                "UPDATE posts SET has_embedding = 1 WHERE id = ?1",
                [post_id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Read the stored Jetstream cursor (microseconds since the epoch).
    pub fn get_cursor(&self) -> Result<Option<i64>> {
        let value: Option<String> = self
            .conn
            .query_row("SELECT value FROM meta WHERE key = 'cursor'", [], |row| {
                row.get(0)
            })
            .optional()?;
        // The Python code stores this as text; tolerate a malformed value
        // rather than refusing to start.
        Ok(value.and_then(|v| v.parse::<i64>().ok()))
    }

    /// Persist the Jetstream cursor.
    pub fn set_cursor(&self, cursor: i64) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('cursor', ?1)",
            [cursor.to_string()],
        )?;
        Ok(())
    }

    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        match self.run_fts_query(query, limit, source_filter, handle_filter) {
            Ok(results) => Ok(results),
            Err(e) => {
                // FTS5 syntax errors manifest as generic sqlite errors;
                // retry the query wrapped as a phrase literal.
                let phrase = format!("\"{}\"", query.replace('"', ""));
                match self.run_fts_query(&phrase, limit, source_filter, handle_filter) {
                    Ok(results) => Ok(results),
                    Err(retry_err) => {
                        // If both attempts fail, propagate the original error
                        // as it is more likely to be informative.
                        Err(retry_err)
                            .with_context(|| format!("FTS query failed (original error: {e})"))
                    }
                }
            }
        }
    }

    fn run_fts_query(
        &self,
        query: &str,
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let mut conditions = vec!["fts_posts MATCH ?1".to_string()];
        let mut param_idx = 2usize;

        if source_filter.is_some() {
            conditions.push(format!("p.source = ?{}", param_idx));
            param_idx += 1;
        }
        if handle_filter.is_some() {
            conditions.push(format!("p.author_handle = ?{}", param_idx));
            param_idx += 1;
        }

        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT p.id, fts_posts.rank AS bm25_rank,
                    p.uri, p.cid, p.author_did, p.author_handle, p.text,
                    p.created_at, p.source, p.indexed_at
             FROM fts_posts
             INNER JOIN posts p ON p.id = fts_posts.rowid
             WHERE {}
             ORDER BY fts_posts.rank
             LIMIT ?{}",
            where_clause, param_idx
        );

        let mut stmt = self.conn.prepare(&sql)?;

        // Build params dynamically
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(query.to_string())];
        if let Some(s) = source_filter {
            params.push(Box::new(s.to_string()));
        }
        if let Some(h) = handle_filter {
            params.push(Box::new(h.to_string()));
        }
        params.push(Box::new(limit as i64));

        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let results = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(SearchResult {
                id: row.get(0)?,
                bm25_rank: row.get(1)?,
                uri: row.get(2)?,
                cid: row.get(3)?,
                author_did: row.get(4)?,
                author_handle: row.get(5)?,
                text: row.get(6)?,
                created_at: row.get(7)?,
                source: row.get(8)?,
                indexed_at: row.get(9)?,
                distance: None,
                rrf_score: None,
                match_type: Some(MatchType::Keyword),
            })
        })?;

        results
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn search_vec(
        &self,
        query_embedding: &[f32; 384],
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        // Over-fetch when filters are active so we have enough results after filtering
        let fetch_limit = if source_filter.is_some() || handle_filter.is_some() {
            limit * 5
        } else {
            limit
        };

        let bytes: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let sql = "SELECT v.rowid AS id, v.distance, p.uri, p.cid, p.author_did, p.author_handle,
                          p.text, p.created_at, p.source, p.indexed_at
                   FROM vec_posts v
                   INNER JOIN posts p ON p.id = v.rowid
                   WHERE v.embedding MATCH ?1 AND k = ?2
                   ORDER BY v.distance";

        let mut stmt = self.conn.prepare(sql)?;

        let mut results: Vec<SearchResult> = stmt
            .query_map(rusqlite::params![bytes, fetch_limit as i64], |row| {
                Ok(SearchResult {
                    id: row.get(0)?,
                    distance: row.get(1)?,
                    uri: row.get(2)?,
                    cid: row.get(3)?,
                    author_did: row.get(4)?,
                    author_handle: row.get(5)?,
                    text: row.get(6)?,
                    created_at: row.get(7)?,
                    source: row.get(8)?,
                    indexed_at: row.get(9)?,
                    bm25_rank: None,
                    rrf_score: None,
                    match_type: Some(MatchType::Semantic),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        if let Some(s) = source_filter {
            results.retain(|r| r.source == s);
        }
        if let Some(h) = handle_filter {
            results.retain(|r| r.author_handle == h);
        }
        results.truncate(limit);

        Ok(results)
    }

    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: Option<&[f32; 384]>,
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
        max_semantic_distance: f64,
    ) -> Result<Vec<SearchResult>> {
        let fetch_limit = limit * 3;

        let fts_results = self.search_fts(query, fetch_limit, source_filter, handle_filter)?;
        let vec_results = if let Some(emb) = query_embedding {
            self.search_vec(emb, fetch_limit, source_filter, handle_filter)?
        } else {
            vec![]
        };

        // Fallback: only one source has results
        if fts_results.is_empty() && !vec_results.is_empty() {
            let mut out = vec_results;
            for r in &mut out {
                r.match_type = Some(MatchType::Semantic);
            }
            out.truncate(limit);
            return Ok(out);
        }
        if !fts_results.is_empty() && vec_results.is_empty() {
            let mut out = fts_results;
            for r in &mut out {
                r.match_type = Some(MatchType::Keyword);
            }
            out.truncate(limit);
            return Ok(out);
        }
        if fts_results.is_empty() && vec_results.is_empty() {
            return Ok(vec![]);
        }

        // Both have results: compute RRF scores
        // k=60 is the standard RRF constant
        const K: f64 = 60.0;

        let mut rrf_scores: BTreeMap<i64, f64> = BTreeMap::new();
        let mut has_keyword: BTreeMap<i64, bool> = BTreeMap::new();
        let mut has_semantic: BTreeMap<i64, bool> = BTreeMap::new();
        let mut vec_distance: BTreeMap<i64, f64> = BTreeMap::new();
        // Keep one representative SearchResult per doc id
        let mut result_map: BTreeMap<i64, SearchResult> = BTreeMap::new();

        for (rank, r) in fts_results.into_iter().enumerate() {
            let score = 1.0 / (K + (rank + 1) as f64);
            *rrf_scores.entry(r.id).or_insert(0.0) += score;
            has_keyword.insert(r.id, true);
            result_map.entry(r.id).or_insert(r);
        }

        for (rank, r) in vec_results.into_iter().enumerate() {
            let score = 1.0 / (K + (rank + 1) as f64);
            *rrf_scores.entry(r.id).or_insert(0.0) += score;
            has_semantic.insert(r.id, true);
            if let Some(d) = r.distance {
                vec_distance.insert(r.id, d);
            }
            result_map.entry(r.id).or_insert(r);
        }

        // Include a result if it has a keyword match, OR if it is a
        // semantic-only match whose embedding distance is below the threshold.
        // This lets semantic search surface genuinely close results that use
        // different words, while filtering out the noise that appears when
        // the embedding is too vague to discriminate.
        let candidate_ids: Vec<i64> = rrf_scores
            .keys()
            .filter(|id| {
                let kw = has_keyword.contains_key(id);
                let sem = has_semantic.contains_key(id);
                if kw {
                    return true;
                }
                if sem {
                    if let Some(&d) = vec_distance.get(id) {
                        return d <= max_semantic_distance;
                    }
                }
                false
            })
            .copied()
            .collect();

        // Sort by RRF score descending
        let mut scored: Vec<(i64, f64)> = candidate_ids
            .into_iter()
            .map(|id| (id, rrf_scores[&id]))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let mut out = Vec::with_capacity(scored.len());
        for (id, score) in scored {
            if let Some(mut r) = result_map.remove(&id) {
                r.rrf_score = Some(score);
                let kw = has_keyword.contains_key(&id);
                let sem = has_semantic.contains_key(&id);
                r.match_type = Some(match (kw, sem) {
                    (true, true) => MatchType::KeywordAndSemantic,
                    (true, false) => MatchType::Keyword,
                    (false, true) => MatchType::Semantic,
                    (false, false) => unreachable!(),
                });
                out.push(r);
            }
        }

        Ok(out)
    }

    pub fn list_by_handle(
        &self,
        handle: &str,
        limit: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let sql = if source_filter.is_some() {
            "SELECT p.id, p.uri, p.cid, p.author_did, p.author_handle, p.text,
                    p.created_at, p.source, p.indexed_at
             FROM posts p
             WHERE p.author_handle = ?1 AND p.source = ?2
             ORDER BY p.created_at DESC
             LIMIT ?3"
        } else {
            "SELECT p.id, p.uri, p.cid, p.author_did, p.author_handle, p.text,
                    p.created_at, p.source, p.indexed_at
             FROM posts p
             WHERE p.author_handle = ?1
             ORDER BY p.created_at DESC
             LIMIT ?2"
        };

        let mut stmt = self.conn.prepare(sql)?;

        let map_row = |row: &rusqlite::Row<'_>| {
            Ok(SearchResult {
                id: row.get(0)?,
                uri: row.get(1)?,
                cid: row.get(2)?,
                author_did: row.get(3)?,
                author_handle: row.get(4)?,
                text: row.get(5)?,
                created_at: row.get(6)?,
                source: row.get(7)?,
                indexed_at: row.get(8)?,
                distance: None,
                bm25_rank: None,
                rrf_score: None,
                match_type: None,
            })
        };

        let results = if let Some(s) = source_filter {
            stmt.query_map(rusqlite::params![handle, s, limit as i64], map_row)?
        } else {
            stmt.query_map(rusqlite::params![handle, limit as i64], map_row)?
        };

        results
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }
}

/// Register the sqlite-vec extension as an auto-extension so that every
/// new `Connection` automatically has vector search available.  The
/// registration is performed only once per process.
fn register_sqlite_vec() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        type AutoExtFn = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut ::std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> ::std::os::raw::c_int;
        // SAFETY: `sqlite3_vec_init` has the same ABI as `AutoExtFn` (the
        // sqlite3 auto-extension entry point signature). The transmute
        // through `*const ()` is required because Rust's type system cannot
        // express the C function pointer equivalence directly. This is the
        // pattern recommended by the sqlite-vec crate documentation.
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<*const (), AutoExtFn>(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::path::{Path, PathBuf};
    use tempfile::NamedTempFile;

    fn create_test_db() -> (NamedTempFile, PathBuf) {
        let file = NamedTempFile::new().expect("failed to create temp file");
        let path = file.path().to_path_buf();

        register_sqlite_vec();
        let conn = Connection::open(&path).expect("failed to open db");
        conn.execute_batch(
            "CREATE TABLE posts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                uri         TEXT NOT NULL DEFAULT '',
                cid         TEXT NOT NULL DEFAULT '',
                author_did  TEXT NOT NULL DEFAULT '',
                author_handle TEXT NOT NULL DEFAULT '',
                text        TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT '',
                source      TEXT NOT NULL DEFAULT '',
                indexed_at  TEXT NOT NULL DEFAULT ''
            );

            CREATE VIRTUAL TABLE fts_posts USING fts5(
                text,
                content=posts,
                content_rowid=id,
                tokenize='porter unicode61'
            );

            CREATE TRIGGER posts_ai AFTER INSERT ON posts BEGIN
                INSERT INTO fts_posts(rowid, text)
                VALUES (new.id, new.text);
            END;",
        )
        .expect("failed to create schema");

        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_posts USING vec0(embedding float[384])",
            [],
        )
        .expect("failed to create vec_posts table");

        (file, path)
    }

    fn insert_post(path: &Path, uri: &str, text: &str, source: &str, handle: &str) {
        let conn = Connection::open(path).expect("failed to open db");
        conn.execute(
            "INSERT INTO posts (uri, cid, author_did, author_handle, text, created_at, source, indexed_at)
             VALUES (?1, '', '', ?2, ?3, '2024-01-01T00:00:00Z', ?4, '2024-01-01T00:00:00Z')",
            rusqlite::params![uri, handle, text, source],
        )
        .expect("failed to insert post");
    }

    fn insert_embedding(path: &Path, rowid: i64, embedding: &[f32; 384]) {
        register_sqlite_vec();
        let conn = Connection::open(path).expect("failed to open db");
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        conn.execute(
            "INSERT INTO vec_posts (rowid, embedding) VALUES (?1, ?2)",
            rusqlite::params![rowid, bytes],
        )
        .expect("failed to insert embedding");
    }

    #[test]
    fn test_fts_finds_match() {
        let (_file, path) = create_test_db();
        insert_post(
            &path,
            "uri:1",
            "hello world",
            "bluesky",
            "alice.bsky.social",
        );
        insert_post(&path, "uri:2", "goodbye moon", "bluesky", "bob.bsky.social");

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("hello", 10, None, None)
            .expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "uri:1");
    }

    #[test]
    fn test_fts_empty_query() {
        let (_file, path) = create_test_db();
        insert_post(&path, "uri:1", "some text", "bluesky", "alice.bsky.social");

        let db = Database::open(&path).expect("open failed");

        let results = db.search_fts("", 10, None, None).expect("search failed");
        assert!(results.is_empty());

        let results = db.search_fts("   ", 10, None, None).expect("search failed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts_source_filter() {
        let (_file, path) = create_test_db();
        insert_post(
            &path,
            "uri:1",
            "hello world",
            "bluesky",
            "alice.bsky.social",
        );
        insert_post(&path, "uri:2", "hello world", "mastodon", "bob.social");

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("hello", 10, Some("bluesky"), None)
            .expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "bluesky");
    }

    #[test]
    fn test_fts_handle_filter() {
        let (_file, path) = create_test_db();
        insert_post(
            &path,
            "uri:1",
            "hello world",
            "bluesky",
            "alice.bsky.social",
        );
        insert_post(&path, "uri:2", "hello world", "bluesky", "bob.bsky.social");

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("hello", 10, None, Some("alice.bsky.social"))
            .expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].author_handle, "alice.bsky.social");
    }

    #[test]
    fn test_fts_special_chars_do_not_crash() {
        let (_file, path) = create_test_db();
        insert_post(
            &path,
            "uri:1",
            "hello world",
            "bluesky",
            "alice.bsky.social",
        );

        let db = Database::open(&path).expect("open failed");
        // Broken FTS5 syntax: unmatched quote and stray operators
        let results = db
            .search_fts("AND OR ( \" broken***", 10, None, None)
            .expect("should not crash");
        // We just verify it doesn't panic; result may be empty
        let _ = results;
    }

    #[test]
    fn test_list_by_handle() {
        let (_file, path) = create_test_db();
        insert_post(&path, "uri:1", "post one", "bluesky", "alice.bsky.social");
        insert_post(&path, "uri:2", "post two", "mastodon", "alice.bsky.social");
        insert_post(&path, "uri:3", "post three", "bluesky", "bob.bsky.social");

        let db = Database::open(&path).expect("open failed");

        // All posts by alice
        let results = db
            .list_by_handle("alice.bsky.social", 10, None)
            .expect("list failed");
        assert_eq!(results.len(), 2);

        // Only bluesky posts by alice
        let results = db
            .list_by_handle("alice.bsky.social", 10, Some("bluesky"))
            .expect("list failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "bluesky");

        // Bob has one post
        let results = db
            .list_by_handle("bob.bsky.social", 10, None)
            .expect("list failed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_hybrid_both_sources() {
        let (_file, path) = create_test_db();
        // Post 1: matches by keyword "rustacean" AND will be the closest vector
        insert_post(
            &path,
            "uri:1",
            "rustacean programming language",
            "bluesky",
            "alice.bsky.social",
        );
        // Post 2: distinct content, different embedding
        insert_post(
            &path,
            "uri:2",
            "cooking recipes and food",
            "bluesky",
            "bob.bsky.social",
        );

        // Insert embeddings: post 1 gets rowid 1, post 2 gets rowid 2
        // Make post 1's embedding close to the query and post 2's distant
        let mut emb1 = [0.0f32; 384];
        emb1[0] = 1.0;
        let mut emb2 = [0.0f32; 384];
        emb2[0] = -1.0;
        insert_embedding(&path, 1, &emb1);
        insert_embedding(&path, 2, &emb2);

        // Query embedding close to emb1
        let mut query_emb = [0.0f32; 384];
        query_emb[0] = 0.9;

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_hybrid("rustacean", Some(&query_emb), 10, None, None, 1.05)
            .expect("hybrid search failed");

        assert!(!results.is_empty(), "should have results");
        // Post 1 should be first: it matches both keyword and vector
        assert_eq!(results[0].uri, "uri:1");
        assert_eq!(
            results[0].match_type,
            Some(MatchType::KeywordAndSemantic),
            "post 1 should match both sources"
        );
    }

    #[test]
    fn test_hybrid_fts_only_fallback() {
        let (_file, path) = create_test_db();
        insert_post(
            &path,
            "uri:1",
            "hello world unique text",
            "bluesky",
            "alice.bsky.social",
        );
        // No embeddings inserted

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_hybrid("hello", None, 10, None, None, 1.05)
            .expect("hybrid search failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "uri:1");
        assert_eq!(results[0].match_type, Some(MatchType::Keyword));
    }

    /// A post with the given URI and text, with all other fields stubbed.
    fn sample_post(uri: &str, text: &str) -> Post {
        Post::new(
            uri.to_string(),
            "cid1".to_string(),
            "did:plc:abc".to_string(),
            "alice.bsky.social".to_string(),
            text.to_string(),
            "2026-03-29T03:11:21.467000+00:00".to_string(),
            crate::models::Source::OwnPost,
        )
    }

    fn empty_db() -> (NamedTempFile, PathBuf) {
        let file = NamedTempFile::new().expect("failed to create temp file");
        let path = file.path().to_path_buf();
        (file, path)
    }

    #[test]
    fn test_open_read_write_creates_schema() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");
        // Opening a second time must be a no-op, not an error.
        drop(db);
        Database::open_read_write(&path).expect("reopen failed");
    }

    /// The daemon and the Python CLI write to the same file, and either may be
    /// the one to create it. This pins the objects we create to exactly the set
    /// `src/bsearch/db.py` produces for a fresh database.
    #[test]
    fn test_schema_matches_python() {
        let (_file, path) = empty_db();
        Database::open_read_write(&path).expect("open failed");

        let conn = Connection::open(&path).expect("failed to reopen");
        let mut stmt = conn
            .prepare("SELECT type || ' ' || name FROM sqlite_master ORDER BY type, name")
            .expect("prepare failed");
        let objects: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .expect("query failed")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("collect failed");

        let expected = vec![
            "index idx_posts_has_embedding",
            "index idx_posts_source",
            "index sqlite_autoindex_meta_1",
            "index sqlite_autoindex_posts_1",
            "index sqlite_autoindex_vec_posts_info_1",
            "index sqlite_autoindex_vec_posts_vector_chunks00_1",
            "table fts_posts",
            "table fts_posts_config",
            "table fts_posts_data",
            "table fts_posts_docsize",
            "table fts_posts_idx",
            "table meta",
            "table posts",
            "table sqlite_sequence",
            "table vec_posts",
            "table vec_posts_chunks",
            "table vec_posts_info",
            "table vec_posts_rowids",
            "table vec_posts_vector_chunks00",
            "trigger posts_ad",
            "trigger posts_ai",
            "trigger posts_au",
        ];
        assert_eq!(objects, expected);
    }

    #[test]
    fn test_fresh_db_marks_fts_initialised() {
        let (_file, path) = empty_db();
        Database::open_read_write(&path).expect("open failed");

        let conn = Connection::open(&path).expect("failed to reopen");
        let flag: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'fts_initialized'",
                [],
                |row| row.get(0),
            )
            .expect("flag should be set so the Python CLI does not rebuild");
        assert_eq!(flag, "1");
    }

    #[test]
    fn test_insert_post_returns_rowid_then_none_on_duplicate() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");

        let post = sample_post("at://uri/1", "hello world");
        let id = db.insert_post(&post).expect("insert failed");
        assert_eq!(id, Some(1));

        // Replayed events (e.g. after a Jetstream reconnect) must be ignored.
        let again = db.insert_post(&post).expect("second insert failed");
        assert_eq!(again, None, "duplicate URI should be ignored");
    }

    #[test]
    fn test_insert_post_populates_fts() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");
        db.insert_post(&sample_post("at://uri/1", "unmistakable phrase"))
            .expect("insert failed");
        drop(db);

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("unmistakable", 10, None, None)
            .expect("search failed");
        assert_eq!(results.len(), 1, "trigger should have indexed the post");
        assert_eq!(results[0].uri, "at://uri/1");
    }

    #[test]
    fn test_embedding_round_trip() {
        let (_file, path) = empty_db();
        let mut db = Database::open_read_write(&path).expect("open failed");
        let id = db
            .insert_post(&sample_post("at://uri/1", "some text"))
            .expect("insert failed")
            .expect("expected a rowid");

        let pending = db.get_posts_without_embeddings(100).expect("query failed");
        assert_eq!(pending, vec![(id, "some text".to_string())]);

        let mut embedding = [0.0f32; 384];
        embedding[0] = 1.0;
        db.store_embeddings(&[(id, embedding)])
            .expect("store failed");

        let pending = db.get_posts_without_embeddings(100).expect("query failed");
        assert!(pending.is_empty(), "post should be marked as embedded");
        drop(db);

        // And it should now be findable by vector search.
        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_vec(&embedding, 10, None, None)
            .expect("vec search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "at://uri/1");
    }

    #[test]
    fn test_get_posts_without_embeddings_respects_limit() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");
        for i in 0..5 {
            db.insert_post(&sample_post(&format!("at://uri/{i}"), "text"))
                .expect("insert failed");
        }
        let pending = db.get_posts_without_embeddings(3).expect("query failed");
        assert_eq!(pending.len(), 3);
    }

    #[test]
    fn test_cursor_round_trip() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");

        assert_eq!(db.get_cursor().expect("read failed"), None);

        db.set_cursor(1_722_000_000_000_000).expect("write failed");
        assert_eq!(
            db.get_cursor().expect("read failed"),
            Some(1_722_000_000_000_000)
        );

        // Overwrite rather than accumulate rows.
        db.set_cursor(1_722_000_000_000_001).expect("write failed");
        assert_eq!(
            db.get_cursor().expect("read failed"),
            Some(1_722_000_000_000_001)
        );
    }

    #[test]
    fn test_hybrid_empty_db() {
        let (_file, path) = create_test_db();

        let db = Database::open(&path).expect("open failed");
        let query_emb = [0.0f32; 384];
        let results = db
            .search_hybrid("anything", Some(&query_emb), 10, None, None, 1.05)
            .expect("hybrid search on empty db failed");

        assert!(results.is_empty());
    }
}
