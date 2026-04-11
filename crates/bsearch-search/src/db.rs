use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::Once;

use anyhow::Context;
use anyhow::Result;
use rusqlite::Connection;
use rusqlite::OpenFlags;

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
    #[allow(dead_code)] // fetched from DB for completeness, not currently displayed
    pub cid: String,
    #[allow(dead_code)] // fetched from DB for completeness, not currently displayed
    pub author_did: String,
    pub author_handle: String,
    pub text: String,
    pub created_at: String,
    pub source: String,
    #[allow(dead_code)] // fetched from DB for completeness, not currently displayed
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
    pub fn open(path: &Path) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        Ok(Self { conn })
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
            result_map.entry(r.id).or_insert(r);
        }

        // Sort by RRF score descending
        let mut scored: Vec<(i64, f64)> = rrf_scores.into_iter().collect();
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
            .search_hybrid("rustacean", Some(&query_emb), 10, None, None)
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
            .search_hybrid("hello", None, 10, None, None)
            .expect("hybrid search failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "uri:1");
        assert_eq!(results[0].match_type, Some(MatchType::Keyword));
    }

    #[test]
    fn test_hybrid_empty_db() {
        let (_file, path) = create_test_db();

        let db = Database::open(&path).expect("open failed");
        let query_emb = [0.0f32; 384];
        let results = db
            .search_hybrid("anything", Some(&query_emb), 10, None, None)
            .expect("hybrid search on empty db failed");

        assert!(results.is_empty());
    }
}
