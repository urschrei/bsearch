use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::Once;

use anyhow::Context;
use anyhow::Result;
use rusqlite::Connection;
use rusqlite::OpenFlags;
use rusqlite::OptionalExtension;

use chrono::Local;

use crate::models::created_at_sort_key;
use crate::models::format_indexed_at;
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

CREATE TABLE IF NOT EXISTS pending_likes (
    uri TEXT PRIMARY KEY,
    queued_at TEXT NOT NULL
);
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

/// How search results should be ranked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Order {
    /// Best match first, by whichever score the search mode produces.
    #[default]
    Relevance,
    /// Newest first. This replaces the relevance ranking rather than breaking
    /// ties within it, so it changes which posts are returned, not just their
    /// order: a recent weak match can displace an older strong one.
    DateDesc,
}

/// How many candidates to retrieve per requested result when ordering by date.
///
/// Vector search is k-nearest-neighbour and the hybrid merge is built on rank
/// positions, so neither has an "all matching posts" set to order by date --
/// both must draw from a fixed-size pool of the best matches. Widening the
/// pool costs little at this database size and makes it far more likely that
/// the newest of the genuinely relevant posts are in it. Keyword search has no
/// such limit and orders in SQL across every match.
const DATE_ORDER_POOL_MULTIPLIER: usize = 20;

/// Re-order results newest first, in place.
///
/// The sort is stable, so posts sharing a timestamp keep the relative order the
/// search gave them. Timestamps that will not parse sort last, on the grounds
/// that a row we cannot date should not be presented as the most recent.
pub fn sort_by_created_at_desc(results: &mut [SearchResult]) {
    // `None` orders below `Some`, so reversing the key puts undatable rows last.
    results.sort_by_cached_key(|r| Reverse(created_at_sort_key(&r.created_at)));
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

    /// Record a liked post's URI as needing resolution.
    ///
    /// A like event names the liked post but does not carry its text, so the
    /// URI has to be held until it can be fetched. Holding it in memory meant
    /// the cursor advanced past a like that was still only queued, so a crash
    /// or a restart in between dropped it with nothing left to say it had
    /// existed. Duplicates are ignored: the same post may be liked again, and
    /// a replayed event must not enqueue it twice.
    pub fn queue_pending_like(&self, uri: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO pending_likes (uri, queued_at) VALUES (?1, ?2)",
            rusqlite::params![uri, format_indexed_at(Local::now().naive_local())],
        )?;
        Ok(())
    }

    /// The oldest queued like URIs, in the order they arrived.
    ///
    /// These are read rather than removed; call [`Self::remove_pending_likes`]
    /// once they have been dealt with, so that a failure leaves them queued.
    pub fn take_pending_likes(&self, limit: usize) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT uri FROM pending_likes ORDER BY rowid LIMIT ?1")?;
        let rows = stmt.query_map([limit as i64], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Drop queued like URIs that no longer need resolution.
    ///
    /// Each delete stands alone, so a crash partway leaves the remainder
    /// queued; re-resolving them is harmless because post inserts ignore a URI
    /// that is already present.
    pub fn remove_pending_likes(&self, uris: &[String]) -> Result<()> {
        for uri in uris {
            self.conn
                .execute("DELETE FROM pending_likes WHERE uri = ?1", [uri])?;
        }
        Ok(())
    }

    /// How many like URIs are waiting to be resolved.
    pub fn count_pending_likes(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT count(*) FROM pending_likes", [], |row| row.get(0))
            .map_err(Into::into)
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
        order: Order,
    ) -> Result<Vec<SearchResult>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        match self.run_fts_query(query, limit, source_filter, handle_filter, order) {
            Ok(results) => Ok(results),
            Err(e) => {
                // FTS5 syntax errors manifest as generic sqlite errors;
                // retry the query wrapped as a phrase literal.
                let phrase = format!("\"{}\"", query.replace('"', ""));
                match self.run_fts_query(&phrase, limit, source_filter, handle_filter, order) {
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
        order: Order,
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
        // Ordering by date here rather than trimming a relevance-ranked pool
        // afterwards means the limit applies to every matching post, so the
        // newest matches are found however weakly they score. SQLite compares
        // the timestamps as text; the caller re-sorts the returned rows with
        // `sort_by_created_at_desc`, which corrects the handful of stored forms
        // where text order and chronological order come apart.
        let order_clause = match order {
            Order::Relevance => "fts_posts.rank",
            Order::DateDesc => "p.created_at DESC",
        };
        let sql = format!(
            "SELECT p.id, fts_posts.rank AS bm25_rank,
                    p.uri, p.cid, p.author_did, p.author_handle, p.text,
                    p.created_at, p.source, p.indexed_at
             FROM fts_posts
             INNER JOIN posts p ON p.id = fts_posts.rowid
             WHERE {}
             ORDER BY {}
             LIMIT ?{}",
            where_clause, order_clause, param_idx
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

        let mut results = results.collect::<rusqlite::Result<Vec<_>>>()?;

        if order == Order::DateDesc {
            // SQLite compared the timestamps as text to pick these rows, which
            // parts company with chronological order for the stored forms that
            // differ in offset or precision. Re-sorting properly fixes the order
            // of what came back; the selection can still be off by a row at the
            // limit boundary if such timestamps straddle it.
            sort_by_created_at_desc(&mut results);
        }

        Ok(results)
    }

    pub fn search_vec(
        &self,
        query_embedding: &[f32; 384],
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
        order: Order,
    ) -> Result<Vec<SearchResult>> {
        // Over-fetch when filters are active so we have enough results after
        // filtering, and again when ordering by date, since the newest close
        // matches need not be the closest ones.
        let fetch_limit = match order {
            Order::DateDesc => limit * DATE_ORDER_POOL_MULTIPLIER,
            Order::Relevance if source_filter.is_some() || handle_filter.is_some() => limit * 5,
            Order::Relevance => limit,
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
        if order == Order::DateDesc {
            sort_by_created_at_desc(&mut results);
        }
        results.truncate(limit);

        Ok(results)
    }

    // The argument list is at the point where grouping the limit, filters and
    // ordering into a shared parameter struct would read better across all
    // three search methods. Left as is for now to keep this signature in the
    // same shape as `search_fts` and `search_vec`.
    #[allow(clippy::too_many_arguments)]
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: Option<&[f32; 384]>,
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
        max_semantic_distance: f64,
        order: Order,
    ) -> Result<Vec<SearchResult>> {
        // Reciprocal rank fusion scores a post by where each search ranked it,
        // so both sub-queries must stay relevance-ordered whatever the caller
        // asked for. Date ordering is applied to the merged set instead, drawn
        // from a wider pool so that recent posts outside the top few relevance
        // ranks can still surface.
        let fetch_limit = match order {
            Order::Relevance => limit * 3,
            Order::DateDesc => limit * DATE_ORDER_POOL_MULTIPLIER,
        };

        let fts_results = self.search_fts(
            query,
            fetch_limit,
            source_filter,
            handle_filter,
            Order::Relevance,
        )?;
        let vec_results = if let Some(emb) = query_embedding {
            self.search_vec(
                emb,
                fetch_limit,
                source_filter,
                handle_filter,
                Order::Relevance,
            )?
        } else {
            vec![]
        };

        // Fallback: only one source has results
        if fts_results.is_empty() && !vec_results.is_empty() {
            let mut out = vec_results;
            for r in &mut out {
                r.match_type = Some(MatchType::Semantic);
            }
            if order == Order::DateDesc {
                sort_by_created_at_desc(&mut out);
            }
            out.truncate(limit);
            return Ok(out);
        }
        if !fts_results.is_empty() && vec_results.is_empty() {
            let mut out = fts_results;
            for r in &mut out {
                r.match_type = Some(MatchType::Keyword);
            }
            if order == Order::DateDesc {
                sort_by_created_at_desc(&mut out);
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
        if order == Order::Relevance {
            // Ordering by date keeps every candidate for now: cutting to the
            // limit on RRF score first would discard the recent posts the sort
            // below exists to surface.
            scored.truncate(limit);
        }

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

        if order == Order::DateDesc {
            sort_by_created_at_desc(&mut out);
            out.truncate(limit);
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
        insert_post_at(path, uri, text, source, handle, "2024-01-01T00:00:00Z");
    }

    fn insert_post_at(
        path: &Path,
        uri: &str,
        text: &str,
        source: &str,
        handle: &str,
        created_at: &str,
    ) {
        let conn = Connection::open(path).expect("failed to open db");
        conn.execute(
            "INSERT INTO posts (uri, cid, author_did, author_handle, text, created_at, source, indexed_at)
             VALUES (?1, '', '', ?2, ?3, ?5, ?4, '2024-01-01T00:00:00Z')",
            rusqlite::params![uri, handle, text, source, created_at],
        )
        .expect("failed to insert post");
    }

    /// The URIs of `results`, in order, for comparing against an expected order.
    fn uris(results: &[SearchResult]) -> Vec<&str> {
        results.iter().map(|r| r.uri.as_str()).collect()
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
            .search_fts("hello", 10, None, None, Order::Relevance)
            .expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].uri, "uri:1");
    }

    #[test]
    fn test_fts_empty_query() {
        let (_file, path) = create_test_db();
        insert_post(&path, "uri:1", "some text", "bluesky", "alice.bsky.social");

        let db = Database::open(&path).expect("open failed");

        let results = db
            .search_fts("", 10, None, None, Order::Relevance)
            .expect("search failed");
        assert!(results.is_empty());

        let results = db
            .search_fts("   ", 10, None, None, Order::Relevance)
            .expect("search failed");
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
            .search_fts("hello", 10, Some("bluesky"), None, Order::Relevance)
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
            .search_fts(
                "hello",
                10,
                None,
                Some("alice.bsky.social"),
                Order::Relevance,
            )
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
            .search_fts("AND OR ( \" broken***", 10, None, None, Order::Relevance)
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
            .search_hybrid(
                "rustacean",
                Some(&query_emb),
                10,
                None,
                None,
                1.05,
                Order::Relevance,
            )
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
            .search_hybrid("hello", None, 10, None, None, 1.05, Order::Relevance)
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
            "index sqlite_autoindex_pending_likes_1",
            "index sqlite_autoindex_posts_1",
            "index sqlite_autoindex_vec_posts_info_1",
            "index sqlite_autoindex_vec_posts_vector_chunks00_1",
            "table fts_posts",
            "table fts_posts_config",
            "table fts_posts_data",
            "table fts_posts_docsize",
            "table fts_posts_idx",
            "table meta",
            "table pending_likes",
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
            .search_fts("unmistakable", 10, None, None, Order::Relevance)
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
            .search_vec(&embedding, 10, None, None, Order::Relevance)
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
    fn test_pending_likes_are_returned_in_arrival_order() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");

        for uri in ["at://a", "at://b", "at://c"] {
            db.queue_pending_like(uri).expect("queue failed");
        }

        assert_eq!(
            db.take_pending_likes(10).expect("read failed"),
            vec!["at://a", "at://b", "at://c"]
        );
        assert_eq!(
            db.take_pending_likes(2).expect("read failed"),
            vec!["at://a", "at://b"],
            "a limit must take the oldest, not an arbitrary subset"
        );
    }

    #[test]
    fn test_take_pending_likes_does_not_consume() {
        // Resolution can fail, so reading must leave the queue intact; only
        // remove_pending_likes drops entries.
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");
        db.queue_pending_like("at://a").expect("queue failed");

        assert_eq!(db.take_pending_likes(10).expect("read failed").len(), 1);
        assert_eq!(
            db.take_pending_likes(10).expect("read failed").len(),
            1,
            "reading twice must return the same entry"
        );
    }

    #[test]
    fn test_remove_pending_likes_drops_only_the_named() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");
        for uri in ["at://a", "at://b", "at://c"] {
            db.queue_pending_like(uri).expect("queue failed");
        }

        db.remove_pending_likes(&["at://a".to_string(), "at://c".to_string()])
            .expect("remove failed");

        assert_eq!(
            db.take_pending_likes(10).expect("read failed"),
            vec!["at://b"]
        );
    }

    #[test]
    fn test_queue_pending_like_ignores_duplicates() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");
        db.queue_pending_like("at://a").expect("queue failed");
        db.queue_pending_like("at://a").expect("requeue failed");

        assert_eq!(db.count_pending_likes().expect("count failed"), 1);
    }

    #[test]
    fn test_removing_an_unqueued_like_is_not_an_error() {
        let (_file, path) = empty_db();
        let db = Database::open_read_write(&path).expect("open failed");
        db.remove_pending_likes(&["at://never-queued".to_string()])
            .expect("removing an absent uri should be a no-op");
        assert_eq!(db.count_pending_likes().expect("count failed"), 0);
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
    fn test_sort_by_created_at_desc_puts_newest_first() {
        let (_file, path) = create_test_db();
        insert_post_at(
            &path,
            "uri:old",
            "text",
            "bluesky",
            "alice",
            "2024-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:new",
            "text",
            "bluesky",
            "alice",
            "2026-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:mid",
            "text",
            "bluesky",
            "alice",
            "2025-01-01T00:00:00+00:00",
        );

        let db = Database::open(&path).expect("open failed");
        let mut results = db
            .search_fts("text", 10, None, None, Order::Relevance)
            .expect("search failed");
        sort_by_created_at_desc(&mut results);

        assert_eq!(uris(&results), vec!["uri:new", "uri:mid", "uri:old"]);
    }

    #[test]
    fn test_sort_by_created_at_desc_puts_undatable_rows_last() {
        let (_file, path) = create_test_db();
        insert_post_at(&path, "uri:junk", "text", "bluesky", "alice", "not a date");
        insert_post_at(
            &path,
            "uri:dated",
            "text",
            "bluesky",
            "alice",
            "2024-01-01T00:00:00+00:00",
        );

        let db = Database::open(&path).expect("open failed");
        let mut results = db
            .search_fts("text", 10, None, None, Order::Relevance)
            .expect("search failed");
        sort_by_created_at_desc(&mut results);

        assert_eq!(uris(&results), vec!["uri:dated", "uri:junk"]);
    }

    #[test]
    fn test_sort_by_created_at_desc_corrects_for_offsets() {
        // These two sort the wrong way round as text: the earlier instant has
        // the later wall-clock reading. Only parsing gets this right, so this
        // is what the Rust-side re-sort buys over SQLite's text comparison.
        let (_file, path) = create_test_db();
        insert_post_at(
            &path,
            "uri:earlier",
            "text",
            "bluesky",
            "alice",
            "2026-03-29T10:00:00+05:00",
        );
        insert_post_at(
            &path,
            "uri:later",
            "text",
            "bluesky",
            "alice",
            "2026-03-29T09:00:00+00:00",
        );

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("text", 10, None, None, Order::DateDesc)
            .expect("search failed");

        assert_eq!(uris(&results), vec!["uri:later", "uri:earlier"]);
    }

    #[test]
    fn test_keyword_date_order_overrides_relevance() {
        let (_file, path) = create_test_db();
        // The old post repeats the term, so BM25 ranks it above the new one.
        insert_post_at(
            &path,
            "uri:old",
            "ferrets ferrets ferrets",
            "bluesky",
            "alice",
            "2024-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:new",
            "ferrets, briefly",
            "bluesky",
            "alice",
            "2026-01-01T00:00:00+00:00",
        );

        let db = Database::open(&path).expect("open failed");

        let by_relevance = db
            .search_fts("ferrets", 10, None, None, Order::Relevance)
            .expect("search failed");
        assert_eq!(uris(&by_relevance), vec!["uri:old", "uri:new"]);

        let by_date = db
            .search_fts("ferrets", 10, None, None, Order::DateDesc)
            .expect("search failed");
        assert_eq!(uris(&by_date), vec!["uri:new", "uri:old"]);
    }

    #[test]
    fn test_keyword_date_order_displaces_stronger_older_match() {
        // The behaviour that makes this an override rather than a tiebreak:
        // at limit 1 the recent weak match wins over the older strong one.
        let (_file, path) = create_test_db();
        insert_post_at(
            &path,
            "uri:old",
            "ferrets ferrets ferrets",
            "bluesky",
            "alice",
            "2024-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:new",
            "ferrets, briefly",
            "bluesky",
            "alice",
            "2026-01-01T00:00:00+00:00",
        );

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("ferrets", 1, None, None, Order::DateDesc)
            .expect("search failed");

        assert_eq!(uris(&results), vec!["uri:new"]);
    }

    #[test]
    fn test_keyword_date_order_respects_filters() {
        let (_file, path) = create_test_db();
        insert_post_at(
            &path,
            "uri:1",
            "ferrets",
            "bluesky",
            "alice",
            "2026-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:2",
            "ferrets",
            "mastodon",
            "alice",
            "2026-06-01T00:00:00+00:00",
        );

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("ferrets", 10, Some("bluesky"), None, Order::DateDesc)
            .expect("search failed");

        assert_eq!(uris(&results), vec!["uri:1"], "newer post is filtered out");
    }

    #[test]
    fn test_semantic_date_order_overrides_distance() {
        let (_file, path) = create_test_db();
        insert_post_at(
            &path,
            "uri:close-old",
            "rustacean",
            "bluesky",
            "alice",
            "2024-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:far-new",
            "rustacean",
            "bluesky",
            "alice",
            "2026-01-01T00:00:00+00:00",
        );

        let mut close = [0.0f32; 384];
        close[0] = 1.0;
        let mut far = [0.0f32; 384];
        far[0] = 0.2;
        insert_embedding(&path, 1, &close);
        insert_embedding(&path, 2, &far);

        let mut query = [0.0f32; 384];
        query[0] = 1.0;

        let db = Database::open(&path).expect("open failed");

        let by_distance = db
            .search_vec(&query, 10, None, None, Order::Relevance)
            .expect("search failed");
        assert_eq!(uris(&by_distance), vec!["uri:close-old", "uri:far-new"]);

        let by_date = db
            .search_vec(&query, 10, None, None, Order::DateDesc)
            .expect("search failed");
        assert_eq!(uris(&by_date), vec!["uri:far-new", "uri:close-old"]);
    }

    #[test]
    fn test_hybrid_date_order_overrides_rrf() {
        let (_file, path) = create_test_db();
        insert_post_at(
            &path,
            "uri:old",
            "rustacean programming language",
            "bluesky",
            "alice",
            "2024-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:new",
            "rustacean notes",
            "bluesky",
            "bob",
            "2026-01-01T00:00:00+00:00",
        );

        let mut emb1 = [0.0f32; 384];
        emb1[0] = 1.0;
        let mut emb2 = [0.0f32; 384];
        emb2[0] = 0.95;
        insert_embedding(&path, 1, &emb1);
        insert_embedding(&path, 2, &emb2);

        let mut query = [0.0f32; 384];
        query[0] = 1.0;

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_hybrid(
                "rustacean",
                Some(&query),
                10,
                None,
                None,
                1.05,
                Order::DateDesc,
            )
            .expect("hybrid search failed");

        assert_eq!(uris(&results), vec!["uri:new", "uri:old"]);
        // The merge still records how each post was found; only order changed.
        assert!(results.iter().all(|r| r.rrf_score.is_some()));
    }

    #[test]
    fn test_hybrid_date_order_applies_to_single_source_fallback() {
        // No embeddings, so the keyword-only fallback path returns the results.
        let (_file, path) = create_test_db();
        insert_post_at(
            &path,
            "uri:old",
            "ferrets ferrets ferrets",
            "bluesky",
            "alice",
            "2024-01-01T00:00:00+00:00",
        );
        insert_post_at(
            &path,
            "uri:new",
            "ferrets, briefly",
            "bluesky",
            "alice",
            "2026-01-01T00:00:00+00:00",
        );

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_hybrid("ferrets", None, 10, None, None, 1.05, Order::DateDesc)
            .expect("hybrid search failed");

        assert_eq!(uris(&results), vec!["uri:new", "uri:old"]);
    }

    #[test]
    fn test_date_order_still_honours_limit() {
        let (_file, path) = create_test_db();
        for year in [2022, 2023, 2024, 2025, 2026] {
            insert_post_at(
                &path,
                &format!("uri:{year}"),
                "ferrets",
                "bluesky",
                "alice",
                &format!("{year}-01-01T00:00:00+00:00"),
            );
        }

        let db = Database::open(&path).expect("open failed");
        let results = db
            .search_fts("ferrets", 2, None, None, Order::DateDesc)
            .expect("search failed");

        assert_eq!(uris(&results), vec!["uri:2026", "uri:2025"]);
    }

    #[test]
    fn test_hybrid_empty_db() {
        let (_file, path) = create_test_db();

        let db = Database::open(&path).expect("open failed");
        let query_emb = [0.0f32; 384];
        let results = db
            .search_hybrid(
                "anything",
                Some(&query_emb),
                10,
                None,
                None,
                1.05,
                Order::Relevance,
            )
            .expect("hybrid search on empty db failed");

        assert!(results.is_empty());
    }
}
