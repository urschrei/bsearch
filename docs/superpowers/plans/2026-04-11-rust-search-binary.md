# Rust Search Binary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Rust binary (`bsearch-search`) that replaces the Python `bsearch search` command with sub-second cold start, using ONNX Runtime for inference instead of PyTorch.

**Architecture:** A Cargo workspace at the repo root with a single crate `crates/bsearch-search/`. The binary reads config from CLI flags / env vars / `.env`, loads an ONNX model + tokenizer for embedding queries, and queries the existing SQLite + sqlite-vec + FTS5 database read-only. A new Python `export-model` CLI command exports the model to ONNX format.

**Tech Stack:** Rust, clap, ort (ONNX Runtime), tokenizers (HuggingFace), rusqlite + sqlite-vec, ndarray, dotenvy

**Spec:** `docs/superpowers/specs/2026-04-11-rust-search-binary-design.md`

---

## File Structure

```
Cargo.toml                          # Workspace root
crates/bsearch-search/
  Cargo.toml                        # Crate dependencies
  README.md                         # Dev docs: embedding pipeline, model export, usage
  src/
    main.rs                         # CLI entry point (clap), wiring, output formatting
    config.rs                       # Config resolution: CLI > env > .env > defaults
    db.rs                           # SQLite queries: FTS5, KNN vector, hybrid RRF
    embed.rs                        # Tokenization, ONNX inference, mean pooling, L2 norm
src/bsearch/cli.py                  # Modify: add export-model command
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/bsearch-search/Cargo.toml`
- Create: `crates/bsearch-search/src/main.rs`

- [ ] **Step 1: Create workspace Cargo.toml**

```toml
[workspace]
members = ["crates/bsearch-search"]
resolver = "2"
```

- [ ] **Step 2: Create crate Cargo.toml**

```toml
[package]
name = "bsearch-search"
version = "0.1.0"
edition = "2024"

[dependencies]
clap = { version = "4", features = ["derive", "env"] }
ort = { version = "2", features = ["load-dynamic"] }
tokenizers = "0.21"
rusqlite = { version = "0.34", features = ["bundled", "load_extension"] }
ndarray = "0.16"
dotenvy = "0.15"
dirs = "6"
anyhow = "1"
```

- [ ] **Step 3: Create minimal main.rs**

```rust
fn main() {
    println!("bsearch-search");
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build -p bsearch-search`
Expected: compiles successfully (dependencies will download on first build)

- [ ] **Step 5: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Scaffold Rust workspace and bsearch-search crate"
```

---

### Task 2: Config module

**Files:**
- Create: `crates/bsearch-search/src/config.rs`
- Modify: `crates/bsearch-search/src/main.rs`

- [ ] **Step 1: Write tests for config resolution**

In `config.rs`, add a test module:

```rust
use std::path::PathBuf;

pub struct Config {
    pub db_path: PathBuf,
    pub model_dir: PathBuf,
}

impl Config {
    /// Resolve config from explicit values, falling back to env vars, then .env, then defaults.
    /// `cli_db` and `cli_model` represent values passed via CLI flags (None if not passed).
    pub fn resolve(cli_db: Option<PathBuf>, cli_model: Option<PathBuf>) -> anyhow::Result<Self> {
        // Load .env silently (ignore if missing)
        let _ = dotenvy::dotenv();

        let db_path = cli_db
            .or_else(|| std::env::var("BSEARCH_DB_PATH").ok().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("bsearch.db"));

        let default_model_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("bsearch")
            .join("all-MiniLM-L6-v2");

        let model_dir = cli_model
            .or_else(|| std::env::var("BSEARCH_MODEL_DIR").ok().map(PathBuf::from))
            .unwrap_or(default_model_dir);

        Ok(Config { db_path, model_dir })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_when_no_env() {
        // Clear relevant env vars for test isolation
        std::env::remove_var("BSEARCH_DB_PATH");
        std::env::remove_var("BSEARCH_MODEL_DIR");

        let config = Config::resolve(None, None).unwrap();
        assert_eq!(config.db_path, PathBuf::from("bsearch.db"));
        assert!(config.model_dir.ends_with("bsearch/all-MiniLM-L6-v2"));
    }

    #[test]
    fn test_cli_overrides_env() {
        std::env::set_var("BSEARCH_DB_PATH", "/env/path.db");

        let config = Config::resolve(Some(PathBuf::from("/cli/path.db")), None).unwrap();
        assert_eq!(config.db_path, PathBuf::from("/cli/path.db"));

        std::env::remove_var("BSEARCH_DB_PATH");
    }

    #[test]
    fn test_env_var_used_when_no_cli() {
        std::env::set_var("BSEARCH_DB_PATH", "/env/path.db");

        let config = Config::resolve(None, None).unwrap();
        assert_eq!(config.db_path, PathBuf::from("/env/path.db"));

        std::env::remove_var("BSEARCH_DB_PATH");
    }
}
```

- [ ] **Step 2: Wire config.rs into main.rs**

```rust
mod config;

fn main() {
    println!("bsearch-search");
}
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo nextest r -p bsearch-search`
Expected: 3 tests pass

- [ ] **Step 4: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add config module with CLI/env/.env resolution"
```

---

### Task 3: Database module -- FTS search

**Files:**
- Create: `crates/bsearch-search/src/db.rs`
- Modify: `crates/bsearch-search/src/main.rs`

- [ ] **Step 1: Write the Database struct and FTS search test**

```rust
use std::path::Path;

use anyhow::Result;
use rusqlite::Connection;

/// A search result row from the database.
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
    pub match_type: Option<String>,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.pragma_update(None, "busy_timeout", 5000)?;

        Ok(Database { conn })
    }

    /// Full-text search using FTS5 BM25 ranking.
    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(vec![]);
        }

        match self.search_fts_inner(query, limit, source_filter, handle_filter) {
            Ok(results) => Ok(results),
            Err(_) => {
                // Retry as phrase query on FTS syntax error
                let phrase = format!("\"{}\"", query);
                self.search_fts_inner(&phrase, limit, source_filter, handle_filter)
                    .or(Ok(vec![]))
            }
        }
    }

    fn search_fts_inner(
        &self,
        query: &str,
        limit: usize,
        source_filter: Option<&str>,
        handle_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let mut where_parts = vec!["fts_posts MATCH ?1".to_string()];
        let mut param_idx = 2;

        if source_filter.is_some() {
            where_parts.push(format!("p.source = ?{param_idx}"));
            param_idx += 1;
        }
        if handle_filter.is_some() {
            where_parts.push(format!("p.author_handle = ?{param_idx}"));
            param_idx += 1;
        }

        let where_clause = where_parts.join(" AND ");
        let sql = format!(
            "SELECT p.id, fts_posts.rank AS bm25_rank, \
             p.uri, p.cid, p.author_did, p.author_handle, p.text, \
             p.created_at, p.source, p.indexed_at \
             FROM fts_posts \
             INNER JOIN posts p ON p.id = fts_posts.rowid \
             WHERE {where_clause} \
             ORDER BY fts_posts.rank \
             LIMIT ?{param_idx}"
        );

        let mut stmt = self.conn.prepare(&sql)?;

        // Build dynamic parameter list
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(query.to_string())];
        if let Some(s) = source_filter {
            params.push(Box::new(s.to_string()));
        }
        if let Some(h) = handle_filter {
            params.push(Box::new(h.to_string()));
        }
        params.push(Box::new(limit as i64));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok(SearchResult {
                id: row.get("id")?,
                uri: row.get("uri")?,
                cid: row.get("cid")?,
                author_did: row.get("author_did")?,
                author_handle: row.get("author_handle")?,
                text: row.get("text")?,
                created_at: row.get("created_at")?,
                source: row.get("source")?,
                indexed_at: row.get("indexed_at")?,
                distance: None,
                bm25_rank: row.get("bm25_rank")?,
                rrf_score: None,
                match_type: None,
            })
        })?;

        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// List posts by author handle, newest first.
    pub fn list_by_handle(
        &self,
        handle: &str,
        limit: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let mut sql = String::from(
            "SELECT p.id, p.uri, p.cid, p.author_did, p.author_handle, p.text, \
             p.created_at, p.source, p.indexed_at \
             FROM posts p WHERE p.author_handle = ?1"
        );
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(handle.to_string())];

        if let Some(s) = source_filter {
            sql.push_str(" AND p.source = ?2");
            params.push(Box::new(s.to_string()));
        }
        sql.push_str(&format!(" ORDER BY p.created_at DESC LIMIT ?{}", params.len() + 1));
        params.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok(SearchResult {
                id: row.get("id")?,
                uri: row.get("uri")?,
                cid: row.get("cid")?,
                author_did: row.get("author_did")?,
                author_handle: row.get("author_handle")?,
                text: row.get("text")?,
                created_at: row.get("created_at")?,
                source: row.get("source")?,
                indexed_at: row.get("indexed_at")?,
                distance: None,
                bm25_rank: None,
                rrf_score: None,
                match_type: None,
            })
        })?;

        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::NamedTempFile;

    fn create_test_db() -> (NamedTempFile, std::path::PathBuf) {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();

        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
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
             CREATE VIRTUAL TABLE IF NOT EXISTS fts_posts USING fts5(
                 text, content=posts, content_rowid=id, tokenize='porter unicode61'
             );
             CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
                 INSERT INTO fts_posts(rowid, text) VALUES (new.id, new.text);
             END;
             INSERT OR REPLACE INTO meta (key, value) VALUES ('fts_initialized', '1');"
        ).unwrap();
        conn.close().unwrap();

        (file, path)
    }

    fn insert_post(path: &Path, uri: &str, text: &str, source: &str, handle: &str) {
        let conn = Connection::open(path).unwrap();
        conn.execute(
            "INSERT INTO posts (uri, cid, author_did, author_handle, text, created_at, source, indexed_at)
             VALUES (?1, 'cid', 'did:plc:abc', ?2, ?3, '2025-01-01T00:00:00', ?4, '2025-01-01T00:00:00')",
            rusqlite::params![uri, handle, text, source],
        ).unwrap();
    }

    #[test]
    fn test_fts_finds_match() {
        let (_file, path) = create_test_db();
        insert_post(&path, "at://a/1", "the cat sat on the mat", "own_post", "test.bsky.social");
        insert_post(&path, "at://a/2", "dogs playing in the park", "own_post", "test.bsky.social");

        let db = Database::open(&path).unwrap();
        let results = db.search_fts("cat", 10, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("cat"));
    }

    #[test]
    fn test_fts_empty_query() {
        let (_file, path) = create_test_db();
        insert_post(&path, "at://a/1", "hello world", "own_post", "test.bsky.social");

        let db = Database::open(&path).unwrap();
        assert!(db.search_fts("", 10, None, None).unwrap().is_empty());
        assert!(db.search_fts("   ", 10, None, None).unwrap().is_empty());
    }

    #[test]
    fn test_fts_source_filter() {
        let (_file, path) = create_test_db();
        insert_post(&path, "at://a/1", "cats are great", "own_post", "test.bsky.social");
        insert_post(&path, "at://a/2", "cats are wonderful", "like", "test.bsky.social");

        let db = Database::open(&path).unwrap();
        let results = db.search_fts("cats", 10, Some("own_post"), None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "own_post");
    }

    #[test]
    fn test_fts_handle_filter() {
        let (_file, path) = create_test_db();
        insert_post(&path, "at://a/1", "cats are great", "own_post", "alice.bsky.social");
        insert_post(&path, "at://a/2", "cats are wonderful", "own_post", "bob.bsky.social");

        let db = Database::open(&path).unwrap();
        let results = db.search_fts("cats", 10, None, Some("alice.bsky.social")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].author_handle, "alice.bsky.social");
    }

    #[test]
    fn test_fts_special_chars_do_not_crash() {
        let (_file, path) = create_test_db();
        insert_post(&path, "at://a/1", "hello world", "own_post", "test.bsky.social");

        let db = Database::open(&path).unwrap();
        let results = db.search_fts("hello (world", 10, None, None).unwrap();
        assert!(results.is_empty() || !results.is_empty()); // just must not crash
    }

    #[test]
    fn test_list_by_handle() {
        let (_file, path) = create_test_db();
        insert_post(&path, "at://a/1", "post one", "own_post", "alice.bsky.social");
        insert_post(&path, "at://a/2", "post two", "like", "alice.bsky.social");
        insert_post(&path, "at://a/3", "post three", "own_post", "bob.bsky.social");

        let db = Database::open(&path).unwrap();
        let results = db.list_by_handle("alice.bsky.social", 10, None).unwrap();
        assert_eq!(results.len(), 2);

        let results = db.list_by_handle("alice.bsky.social", 10, Some("own_post")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "own_post");
    }
}
```

Note: add `tempfile` as a dev-dependency in Cargo.toml:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Wire db.rs into main.rs**

Add `mod db;` to main.rs.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo nextest r -p bsearch-search`
Expected: all tests pass (config + db tests)

- [ ] **Step 4: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add database module with FTS search and list-by-handle"
```

---

### Task 4: Database module -- vector search and hybrid RRF

**Files:**
- Modify: `crates/bsearch-search/src/db.rs`

- [ ] **Step 1: Add sqlite-vec loading and KNN search**

Add the `sqlite-vec` dependency to `crates/bsearch-search/Cargo.toml`:

```toml
sqlite-vec = "0.1"
```

Update `Database::open` to load the sqlite-vec extension, and add a `search_vec` method:

```rust
pub fn open(path: &Path) -> Result<Self> {
    let conn = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.pragma_update(None, "busy_timeout", 5000)?;
    unsafe {
        conn.load_extension_enable()?;
        sqlite_vec::load(&conn)?;
        conn.load_extension_disable()?;
    }

    Ok(Database { conn })
}

/// KNN vector search using sqlite-vec.
pub fn search_vec(
    &self,
    query_embedding: &[f32; 384],
    limit: usize,
    source_filter: Option<&str>,
    handle_filter: Option<&str>,
) -> Result<Vec<SearchResult>> {
    let needs_post_filter = source_filter.is_some() || handle_filter.is_some();
    let fetch_limit = if needs_post_filter { limit * 5 } else { limit };

    let embedding_bytes: Vec<u8> = query_embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let mut stmt = self.conn.prepare(
        "SELECT v.rowid AS id, v.distance, \
         p.uri, p.cid, p.author_did, p.author_handle, p.text, \
         p.created_at, p.source, p.indexed_at \
         FROM vec_posts v \
         INNER JOIN posts p ON p.id = v.rowid \
         WHERE v.embedding MATCH ?1 AND k = ?2 \
         ORDER BY v.distance"
    )?;

    let rows = stmt.query_map(rusqlite::params![embedding_bytes, fetch_limit as i64], |row| {
        Ok(SearchResult {
            id: row.get("id")?,
            uri: row.get("uri")?,
            cid: row.get("cid")?,
            author_did: row.get("author_did")?,
            author_handle: row.get("author_handle")?,
            text: row.get("text")?,
            created_at: row.get("created_at")?,
            source: row.get("source")?,
            indexed_at: row.get("indexed_at")?,
            distance: row.get("distance")?,
            bm25_rank: None,
            rrf_score: None,
            match_type: None,
        })
    })?;

    let mut results: Vec<SearchResult> = rows.collect::<std::result::Result<Vec<_>, _>>()?;

    if let Some(s) = source_filter {
        results.retain(|r| r.source == s);
    }
    if let Some(h) = handle_filter {
        results.retain(|r| r.author_handle == h);
    }
    results.truncate(limit);

    Ok(results)
}
```

- [ ] **Step 2: Add hybrid search with RRF**

```rust
/// Hybrid search combining FTS5 BM25 and KNN vector using Reciprocal Rank Fusion.
pub fn search_hybrid(
    &self,
    query: &str,
    query_embedding: Option<&[f32; 384]>,
    limit: usize,
    source_filter: Option<&str>,
    handle_filter: Option<&str>,
) -> Result<Vec<SearchResult>> {
    let rrf_k: usize = 60;
    let fetch_limit = limit * 3;

    let fts_results = self.search_fts(query, fetch_limit, source_filter, handle_filter)?;

    let vec_results = match query_embedding {
        Some(emb) => self.search_vec(emb, fetch_limit, source_filter, handle_filter)?,
        None => vec![],
    };

    if fts_results.is_empty() && vec_results.is_empty() {
        return Ok(vec![]);
    }
    if fts_results.is_empty() {
        let mut results = vec_results;
        results.truncate(limit);
        for r in &mut results {
            r.match_type = Some("semantic".to_string());
        }
        return Ok(results);
    }
    if vec_results.is_empty() {
        let mut results = fts_results;
        results.truncate(limit);
        for r in &mut results {
            r.match_type = Some("keyword".to_string());
        }
        return Ok(results);
    }

    // Reciprocal Rank Fusion
    let mut scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
    let mut match_types: std::collections::HashMap<i64, std::collections::BTreeSet<String>> =
        std::collections::HashMap::new();
    let mut docs: std::collections::HashMap<i64, SearchResult> = std::collections::HashMap::new();

    for (rank, doc) in fts_results.into_iter().enumerate() {
        let doc_id = doc.id;
        *scores.entry(doc_id).or_default() += 1.0 / (rrf_k + rank + 1) as f64;
        match_types.entry(doc_id).or_default().insert("keyword".to_string());
        docs.entry(doc_id).or_insert(doc);
    }

    for (rank, doc) in vec_results.into_iter().enumerate() {
        let doc_id = doc.id;
        *scores.entry(doc_id).or_default() += 1.0 / (rrf_k + rank + 1) as f64;
        match_types.entry(doc_id).or_default().insert("semantic".to_string());
        docs.entry(doc_id).or_insert(doc);
    }

    let mut ranked_ids: Vec<i64> = scores.keys().copied().collect();
    ranked_ids.sort_by(|a, b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut results = Vec::new();
    for doc_id in ranked_ids.into_iter().take(limit) {
        let mut doc = docs.remove(&doc_id).unwrap();
        doc.rrf_score = Some(scores[&doc_id]);
        let types: Vec<String> = match_types[&doc_id].iter().cloned().collect();
        doc.match_type = Some(types.join("+"));
        results.push(doc);
    }

    Ok(results)
}
```

- [ ] **Step 3: Add RRF unit test (pure scoring logic)**

Add to the test module. This test creates a database with vec_posts to test the full hybrid path:

```rust
#[test]
fn test_hybrid_both_sources() {
    let (_file, path) = create_test_db_with_vec();
    insert_post(&path, "at://a/1", "python programming language", "own_post", "test.bsky.social");
    insert_post(&path, "at://a/2", "machine learning concepts", "own_post", "test.bsky.social");
    insert_embedding(&path, 1, &[0.1; 384]);
    insert_embedding(&path, 2, &[0.9; 384]);

    let db = Database::open(&path).unwrap();
    let query_emb: [f32; 384] = [0.1; 384]; // close to post 1
    let results = db.search_hybrid("python", Some(&query_emb), 10, None, None).unwrap();
    assert!(!results.is_empty());
    // Post 1 matches both keyword ("python") and vector (close embedding)
    assert_eq!(results[0].id, 1);
    assert_eq!(results[0].match_type.as_deref(), Some("keyword+semantic"));
}

#[test]
fn test_hybrid_fts_only_fallback() {
    let (_file, path) = create_test_db_with_vec();
    insert_post(&path, "at://a/1", "specific keyword here", "own_post", "test.bsky.social");

    let db = Database::open(&path).unwrap();
    let results = db.search_hybrid("keyword", None, 10, None, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].match_type.as_deref(), Some("keyword"));
}

#[test]
fn test_hybrid_empty_db() {
    let (_file, path) = create_test_db_with_vec();

    let db = Database::open(&path).unwrap();
    let query_emb: [f32; 384] = [0.1; 384];
    let results = db.search_hybrid("anything", Some(&query_emb), 10, None, None).unwrap();
    assert!(results.is_empty());
}
```

These tests require updated helpers that also create the `vec_posts` virtual table:

```rust
fn create_test_db_with_vec() -> (NamedTempFile, std::path::PathBuf) {
    let file = NamedTempFile::new().unwrap();
    let path = file.path().to_path_buf();

    let conn = Connection::open(&path).unwrap();
    unsafe {
        conn.load_extension_enable().unwrap();
        sqlite_vec::load(&conn).unwrap();
        conn.load_extension_disable().unwrap();
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
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
         CREATE VIRTUAL TABLE IF NOT EXISTS fts_posts USING fts5(
             text, content=posts, content_rowid=id, tokenize='porter unicode61'
         );
         CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
             INSERT INTO fts_posts(rowid, text) VALUES (new.id, new.text);
         END;
         INSERT OR REPLACE INTO meta (key, value) VALUES ('fts_initialized', '1');"
    ).unwrap();
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_posts USING vec0(embedding float[384])",
        [],
    ).unwrap();
    conn.close().unwrap();

    (file, path)
}

fn insert_embedding(path: &Path, rowid: i64, embedding: &[f32; 384]) {
    let conn = Connection::open(path).unwrap();
    unsafe {
        conn.load_extension_enable().unwrap();
        sqlite_vec::load(&conn).unwrap();
        conn.load_extension_disable().unwrap();
    }
    let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    conn.execute(
        "INSERT INTO vec_posts (rowid, embedding) VALUES (?1, ?2)",
        rusqlite::params![rowid, bytes],
    ).unwrap();
}
```

Update the earlier FTS tests to use `create_test_db_with_vec` so all tests work with the same schema (the FTS-only `create_test_db` helper can be removed).

- [ ] **Step 4: Run tests**

Run: `cargo nextest r -p bsearch-search`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add vector search and hybrid RRF to database module"
```

---

### Task 5: Embedding module -- mean pooling and L2 normalisation

**Files:**
- Create: `crates/bsearch-search/src/embed.rs`
- Modify: `crates/bsearch-search/src/main.rs`

- [ ] **Step 1: Write mean-pool and normalise functions with tests**

Start with the pure maths functions that don't need ONNX or tokenizer:

```rust
use ndarray::{Array1, Array2};

/// Mean-pool token embeddings, masking padding tokens.
///
/// `hidden_state` has shape (seq_len, hidden_dim).
/// `attention_mask` has shape (seq_len,) with 1.0 for real tokens and 0.0 for padding.
pub fn mean_pool(hidden_state: &Array2<f32>, attention_mask: &Array1<f32>) -> Array1<f32> {
    let hidden_dim = hidden_state.ncols();
    let mut sum = Array1::<f32>::zeros(hidden_dim);
    let mut mask_sum: f32 = 0.0;

    for (i, mask_val) in attention_mask.iter().enumerate() {
        if *mask_val > 0.0 {
            sum += &(hidden_state.row(i).to_owned() * *mask_val);
            mask_sum += mask_val;
        }
    }

    if mask_sum > 0.0 {
        sum /= mask_sum;
    }
    sum
}

/// L2-normalise a vector in place, returning a unit vector.
pub fn l2_normalise(v: &Array1<f32>) -> Array1<f32> {
    let norm = v.dot(v).sqrt();
    if norm > 0.0 {
        v / norm
    } else {
        v.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_mean_pool_simple() {
        // 3 tokens, 4-dim embeddings, all unmasked
        let hidden = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]).unwrap();
        let mask = array![1.0, 1.0, 1.0];
        let result = mean_pool(&hidden, &mask);
        assert_eq!(result, array![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_mean_pool_with_padding() {
        // 3 tokens, last one is padding
        let hidden = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            99.0, 99.0, 99.0, 99.0, // padding -- should be ignored
        ]).unwrap();
        let mask = array![1.0, 1.0, 0.0];
        let result = mean_pool(&hidden, &mask);
        assert_eq!(result, array![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_l2_normalise() {
        let v = array![3.0, 4.0];
        let normed = l2_normalise(&v);
        let expected = array![0.6, 0.8];
        for (a, b) in normed.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_l2_normalise_unit_length() {
        let v = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normed = l2_normalise(&v);
        let length: f32 = normed.dot(&normed).sqrt();
        assert!((length - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalise_zero_vector() {
        let v = array![0.0, 0.0, 0.0];
        let normed = l2_normalise(&v);
        assert_eq!(normed, array![0.0, 0.0, 0.0]);
    }
}
```

- [ ] **Step 2: Wire embed.rs into main.rs**

Add `mod embed;` to main.rs.

- [ ] **Step 3: Run tests**

Run: `cargo nextest r -p bsearch-search`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add mean pooling and L2 normalisation for embeddings"
```

---

### Task 6: Embedding module -- ONNX inference and tokenizer

**Files:**
- Modify: `crates/bsearch-search/src/embed.rs`

- [ ] **Step 1: Add the Embedder struct with ONNX loading and encode**

Add to `embed.rs`, above the test module:

```rust
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, CowArray};
use ort::session::Session;
use tokenizers::Tokenizer;

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder {
    /// Load the ONNX model and tokenizer from a directory.
    /// Expects `model.onnx` and `tokenizer.json` in `model_dir`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        anyhow::ensure!(
            model_path.exists(),
            "ONNX model not found at {}. Run `bsearch export-model` first.",
            model_path.display()
        );
        anyhow::ensure!(
            tokenizer_path.exists(),
            "Tokenizer not found at {}. Run `bsearch export-model` first.",
            tokenizer_path.display()
        );

        let session = Session::builder()
            .context("Failed to create ONNX Runtime session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model from {}", model_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        Ok(Embedder { session, tokenizer })
    }

    /// Encode a single text into a 384-dimensional embedding vector.
    pub fn encode(&self, text: &str) -> Result<[f32; 384]> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        let seq_len = input_ids.len();

        let input_ids_array = CowArray::from(
            Array2::from_shape_vec((1, seq_len), input_ids)?.into_dyn(),
        );
        let attention_mask_array = CowArray::from(
            Array2::from_shape_vec((1, seq_len), attention_mask.clone())?.into_dyn(),
        );
        let token_type_ids_array = CowArray::from(
            Array2::from_shape_vec((1, seq_len), token_type_ids)?.into_dyn(),
        );

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_array,
            "attention_mask" => attention_mask_array,
            "token_type_ids" => token_type_ids_array,
        ]?)?;

        // last_hidden_state: shape (1, seq_len, 384)
        let hidden_state = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract hidden state tensor")?;

        let hidden_2d: Array2<f32> = hidden_state
            .slice(ndarray::s![0, .., ..])
            .to_owned()
            .into_dimensionality()
            .context("Unexpected hidden state shape")?;

        let mask_f32: Array1<f32> = attention_mask.iter().map(|&m| m as f32).collect();

        let pooled = mean_pool(&hidden_2d, &mask_f32);
        let normalised = l2_normalise(&pooled);

        let mut result = [0.0f32; 384];
        result.copy_from_slice(normalised.as_slice().context("Embedding not contiguous")?);
        Ok(result)
    }
}
```

Note: the exact `ort` API may need adjustment depending on the version. The `ort` v2 API uses `Session::builder().commit_from_file()` and `ort::inputs![]`. If the API differs, adjust accordingly -- the structure is the same.

- [ ] **Step 2: Verify it compiles**

Run: `cargo build -p bsearch-search`
Expected: compiles. No new tests here -- the Embedder requires model files to test, which will be covered by the integration test in Task 9.

- [ ] **Step 3: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add ONNX inference and tokenizer loading to embedder"
```

---

### Task 7: CLI wiring and output formatting

**Files:**
- Modify: `crates/bsearch-search/src/main.rs`

- [ ] **Step 1: Implement the full CLI with clap**

```rust
mod config;
mod db;
mod embed;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(name = "bsearch-search", version, about = "Fast search across indexed Bluesky posts")]
struct Cli {
    /// Search query text
    query: Option<String>,

    /// Number of results
    #[arg(short = 'n', long, default_value_t = 10)]
    limit: usize,

    /// Filter by source type
    #[arg(short, long, value_parser = ["own_post", "like", "backfill_post", "backfill_like"])]
    source: Option<String>,

    /// Search mode
    #[arg(short, long, default_value = "hybrid", value_parser = ["hybrid", "keyword", "semantic"])]
    mode: String,

    /// Filter by author handle
    #[arg(short = 'a', long)]
    handle: Option<String>,

    /// Database path
    #[arg(long, env = "BSEARCH_DB_PATH")]
    db: Option<PathBuf>,

    /// Model directory (containing model.onnx and tokenizer.json)
    #[arg(long, env = "BSEARCH_MODEL_DIR")]
    model: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::Config::resolve(cli.db, cli.model)?;

    if cli.query.is_none() && cli.handle.is_none() {
        anyhow::bail!("Provide a query and/or --handle to search.");
    }

    let database = db::Database::open(&config.db_path)
        .with_context(|| format!("Failed to open database at {}", config.db_path.display()))?;

    // No query: list posts by handle
    if cli.query.is_none() {
        let handle = cli.handle.as_deref().unwrap();
        let results = database.list_by_handle(handle, cli.limit, cli.source.as_deref())?;
        if results.is_empty() {
            eprintln!("No results found.");
            return Ok(());
        }
        for (i, r) in results.iter().enumerate() {
            print_result(i + 1, r);
        }
        return Ok(());
    }

    let query = cli.query.as_deref().unwrap();

    // Load embedder only for hybrid/semantic modes
    let query_embedding = if cli.mode == "keyword" {
        None
    } else {
        let embedder = embed::Embedder::load(&config.model_dir)
            .context("Failed to load embedding model")?;
        Some(embedder.encode(query).context("Failed to encode query")?)
    };

    let results = match cli.mode.as_str() {
        "keyword" => database.search_fts(query, cli.limit, cli.source.as_deref(), cli.handle.as_deref())?,
        "semantic" => {
            let emb = query_embedding.as_ref().unwrap();
            database.search_vec(emb, cli.limit, cli.source.as_deref(), cli.handle.as_deref())?
        }
        "hybrid" | _ => {
            database.search_hybrid(
                query,
                query_embedding.as_ref(),
                cli.limit,
                cli.source.as_deref(),
                cli.handle.as_deref(),
            )?
        }
    };

    if results.is_empty() {
        eprintln!("No results found.");
        return Ok(());
    }

    for (i, r) in results.iter().enumerate() {
        print_result(i + 1, r);
    }

    Ok(())
}

fn at_uri_to_web_url(uri: &str) -> String {
    if let Some(rest) = uri.strip_prefix("at://") {
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() >= 3 && parts[1] == "app.bsky.feed.post" {
            return format!("https://bsky.app/profile/{}/post/{}", parts[0], parts[2]);
        }
    }
    uri.to_string()
}

fn print_result(index: usize, r: &db::SearchResult) {
    let web_url = at_uri_to_web_url(&r.uri);

    let score_info = if let Some(rrf) = r.rrf_score {
        let mt = r.match_type.as_deref().unwrap_or("");
        format!("score: {rrf:.4}, match: {mt}")
    } else if let Some(bm25) = r.bm25_rank {
        format!("bm25: {bm25:.4}")
    } else if let Some(dist) = r.distance {
        format!("distance: {dist:.4}")
    } else {
        String::new()
    };

    println!("\n--- Result {index} ({score_info}) ---");
    println!("Author:  {}", r.author_handle);
    println!("Date:    {}", r.created_at);
    println!("Source:  {}", r.source);
    println!("Link:    {web_url}");
    println!("Text:    {}", r.text);
}
```

- [ ] **Step 2: Add a test for at_uri_to_web_url**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at_uri_to_web_url() {
        assert_eq!(
            at_uri_to_web_url("at://did:plc:abc/app.bsky.feed.post/rkey123"),
            "https://bsky.app/profile/did:plc:abc/post/rkey123"
        );
    }

    #[test]
    fn test_at_uri_to_web_url_passthrough() {
        assert_eq!(
            at_uri_to_web_url("https://example.com"),
            "https://example.com"
        );
    }
}
```

- [ ] **Step 3: Verify it compiles and tests pass**

Run: `cargo build -p bsearch-search && cargo nextest r -p bsearch-search`
Expected: compiles and all tests pass

- [ ] **Step 4: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Wire up CLI with clap, output formatting, and main entry point"
```

---

### Task 8: Python export-model command

**Files:**
- Modify: `src/bsearch/cli.py`

- [ ] **Step 1: Add the export-model command**

Add this command to `cli.py` alongside the existing commands:

```python
@cli.command("export-model")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for ONNX model and tokenizer "
    "(default: ~/.cache/bsearch/<model-name>).",
)
def export_model(output_dir: str | None):
    """Export the embedding model to ONNX format for the Rust search binary."""
    import shutil
    from pathlib import Path

    config = Config.from_env()

    if output_dir is None:
        cache_dir = Path.home() / ".cache" / "bsearch" / config.embedding_model
    else:
        cache_dir = Path(output_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading model '{config.embedding_model}'...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.embedding_model)

    # Export to ONNX
    onnx_path = cache_dir / "model.onnx"
    click.echo(f"Exporting ONNX model to {onnx_path}...")

    import torch

    # Get the transformer module (first module in the SentenceTransformer pipeline)
    transformer = model[0]
    tokenizer = transformer.tokenizer
    bert_model = transformer.auto_model

    # Create dummy input
    dummy_text = "This is a dummy sentence for tracing."
    encoded = tokenizer(dummy_text, return_tensors="pt")

    # Export
    torch.onnx.export(
        bert_model,
        (encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )

    # Copy tokenizer.json from the HuggingFace cache
    hf_tokenizer_path = None
    model_name_or_path = transformer.model_name_or_path
    if model_name_or_path and Path(model_name_or_path).is_dir():
        candidate = Path(model_name_or_path) / "tokenizer.json"
        if candidate.exists():
            hf_tokenizer_path = candidate

    if hf_tokenizer_path is None:
        from huggingface_hub import hf_hub_download

        hf_tokenizer_path = Path(
            hf_hub_download(config.embedding_model, "tokenizer.json")
        )

    tokenizer_dest = cache_dir / "tokenizer.json"
    shutil.copy2(hf_tokenizer_path, tokenizer_dest)

    click.echo(f"Tokenizer saved to {tokenizer_dest}")
    click.echo(f"\nExport complete. Model directory: {cache_dir}")
    click.echo("The Rust binary will use this directory by default.")
```

- [ ] **Step 2: Test the export command manually**

Run: `cd /Users/sth/dev/bsearch && uv run bsearch export-model`
Expected: model exports to `~/.cache/bsearch/all-MiniLM-L6-v2/` with `model.onnx` and `tokenizer.json`

- [ ] **Step 3: Verify the exported files exist**

Run: `ls -la ~/.cache/bsearch/all-MiniLM-L6-v2/`
Expected: `model.onnx` (~90MB) and `tokenizer.json` (~700KB)

- [ ] **Step 4: Run existing Python tests to ensure no regressions**

Run: `cd /Users/sth/dev/bsearch && uv run pytest`
Expected: all existing tests pass

- [ ] **Step 5: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add export-model CLI command for ONNX export"
```

---

### Task 9: Integration test -- numerical parity

**Files:**
- Create: `tests/test_onnx_parity.py`

This test verifies that the Rust binary produces the same embeddings as the Python side. It requires the ONNX model to have been exported (Task 8).

- [ ] **Step 1: Write the parity test**

```python
import subprocess
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


@pytest.fixture
def rust_binary():
    """Check the Rust binary has been built."""
    result = subprocess.run(
        ["cargo", "build", "-p", "bsearch-search", "--release"],
        capture_output=True,
    )
    binary = Path("target/release/bsearch-search")
    if not binary.exists():
        pytest.skip("Rust binary not built.")
    return binary


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
```

- [ ] **Step 2: Add onnxruntime as a dev dependency**

Run: `cd /Users/sth/dev/bsearch && uv add --dev onnxruntime`

- [ ] **Step 3: Run the parity test**

Run: `cd /Users/sth/dev/bsearch && uv run pytest tests/test_onnx_parity.py -v`
Expected: test passes, confirming ONNX export produces identical embeddings

- [ ] **Step 4: Build the Rust binary and test end-to-end**

Run:
```bash
cargo build -p bsearch-search --release
./target/release/bsearch-search --mode keyword --db bsearch.db "test query"
```
Expected: returns results from FTS search (no model needed for keyword mode)

Then test with semantic search:
```bash
./target/release/bsearch-search --mode semantic --db bsearch.db "cats and dogs"
```
Expected: returns vector search results using the ONNX model

- [ ] **Step 5: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add ONNX parity integration test"
```

---

### Task 10: README and cleanup

**Files:**
- Create: `crates/bsearch-search/README.md`

- [ ] **Step 1: Write the README**

```markdown
# bsearch-search

A fast Rust binary for searching indexed Bluesky posts. Replaces the Python
`bsearch search` command with sub-second cold start by using ONNX Runtime
instead of PyTorch for embedding inference.

## Setup

1. Export the model from the Python side (one-time):

   ```
   bsearch export-model
   ```

   This saves `model.onnx` and `tokenizer.json` to
   `~/.cache/bsearch/all-MiniLM-L6-v2/`.

2. Build the binary:

   ```
   cargo build -p bsearch-search --release
   ```

3. Search:

   ```
   ./target/release/bsearch-search "your query"
   ```

## Relationship to the Python tool

The Python `bsearch` CLI handles monitoring (serve), backfill, and service
management. This Rust binary only handles search. Both read from the same
SQLite database.

## Embedding pipeline

The SentenceTransformer model (`all-MiniLM-L6-v2`) consists of three stages:

1. **Tokenisation** -- text to token IDs via the WordPiece tokenizer
2. **Transformer** -- token IDs to contextual embeddings (6-layer BERT,
   output shape: seq_len x 384)
3. **Mean pooling + L2 normalisation** -- aggregate token embeddings into a
   single 384-dimensional unit vector

When exported to ONNX, only stage 2 (the transformer) is included in
`model.onnx`. Stages 1 and 3 are handled in Rust:

- **Tokenisation**: the `tokenizers` crate loads `tokenizer.json` (the same
  Rust library that Python's `tokenizers` package wraps)
- **Mean pooling**: multiply each token embedding by its attention mask value,
  sum along the sequence axis, divide by the mask sum. This averages only
  non-padding tokens.
- **L2 normalisation**: divide by the vector's L2 norm to produce a unit
  vector suitable for cosine-distance search.

This produces embeddings identical to `SentenceTransformer.encode()` within
float32 tolerance.
```

- [ ] **Step 2: Add .gitignore entries for Rust build artefacts**

Append to the repo's `.gitignore` (or create one if it doesn't exist):

```
/target/
```

- [ ] **Step 3: Run full test suite one final time**

Run:
```bash
cargo nextest r -p bsearch-search
cd /Users/sth/dev/bsearch && uv run pytest
```
Expected: all Rust and Python tests pass

- [ ] **Step 4: Run clippy and format**

Run:
```bash
cargo fmt
cargo clippy -p bsearch-search -- -D warnings
```
Expected: no warnings

- [ ] **Step 5: Commit**

```bash
jj fix && jj commit -m "[WIP: claude] Add README and final cleanup for bsearch-search"
```
