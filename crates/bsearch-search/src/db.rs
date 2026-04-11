use std::path::Path;

use anyhow::Result;
use rusqlite::{Connection, OpenFlags};

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
            Err(_) => {
                // Retry as phrase query by wrapping in double quotes
                let phrase = format!("\"{}\"", query.replace('"', ""));
                match self.run_fts_query(&phrase, limit, source_filter, handle_filter) {
                    Ok(results) => Ok(results),
                    Err(_) => Ok(vec![]),
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
                match_type: Some("fts".to_string()),
            })
        })?;

        let mut out = Vec::new();
        for r in results {
            out.push(r?);
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
                .to_string()
        } else {
            "SELECT p.id, p.uri, p.cid, p.author_did, p.author_handle, p.text,
                    p.created_at, p.source, p.indexed_at
             FROM posts p
             WHERE p.author_handle = ?1
             ORDER BY p.created_at DESC
             LIMIT ?2"
                .to_string()
        };

        let mut stmt = self.conn.prepare(&sql)?;

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

        let mut out = Vec::new();
        for r in results {
            out.push(r?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn create_test_db() -> (NamedTempFile, PathBuf) {
        let file = NamedTempFile::new().expect("failed to create temp file");
        let path = file.path().to_path_buf();

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
}
