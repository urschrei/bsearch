//! Event ingestion: own posts are indexed immediately, likes are queued
//! for batch resolution because the event carries only a reference to the
//! liked post, not its text.

use std::sync::Arc;

use anyhow::Result;
use bsearch_core::db::Database;
use bsearch_core::models::parse_created_at;
use bsearch_core::models::Post;
use bsearch_core::models::Source;
use tokio::sync::Mutex;

use super::events::Event;
use super::events::Payload;
use super::replay::EventSink;

pub const COLLECTION_POST: &str = "app.bsky.feed.post";
pub const COLLECTION_LIKE: &str = "app.bsky.feed.like";

/// Handles events from either transport: the live tail and archive replay
/// deliver the same model, so ingestion cannot tell them apart.
///
/// Mirrors `Service._handle_event` in `src/bsearch/service.py`.
pub struct IngestHandler {
    db: Arc<Mutex<Database>>,
    handle: String,
}

impl IngestHandler {
    pub fn new(db: Arc<Mutex<Database>>, handle: String) -> Self {
        Self { db, handle }
    }

    async fn handle_own_post(
        &self,
        did: &str,
        rkey: &str,
        cid: &str,
        record: &serde_json::Value,
    ) -> Result<()> {
        let text = record.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if text.is_empty() {
            return Ok(());
        }

        let created_at = parse_created_at(record.get("createdAt").and_then(|v| v.as_str()));
        let uri = format!("at://{did}/{COLLECTION_POST}/{rkey}");

        let post = Post::new(
            uri.clone(),
            cid.to_string(),
            did.to_string(),
            self.handle.clone(),
            text.to_string(),
            created_at,
            Source::OwnPost,
        );

        let inserted = {
            let db = self.db.lock().await;
            db.insert_post(&post)?
        };
        if inserted.is_some() {
            tracing::info!(uri = %uri, "Indexed own post");
        }
        Ok(())
    }

    /// Record a like for later resolution.
    ///
    /// The write is durable and its failure is propagated, so that the
    /// cursor is not advanced past a like we did not manage to record. The
    /// event is replayed instead, which is harmless: the queue ignores a
    /// duplicate URI.
    async fn handle_like(&self, record: &serde_json::Value) -> Result<()> {
        let uri = record
            .get("subject")
            .and_then(|s| s.get("uri"))
            .and_then(|v| v.as_str());
        if let Some(uri) = uri {
            let db = self.db.lock().await;
            db.queue_pending_like(uri)?;
            tracing::debug!(uri = %uri, "Queued like for resolution");
        }
        Ok(())
    }

    /// Handle one event and advance the cursor past it.
    ///
    /// Deletes, updates, identity, account and sync events are not
    /// indexed, matching the `operation != "create"` filter in the Python
    /// client -- but they do still move the cursor, below.
    pub async fn handle_event(&self, event: &Event) -> Result<()> {
        if let Payload::Commit(commit) = &event.payload {
            if commit.operation == "create" {
                if let Some(record) = &commit.record {
                    let cid = commit.cid.as_deref().unwrap_or("");
                    match commit.collection.as_str() {
                        COLLECTION_POST => {
                            self.handle_own_post(&event.did, &commit.rkey, cid, record)
                                .await?
                        }
                        COLLECTION_LIKE => self.handle_like(record).await?,
                        _ => {}
                    }
                }
            }
        }

        // Advance the cursor for every event delivered, not only the ones
        // that produced a row. The cursor records how far through the
        // stream we have read, and an event we deliberately ignored has
        // still been read. Reached only after handling succeeded, so a
        // crash mid-handling replays the event rather than skipping it.
        // The stored value is a v2 seq; both transports redeliver the
        // boundary event on resume and ingestion is idempotent.
        {
            let db = self.db.lock().await;
            db.set_cursor(event.seq as i64)?;
        }
        Ok(())
    }
}

impl EventSink for IngestHandler {
    async fn apply(&self, event: &Event) -> Result<()> {
        self.handle_event(event).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jetstream::events::Commit;
    use bsearch_core::db::Order;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    const DID: &str = "did:plc:aefyqfi5jdig6vjfa73debzc";

    fn commit_event(collection: &str, operation: &str, record: serde_json::Value) -> Event {
        commit_event_seq(4_242, collection, operation, record)
    }

    fn commit_event_seq(
        seq: u64,
        collection: &str,
        operation: &str,
        record: serde_json::Value,
    ) -> Event {
        let is_create = operation == "create";
        Event {
            seq,
            did: DID.to_string(),
            time_us: 1_785_009_595_019_071,
            payload: Payload::Commit(Commit {
                operation: operation.to_string(),
                collection: collection.to_string(),
                rkey: "3lxyz123".to_string(),
                rev: "3mrir7qvgmk2o".to_string(),
                record: is_create.then_some(record),
                cid: is_create.then(|| "bafyreiabc123".to_string()),
            }),
        }
    }

    /// The handler under test plus the database it writes through. The
    /// temp file is held so the database outlives the test, and its path
    /// is kept so a test can reopen it as a restarted daemon would.
    struct Harness {
        _file: NamedTempFile,
        path: PathBuf,
        handler: IngestHandler,
        db: Arc<Mutex<Database>>,
    }

    fn harness() -> Harness {
        let file = NamedTempFile::new().expect("temp file");
        let path = file.path().to_path_buf();
        let db = Database::open_read_write(&path).expect("open db");
        let db = Arc::new(Mutex::new(db));
        let handler = IngestHandler::new(db.clone(), "alice.bsky.social".to_string());
        Harness {
            _file: file,
            path,
            handler,
            db,
        }
    }

    #[tokio::test]
    async fn test_own_post_is_indexed_with_python_compatible_fields() {
        let h = harness();
        let event = commit_event(
            COLLECTION_POST,
            "create",
            serde_json::json!({"text": "a post about ferrets", "createdAt": "2026-07-25T21:01:36.005Z"}),
        );

        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        let results = db
            .search_fts("ferrets", 10, None, None, Order::Relevance)
            .expect("search failed");
        assert_eq!(
            results.len(),
            1,
            "post should be indexed and FTS-searchable"
        );
        let row = &results[0];
        assert_eq!(row.uri, format!("at://{DID}/{COLLECTION_POST}/3lxyz123"));
        assert_eq!(row.source, "own_post");
        assert_eq!(row.author_handle, "alice.bsky.social");
        assert_eq!(row.author_did, DID);
        assert_eq!(row.cid, "bafyreiabc123");
        // Exactly what Python's datetime.isoformat() would have written.
        assert_eq!(row.created_at, "2026-07-25T21:01:36.005000+00:00");
        // Naive local time, so no offset.
        assert!(
            !row.indexed_at.contains('+'),
            "indexed_at should be naive: {}",
            row.indexed_at
        );

        // The cursor is now the event's seq, not its timestamp.
        assert_eq!(db.get_cursor().expect("cursor"), Some(4_242));
    }

    #[tokio::test]
    async fn test_ignored_event_still_advances_cursor() {
        // The bug this guards: an account whose only activity is outside
        // the indexed collections would leave the cursor parked, so every
        // reconnect replayed from further and further back.
        let h = harness();

        let event = commit_event(
            "app.bsky.graph.follow",
            "create",
            serde_json::json!({"subject": "did:plc:someone"}),
        );

        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        assert_eq!(db.get_cursor().expect("read failed"), Some(4_242));
        assert_eq!(
            db.search_fts("", 10, None, None, Order::Relevance)
                .expect("search failed")
                .len(),
            0,
            "the event should move the cursor without being indexed"
        );
    }

    #[tokio::test]
    async fn test_delete_event_advances_cursor() {
        let h = harness();
        let event = commit_event(COLLECTION_POST, "delete", serde_json::Value::Null);

        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        assert_eq!(db.get_cursor().expect("read failed"), Some(4_242));
    }

    #[tokio::test]
    async fn test_marker_event_advances_cursor() {
        // Identity, account and sync markers arrive even under a
        // collection filter; they must move the cursor like anything else.
        let h = harness();
        let event = Event {
            seq: 777,
            did: DID.to_string(),
            time_us: 1_785_009_595_019_071,
            payload: Payload::Identity,
        };

        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        assert_eq!(db.get_cursor().expect("read failed"), Some(777));
    }

    #[tokio::test]
    async fn test_seq_cursor_replaces_legacy_timestamp() {
        // A database carried over from the v1 daemon holds a time_us
        // cursor; the first handled v2 event overwrites it with a seq.
        let h = harness();
        {
            let db = h.db.lock().await;
            db.set_cursor(1_785_009_595_019_071).expect("seed cursor");
        }

        let event = commit_event(
            COLLECTION_POST,
            "create",
            serde_json::json!({"text": "migrated", "createdAt": "2026-07-25T21:01:36.005Z"}),
        );
        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        assert_eq!(db.get_cursor().expect("cursor"), Some(4_242));
    }

    #[tokio::test]
    async fn test_like_is_queued_not_indexed() {
        let h = harness();
        let liked = "at://did:plc:other/app.bsky.feed.post/abc";
        let event = commit_event(
            COLLECTION_LIKE,
            "create",
            serde_json::json!({"subject": {"uri": liked, "cid": "bafy"}}),
        );

        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        assert_eq!(
            db.take_pending_likes(10).expect("read queue"),
            vec![liked.to_string()]
        );
        assert!(
            db.get_posts_without_embeddings(10)
                .expect("query")
                .is_empty(),
            "the like itself must not be indexed; only the post it points at"
        );
    }

    #[tokio::test]
    async fn test_queued_like_survives_restart() {
        // The gap this closes: the cursor advances past a like as soon as
        // it is queued, so a queue that lived only in memory lost it on
        // restart with nothing left to say it had been there.
        let liked = "at://did:plc:other/app.bsky.feed.post/durable";
        let h = harness();

        h.handler
            .handle_event(&commit_event(
                COLLECTION_LIKE,
                "create",
                serde_json::json!({"subject": {"uri": liked, "cid": "bafy"}}),
            ))
            .await
            .expect("handle failed");

        // The cursor has already moved on, so the queue is the only record.
        {
            let db = h.db.lock().await;
            assert!(db.get_cursor().expect("cursor").is_some());
        }
        drop(h.db);

        let restarted = Database::open_read_write(&h.path).expect("reopen db");
        assert_eq!(
            restarted.take_pending_likes(10).expect("read queue"),
            vec![liked.to_string()],
            "a queued like must outlive the process that queued it"
        );
    }

    #[tokio::test]
    async fn test_replayed_like_is_queued_once() {
        let h = harness();
        let liked = "at://did:plc:other/app.bsky.feed.post/abc";
        let event = || {
            commit_event(
                COLLECTION_LIKE,
                "create",
                serde_json::json!({"subject": {"uri": liked, "cid": "bafy"}}),
            )
        };

        h.handler.handle_event(&event()).await.expect("first");
        h.handler.handle_event(&event()).await.expect("replay");

        let db = h.db.lock().await;
        assert_eq!(
            db.count_pending_likes().expect("count"),
            1,
            "a replayed event must not queue the same like twice"
        );
    }

    #[tokio::test]
    async fn test_non_create_operations_are_ignored() {
        let h = harness();
        let event = commit_event(
            COLLECTION_POST,
            "update",
            serde_json::json!({"text": "should not be indexed"}),
        );

        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        assert!(db
            .get_posts_without_embeddings(10)
            .expect("query")
            .is_empty());
        assert_eq!(
            db.get_cursor().expect("cursor"),
            Some(4_242),
            "an ignored event has still been read, so the cursor must advance"
        );
    }

    #[tokio::test]
    async fn test_post_without_text_is_skipped() {
        let h = harness();
        let event = commit_event(
            COLLECTION_POST,
            "create",
            serde_json::json!({"createdAt": "2026-07-25T21:01:36.005Z"}),
        );

        h.handler.handle_event(&event).await.expect("handle failed");

        let db = h.db.lock().await;
        assert!(db
            .get_posts_without_embeddings(10)
            .expect("query")
            .is_empty());
    }

    #[tokio::test]
    async fn test_replayed_event_does_not_duplicate() {
        let h = harness();
        let event = commit_event(
            COLLECTION_POST,
            "create",
            serde_json::json!({"text": "replayed post", "createdAt": "2026-07-25T21:01:36.005Z"}),
        );

        // Both transports redeliver the boundary event on resume.
        h.handler.handle_event(&event).await.expect("first");
        h.handler.handle_event(&event).await.expect("second");

        let db = h.db.lock().await;
        assert_eq!(
            db.get_posts_without_embeddings(10).expect("query").len(),
            1,
            "the UNIQUE uri constraint should absorb the replay"
        );
    }
}
