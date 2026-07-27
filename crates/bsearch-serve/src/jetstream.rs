use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use atproto_jetstream::CancellationToken;
use atproto_jetstream::Consumer;
use atproto_jetstream::ConsumerTaskConfig;
use atproto_jetstream::EventHandler;
use atproto_jetstream::JetstreamEvent;
use bsearch_core::db::Database;
use bsearch_core::models::parse_created_at;
use bsearch_core::models::Post;
use bsearch_core::models::Source;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::notify;

pub const COLLECTION_POST: &str = "app.bsky.feed.post";
pub const COLLECTION_LIKE: &str = "app.bsky.feed.like";

/// Handles Jetstream events: own posts are indexed immediately, likes are
/// queued for batch resolution because the event carries only a reference to
/// the liked post, not its text.
///
/// Mirrors `Service._handle_event` in `src/bsearch/service.py`.
pub struct IngestHandler {
    db: Arc<Mutex<Database>>,
    pending_likes: Arc<Mutex<VecDeque<String>>>,
    handle: String,
    id: String,
}

impl IngestHandler {
    pub fn new(
        db: Arc<Mutex<Database>>,
        pending_likes: Arc<Mutex<VecDeque<String>>>,
        handle: String,
    ) -> Self {
        Self {
            db,
            pending_likes,
            handle,
            id: "bsearch-ingest".to_string(),
        }
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

    async fn handle_like(&self, record: &serde_json::Value) {
        let uri = record
            .get("subject")
            .and_then(|s| s.get("uri"))
            .and_then(|v| v.as_str());
        if let Some(uri) = uri {
            self.pending_likes.lock().await.push_back(uri.to_string());
            tracing::debug!(uri = %uri, "Queued like for resolution");
        }
    }
}

#[async_trait]
impl EventHandler for IngestHandler {
    async fn handle_event(&self, event: Arc<JetstreamEvent>) -> Result<()> {
        // Deletes, identity and account events are all ignored, matching the
        // `operation != "create"` filter in the Python client.
        let JetstreamEvent::Commit {
            did,
            time_us,
            commit,
            ..
        } = event.as_ref()
        else {
            return Ok(());
        };

        if commit.operation != "create" {
            return Ok(());
        }

        match commit.collection.as_str() {
            COLLECTION_POST => {
                self.handle_own_post(did, &commit.rkey, &commit.cid, &commit.record)
                    .await?
            }
            COLLECTION_LIKE => self.handle_like(&commit.record).await,
            _ => return Ok(()),
        }

        // Advance the cursor only after the event has been handled, so a crash
        // mid-handling replays the event rather than skipping it.
        {
            let db = self.db.lock().await;
            db.set_cursor(*time_us as i64)?;
        }
        Ok(())
    }

    fn handler_id(&self) -> &str {
        &self.id
    }
}

/// Run the Jetstream consumer, reconnecting until cancelled.
///
/// `Consumer::run_background` returns when the socket closes rather than
/// reconnecting itself, so the retry loop lives here. Reconnecting rereads the
/// cursor from the database and rewinds it by
/// `reconnect_cursor_safety_seconds`, as `JetstreamClient.run` does in Python;
/// replayed events are harmless because `posts.uri` is UNIQUE and inserts use
/// INSERT OR IGNORE.
pub async fn run(
    config: &Config,
    db: Arc<Mutex<Database>>,
    handler: Arc<IngestHandler>,
    token: CancellationToken,
) -> Result<()> {
    while !token.is_cancelled() {
        let cursor = {
            let db = db.lock().await;
            db.get_cursor()?
        }
        .map(|c| c - config.reconnect_cursor_safety_seconds * 1_000_000);

        let task_config = ConsumerTaskConfig {
            user_agent: concat!("bsearch/", env!("CARGO_PKG_VERSION")).to_string(),
            // Compression would require shipping Jetstream's zstd dictionary,
            // and we are filtered to a single DID, so there is nothing to save.
            compression: false,
            zstd_dictionary_location: String::new(),
            jetstream_hostname: config.jetstream_hostname.clone(),
            collections: vec![COLLECTION_POST.to_string(), COLLECTION_LIKE.to_string()],
            dids: vec![config.did.clone()],
            max_message_size_bytes: None,
            cursor,
            // Send the filters in the connect URL rather than waiting to
            // negotiate them after connecting.
            require_hello: false,
        };

        let consumer = Consumer::new(task_config);
        consumer.register_handler(handler.clone()).await?;

        tracing::info!(cursor = ?cursor, host = %config.jetstream_hostname, "Connecting to Jetstream");

        match consumer.run_background(token.clone()).await {
            Ok(()) if token.is_cancelled() => break,
            Ok(()) => {
                tracing::warn!("Jetstream connection closed");
                notify("bsearch: disconnected", "Jetstream connection closed.");
            }
            Err(e) => {
                tracing::error!(error = ?e, "Jetstream connection error");
                notify("bsearch: error", "Jetstream connection error.");
            }
        }

        if token.is_cancelled() {
            break;
        }

        tracing::info!("Reconnecting in {} seconds...", RECONNECT_DELAY_SECS);
        tokio::select! {
            () = token.cancelled() => break,
            () = tokio::time::sleep(std::time::Duration::from_secs(RECONNECT_DELAY_SECS)) => {}
        }
    }

    Ok(())
}

const RECONNECT_DELAY_SECS: u64 = 5;

#[cfg(test)]
mod tests {
    use super::*;
    use atproto_jetstream::JetstreamEventCommit;
    use bsearch_core::db::Order;
    use tempfile::NamedTempFile;

    const DID: &str = "did:plc:aefyqfi5jdig6vjfa73debzc";

    fn commit_event(
        collection: &str,
        operation: &str,
        record: serde_json::Value,
    ) -> JetstreamEvent {
        JetstreamEvent::Commit {
            did: DID.to_string(),
            time_us: 1_785_009_595_019_071,
            kind: "commit".to_string(),
            commit: JetstreamEventCommit {
                rev: "3mrir7qvgmk2o".to_string(),
                operation: operation.to_string(),
                collection: collection.to_string(),
                rkey: "3lxyz123".to_string(),
                cid: "bafyreiabc123".to_string(),
                record,
            },
        }
    }

    /// The handler under test plus the shared state it writes through. The
    /// temp file is held so the database outlives the test.
    struct Harness {
        _file: NamedTempFile,
        handler: IngestHandler,
        db: Arc<Mutex<Database>>,
        queue: Arc<Mutex<VecDeque<String>>>,
    }

    fn harness() -> Harness {
        let file = NamedTempFile::new().expect("temp file");
        let db = Database::open_read_write(file.path()).expect("open db");
        let db = Arc::new(Mutex::new(db));
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let handler =
            IngestHandler::new(db.clone(), queue.clone(), "alice.bsky.social".to_string());
        Harness {
            _file: file,
            handler,
            db,
            queue,
        }
    }

    #[tokio::test]
    async fn test_own_post_is_indexed_with_python_compatible_fields() {
        let h = harness();
        let (handler, db) = (&h.handler, &h.db);
        let event = commit_event(
            COLLECTION_POST,
            "create",
            serde_json::json!({"text": "a post about ferrets", "createdAt": "2026-07-25T21:01:36.005Z"}),
        );

        handler
            .handle_event(Arc::new(event))
            .await
            .expect("handle failed");

        let db = db.lock().await;
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

        assert_eq!(
            db.get_cursor().expect("cursor"),
            Some(1_785_009_595_019_071)
        );
    }

    #[tokio::test]
    async fn test_like_is_queued_not_indexed() {
        let h = harness();
        let (handler, db, queue) = (&h.handler, &h.db, &h.queue);
        let liked = "at://did:plc:other/app.bsky.feed.post/abc";
        let event = commit_event(
            COLLECTION_LIKE,
            "create",
            serde_json::json!({"subject": {"uri": liked, "cid": "bafy"}}),
        );

        handler
            .handle_event(Arc::new(event))
            .await
            .expect("handle failed");

        assert_eq!(queue.lock().await.iter().collect::<Vec<_>>(), vec![liked]);
        let db = db.lock().await;
        assert!(
            db.get_posts_without_embeddings(10)
                .expect("query")
                .is_empty(),
            "the like itself must not be indexed; only the post it points at"
        );
    }

    #[tokio::test]
    async fn test_non_create_operations_are_ignored() {
        let h = harness();
        let (handler, db) = (&h.handler, &h.db);
        let event = commit_event(
            COLLECTION_POST,
            "delete",
            serde_json::json!({"text": "should not be indexed"}),
        );

        handler
            .handle_event(Arc::new(event))
            .await
            .expect("handle failed");

        let db = db.lock().await;
        assert!(db
            .get_posts_without_embeddings(10)
            .expect("query")
            .is_empty());
        assert_eq!(
            db.get_cursor().expect("cursor"),
            None,
            "an ignored event must not advance the cursor"
        );
    }

    #[tokio::test]
    async fn test_post_without_text_is_skipped() {
        let h = harness();
        let (handler, db) = (&h.handler, &h.db);
        let event = commit_event(
            COLLECTION_POST,
            "create",
            serde_json::json!({"createdAt": "2026-07-25T21:01:36.005Z"}),
        );

        handler
            .handle_event(Arc::new(event))
            .await
            .expect("handle failed");

        let db = db.lock().await;
        assert!(db
            .get_posts_without_embeddings(10)
            .expect("query")
            .is_empty());
    }

    #[tokio::test]
    async fn test_replayed_event_does_not_duplicate() {
        let h = harness();
        let (handler, db) = (&h.handler, &h.db);
        let event = Arc::new(commit_event(
            COLLECTION_POST,
            "create",
            serde_json::json!({"text": "replayed post", "createdAt": "2026-07-25T21:01:36.005Z"}),
        ));

        // Reconnecting rewinds the cursor, so the same event arrives twice.
        handler.handle_event(event.clone()).await.expect("first");
        handler.handle_event(event).await.expect("second");

        let db = db.lock().await;
        assert_eq!(
            db.get_posts_without_embeddings(10).expect("query").len(),
            1,
            "the UNIQUE uri constraint should absorb the replay"
        );
    }
}
