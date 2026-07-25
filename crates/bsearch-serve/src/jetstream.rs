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
