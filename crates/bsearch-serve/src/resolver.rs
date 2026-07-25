use anyhow::Context;
use anyhow::Result;
use atrium_api::agent::atp_agent::store::MemorySessionStore;
use atrium_api::agent::atp_agent::AtpAgent;
use atrium_api::app::bsky::feed::defs::PostView;
use atrium_xrpc_client::reqwest::ReqwestClient;
use bsearch_core::models::parse_created_at;
use bsearch_core::models::Post;
use bsearch_core::models::Source;

use crate::config::Config;

/// The AT Protocol caps `getPosts` at 25 URIs per call.
pub const MAX_URIS_PER_CALL: usize = 25;

/// AT Protocol client for turning liked-post references into full posts.
///
/// Mirrors the subset of `ATProtoResolver` in `src/bsearch/resolver.py` that
/// the daemon needs; backfill remains in the Python CLI.
pub struct Resolver {
    agent: AtpAgent<MemorySessionStore, ReqwestClient>,
}

impl Resolver {
    pub fn new(config: &Config) -> Self {
        Self {
            agent: AtpAgent::new(
                ReqwestClient::new(config.pds_url.clone()),
                MemorySessionStore::default(),
            ),
        }
    }

    /// Authenticate, returning the DID of the logged-in account.
    ///
    /// `createSession` already reports the account's DID, so unlike the Python
    /// code there is no need for a follow-up `resolveHandle` call.
    pub async fn login(&self, config: &Config) -> Result<String> {
        let session = self
            .agent
            .login(&config.handle, &config.app_password)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("Failed to log in as {}", config.handle))?;
        tracing::info!(handle = %config.handle, "Logged in");
        Ok(session.did.as_str().to_string())
    }

    /// Resolve liked-post URIs into posts, in chunks of [`MAX_URIS_PER_CALL`].
    ///
    /// A failing chunk is logged and skipped rather than aborting the batch,
    /// matching the Python behaviour; the caller re-queues on a hard error.
    pub async fn resolve_post_uris(&self, uris: &[String]) -> Result<Vec<Post>> {
        let mut posts = Vec::new();
        for chunk in uris.chunks(MAX_URIS_PER_CALL) {
            let params = atrium_api::app::bsky::feed::get_posts::ParametersData {
                uris: chunk.to_vec(),
            };
            match self.agent.api.app.bsky.feed.get_posts(params.into()).await {
                Ok(output) => {
                    for view in &output.posts {
                        if let Some(post) = post_view_to_post(view, Source::Like) {
                            posts.push(post);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, uris = ?chunk, "Failed to resolve post URIs");
                }
            }
        }
        Ok(posts)
    }
}

/// Convert an AT Protocol `PostView` into our `Post`, or `None` when the
/// record carries no text (the Python code skips those too).
///
/// `record` is an untyped `Unknown`, so the text and timestamp are read out of
/// its JSON representation rather than a generated struct.
fn post_view_to_post(view: &PostView, source: Source) -> Option<Post> {
    let record = serde_json::to_value(&view.record).ok()?;
    let text = record.get("text")?.as_str()?;
    if text.is_empty() {
        return None;
    }

    let created_at = parse_created_at(record.get("createdAt").and_then(|v| v.as_str()));

    Some(Post::new(
        view.uri.clone(),
        view.cid.as_ref().to_string(),
        view.author.did.as_str().to_string(),
        view.author.handle.as_str().to_string(),
        text.to_string(),
        created_at,
        source,
    ))
}
