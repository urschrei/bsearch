use anyhow::Context;
use anyhow::Result;
use atrium_api::agent::atp_agent::AtpAgent;
use atrium_api::agent::atp_agent::store::MemorySessionStore;
use atrium_api::app::bsky::feed::defs::PostView;
use atrium_xrpc_client::reqwest::ReqwestClient;
use bsearch_core::models::Post;
use bsearch_core::models::Source;
use bsearch_core::models::parse_created_at;

use crate::config::Config;
use crate::images::extract_images;
use crate::links::Link;
use crate::links::extract_links;

/// The AT Protocol caps `getPosts` at 25 URIs per call.
pub const MAX_URIS_PER_CALL: usize = 25;

/// A liked post together with the web links it carries.
#[derive(Debug, Clone)]
pub struct ResolvedPost {
    pub post: Post,
    pub links: Vec<Link>,
}

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
    pub async fn resolve_post_uris(&self, uris: &[String]) -> Result<Vec<ResolvedPost>> {
        let mut posts = Vec::new();
        for chunk in uris.chunks(MAX_URIS_PER_CALL) {
            let params = atrium_api::app::bsky::feed::get_posts::ParametersData {
                uris: chunk.to_vec(),
            };
            match self.agent.api.app.bsky.feed.get_posts(params.into()).await {
                Ok(output) => {
                    for view in &output.posts {
                        if let Some(post) = resolve_post_view(view, Source::Like) {
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

/// Convert an AT Protocol `PostView` into our `Post` plus its links, or
/// `None` when the record carries neither text nor images.
///
/// `record` is an untyped `Unknown`, so the text, timestamp, images and
/// links are read out of its JSON representation rather than a generated
/// struct.
fn resolve_post_view(view: &PostView, source: Source) -> Option<ResolvedPost> {
    let record = serde_json::to_value(&view.record).ok()?;
    let text = record.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let images = extract_images(&record);
    if text.is_empty() && images.is_empty() {
        return None;
    }

    let created_at = parse_created_at(record.get("createdAt").and_then(|v| v.as_str()));

    let post = Post::new(
        view.uri.clone(),
        view.cid.as_ref().to_string(),
        view.author.did.as_str().to_string(),
        view.author.handle.as_str().to_string(),
        text.to_string(),
        created_at,
        source,
    )
    .with_images(images);
    Some(ResolvedPost {
        post,
        links: extract_links(&record),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsearch_core::models::PostImage;

    /// A `getPosts` view as the AppView returns it, for an image-only post.
    fn image_post_view() -> PostView {
        serde_json::from_value(serde_json::json!({
            "uri": "at://did:plc:author/app.bsky.feed.post/3limg",
            "cid": "bafyreihv3ekzrewqgbnctxemlbb5fgdcyqymmhumftcyvx5yxsukp3x3wi",
            "author": {"did": "did:plc:author", "handle": "author.bsky.social"},
            "record": {
                "$type": "app.bsky.feed.post",
                "text": "",
                "createdAt": "2026-11-20T10:00:00.000Z",
                "embed": {
                    "$type": "app.bsky.embed.images",
                    "images": [{
                        "alt": "A chart of ferret sightings",
                        "image": {
                            "$type": "blob",
                            "ref": {"$link": "bafkreigh2akiscaildcqabsyg3dfr6chu3fgpregiymsck7e7aqa4s52zy"},
                            "mimeType": "image/jpeg",
                            "size": 345_678
                        }
                    }]
                }
            },
            "indexedAt": "2026-11-20T10:00:01.000Z"
        }))
        .expect("fixture should deserialise as a PostView")
    }

    #[test]
    fn test_image_only_post_is_resolved_with_its_images() {
        let resolved =
            resolve_post_view(&image_post_view(), Source::Like).expect("post should resolve");
        assert_eq!(resolved.post.text, "");
        assert_eq!(
            resolved.post.images,
            vec![PostImage {
                blob_cid: "bafkreigh2akiscaildcqabsyg3dfr6chu3fgpregiymsck7e7aqa4s52zy".to_string(),
                alt: "A chart of ferret sightings".to_string(),
            }]
        );
    }
}
