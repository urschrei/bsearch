//! Extraction of web links from a post record.
//!
//! A post carries links in two places: as `app.bsky.richtext.facet#link`
//! facets over its text, and as an `app.bsky.embed.external` link card,
//! either directly or as the media half of `app.bsky.embed.recordWithMedia`.
//! Clients write a facet for every URL they detect, so bare text is not
//! scanned.

use bsearch_core::models::PendingLink;
use bsearch_core::models::Post;
use serde_json::Value;

const FACET_LINK: &str = "app.bsky.richtext.facet#link";
const EMBED_EXTERNAL: &str = "app.bsky.embed.external";
const EMBED_RECORD_WITH_MEDIA: &str = "app.bsky.embed.recordWithMedia";

/// A web link found in a post.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub url: String,
    /// The page title, when the link came from a link card.
    pub title: Option<String>,
}

/// Every distinct `http(s)` link in `record`, link card first.
///
/// The card goes first because it is the one link that has a title; a
/// facet for the same URL is then absorbed as a duplicate rather than
/// adding a second, title-less entry.
pub fn extract_links(record: &Value) -> Vec<Link> {
    let mut links: Vec<Link> = Vec::new();
    let mut push = |url: &str, title: Option<&str>| {
        if !is_web_url(url) || links.iter().any(|l| l.url == url) {
            return;
        }
        links.push(Link {
            url: url.to_string(),
            title: title.filter(|t| !t.is_empty()).map(str::to_string),
        });
    };

    if let Some(external) = record.get("embed").and_then(external_of_embed) {
        if let Some(url) = external.get("uri").and_then(Value::as_str) {
            push(url, external.get("title").and_then(Value::as_str));
        }
    }

    let features = record
        .get("facets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|facet| facet.get("features").and_then(Value::as_array))
        .flatten();
    for feature in features {
        if feature.get("$type").and_then(Value::as_str) != Some(FACET_LINK) {
            continue;
        }
        if let Some(url) = feature.get("uri").and_then(Value::as_str) {
            push(url, None);
        }
    }

    links
}

/// Prepare a link found in `post` for submission, with the post's author
/// and text as the bookmark description.
pub fn to_pending(post: &Post, link: &Link) -> PendingLink {
    PendingLink {
        url: link.url.clone(),
        post_uri: post.uri.clone(),
        title: link.title.clone(),
        description: format!("@{}: {}", post.author_handle, post.text),
    }
}

/// The `external` object of a link-card embed, looking through a
/// record-with-media wrapper if there is one.
fn external_of_embed(embed: &Value) -> Option<&Value> {
    match embed.get("$type").and_then(Value::as_str)? {
        EMBED_EXTERNAL => embed.get("external"),
        EMBED_RECORD_WITH_MEDIA => embed.get("media").and_then(external_of_embed),
        _ => None,
    }
}

fn is_web_url(url: &str) -> bool {
    url.starts_with("https://") || url.starts_with("http://")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn link_facet(uri: &str) -> Value {
        json!({
            "index": {"byteStart": 0, "byteEnd": 10},
            "features": [{"$type": FACET_LINK, "uri": uri}]
        })
    }

    #[test]
    fn test_link_facet_is_extracted() {
        let record = json!({
            "text": "see example.com",
            "facets": [link_facet("https://example.com/article")]
        });
        assert_eq!(
            extract_links(&record),
            vec![Link {
                url: "https://example.com/article".to_string(),
                title: None
            }]
        );
    }

    #[test]
    fn test_external_embed_supplies_title() {
        let record = json!({
            "text": "a card",
            "embed": {
                "$type": EMBED_EXTERNAL,
                "external": {
                    "uri": "https://example.com/card",
                    "title": "Card title",
                    "description": "ignored"
                }
            }
        });
        assert_eq!(
            extract_links(&record),
            vec![Link {
                url: "https://example.com/card".to_string(),
                title: Some("Card title".to_string())
            }]
        );
    }

    #[test]
    fn test_record_with_media_is_looked_through() {
        let record = json!({
            "text": "a quote with a card",
            "embed": {
                "$type": EMBED_RECORD_WITH_MEDIA,
                "record": {"record": {"uri": "at://x/app.bsky.feed.post/y", "cid": "bafy"}},
                "media": {
                    "$type": EMBED_EXTERNAL,
                    "external": {"uri": "https://example.com/media", "title": "Media"}
                }
            }
        });
        let links = extract_links(&record);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].url, "https://example.com/media");
        assert_eq!(links[0].title.as_deref(), Some("Media"));
    }

    #[test]
    fn test_card_and_facet_for_the_same_url_yield_one_link_with_title() {
        // The common case: the client writes a facet for the URL in the
        // text and attaches a card for it as well.
        let record = json!({
            "text": "https://example.com/both",
            "facets": [link_facet("https://example.com/both")],
            "embed": {
                "$type": EMBED_EXTERNAL,
                "external": {"uri": "https://example.com/both", "title": "Both"}
            }
        });
        let links = extract_links(&record);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].title.as_deref(), Some("Both"));
    }

    #[test]
    fn test_other_facets_and_embeds_are_ignored() {
        let record = json!({
            "text": "@alice #tag",
            "facets": [
                {"index": {"byteStart": 0, "byteEnd": 6},
                 "features": [{"$type": "app.bsky.richtext.facet#mention", "did": "did:plc:a"}]},
                {"index": {"byteStart": 7, "byteEnd": 11},
                 "features": [{"$type": "app.bsky.richtext.facet#tag", "tag": "tag"}]}
            ],
            "embed": {
                "$type": "app.bsky.embed.images",
                "images": [{"alt": "", "image": {"$type": "blob"}}]
            }
        });
        assert!(extract_links(&record).is_empty());
    }

    #[test]
    fn test_non_web_schemes_are_dropped() {
        let record = json!({
            "facets": [link_facet("mailto:someone@example.com"), link_facet("ftp://x/y")]
        });
        assert!(extract_links(&record).is_empty());
    }

    #[test]
    fn test_empty_card_title_is_none() {
        let record = json!({
            "embed": {
                "$type": EMBED_EXTERNAL,
                "external": {"uri": "https://example.com/", "title": ""}
            }
        });
        assert_eq!(extract_links(&record)[0].title, None);
    }

    #[test]
    fn test_record_without_links_yields_nothing() {
        assert!(extract_links(&json!({"text": "plain"})).is_empty());
        assert!(extract_links(&Value::Null).is_empty());
    }

    #[test]
    fn test_to_pending_describes_the_post() {
        let post = Post::new(
            "at://did:plc:a/app.bsky.feed.post/1".to_string(),
            "cid".to_string(),
            "did:plc:a".to_string(),
            "alice.bsky.social".to_string(),
            "worth a read".to_string(),
            "2026-03-29T03:11:21+00:00".to_string(),
            bsearch_core::models::Source::Like,
        );
        let link = Link {
            url: "https://example.com/".to_string(),
            title: Some("Example".to_string()),
        };
        let pending = to_pending(&post, &link);
        assert_eq!(pending.url, "https://example.com/");
        assert_eq!(pending.post_uri, post.uri);
        assert_eq!(pending.title.as_deref(), Some("Example"));
        assert_eq!(pending.description, "@alice.bsky.social: worth a read");
    }

    #[test]
    fn test_links_keep_first_seen_order() {
        let record = json!({
            "facets": [link_facet("https://a.example/"), link_facet("https://b.example/")]
        });
        let urls: Vec<_> = extract_links(&record).into_iter().map(|l| l.url).collect();
        assert_eq!(urls, vec!["https://a.example/", "https://b.example/"]);
    }
}
