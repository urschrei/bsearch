//! Extraction of embedded images from a post record.
//!
//! A post carries images in an `app.bsky.embed.images` embed, either
//! directly or as the media half of `app.bsky.embed.recordWithMedia`. Each
//! image is a blob reference with optional alt text.

use bsearch_core::models::PostImage;
use serde_json::Value;

const EMBED_IMAGES: &str = "app.bsky.embed.images";
const EMBED_RECORD_WITH_MEDIA: &str = "app.bsky.embed.recordWithMedia";

/// Every image embedded in `record`, in the order the post shows them.
///
/// An image without a usable blob reference is left out.
pub fn extract_images(record: &Value) -> Vec<PostImage> {
    let Some(images) = record.get("embed").and_then(images_of_embed) else {
        return Vec::new();
    };
    images
        .iter()
        .filter_map(|image| {
            let blob_cid = image.get("image").and_then(blob_cid)?;
            let alt = image.get("alt").and_then(Value::as_str).unwrap_or("");
            Some(PostImage {
                blob_cid: blob_cid.to_string(),
                alt: alt.to_string(),
            })
        })
        .collect()
}

/// The `images` array of an images embed, looking through a
/// record-with-media wrapper if there is one.
fn images_of_embed(embed: &Value) -> Option<&Vec<Value>> {
    match embed.get("$type").and_then(Value::as_str)? {
        EMBED_IMAGES => embed.get("images")?.as_array(),
        EMBED_RECORD_WITH_MEDIA => embed.get("media").and_then(images_of_embed),
        _ => None,
    }
}

/// The CID of a blob reference: `ref.$link` in the current blob form, or
/// `cid` in the legacy form that older records use.
fn blob_cid(blob: &Value) -> Option<&str> {
    blob.get("ref")
        .and_then(|r| r.get("$link"))
        .and_then(Value::as_str)
        .or_else(|| blob.get("cid").and_then(Value::as_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn image(cid: &str, alt: &str) -> Value {
        json!({
            "alt": alt,
            "image": {
                "$type": "blob",
                "ref": {"$link": cid},
                "mimeType": "image/jpeg",
                "size": 123_456
            },
            "aspectRatio": {"width": 1000, "height": 750}
        })
    }

    fn expected(cid: &str, alt: &str) -> PostImage {
        PostImage {
            blob_cid: cid.to_string(),
            alt: alt.to_string(),
        }
    }

    #[test]
    fn test_images_embed_is_extracted_in_order() {
        let record = json!({
            "text": "two pictures",
            "embed": {
                "$type": EMBED_IMAGES,
                "images": [image("bafkreia", "a heron"), image("bafkreib", "")]
            }
        });
        assert_eq!(
            extract_images(&record),
            vec![expected("bafkreia", "a heron"), expected("bafkreib", "")]
        );
    }

    #[test]
    fn test_images_inside_record_with_media_are_extracted() {
        let record = json!({
            "text": "quoting with a picture",
            "embed": {
                "$type": EMBED_RECORD_WITH_MEDIA,
                "record": {"record": {"uri": "at://did:plc:x/app.bsky.feed.post/y", "cid": "bafy"}},
                "media": {"$type": EMBED_IMAGES, "images": [image("bafkreia", "chart")]}
            }
        });
        assert_eq!(extract_images(&record), vec![expected("bafkreia", "chart")]);
    }

    #[test]
    fn test_legacy_blob_form_is_read() {
        let record = json!({
            "embed": {
                "$type": EMBED_IMAGES,
                "images": [{"alt": "old", "image": {"cid": "bafkreiold", "mimeType": "image/png"}}]
            }
        });
        assert_eq!(extract_images(&record), vec![expected("bafkreiold", "old")]);
    }

    #[test]
    fn test_missing_alt_is_empty() {
        let record = json!({
            "embed": {
                "$type": EMBED_IMAGES,
                "images": [{"image": {"$type": "blob", "ref": {"$link": "bafkreia"}}}]
            }
        });
        assert_eq!(extract_images(&record), vec![expected("bafkreia", "")]);
    }

    #[test]
    fn test_other_embeds_carry_no_images() {
        for embed in [
            json!({"$type": "app.bsky.embed.external", "external": {"uri": "https://example.com"}}),
            json!({"$type": "app.bsky.embed.video", "video": {"ref": {"$link": "bafkreiv"}}, "alt": "a clip"}),
            json!({"$type": "app.bsky.embed.record", "record": {"uri": "at://x", "cid": "bafy"}}),
        ] {
            let record = json!({"text": "t", "embed": embed});
            assert!(extract_images(&record).is_empty(), "{record}");
        }
        assert!(extract_images(&json!({"text": "no embed"})).is_empty());
    }

    #[test]
    fn test_image_without_blob_is_skipped() {
        let record = json!({
            "embed": {
                "$type": EMBED_IMAGES,
                "images": [{"alt": "broken"}, image("bafkreia", "fine")]
            }
        });
        assert_eq!(extract_images(&record), vec![expected("bafkreia", "fine")]);
    }
}
