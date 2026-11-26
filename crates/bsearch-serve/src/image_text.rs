//! Recognition of the text in post images.
//!
//! Images are queued in `post_images` when their post is indexed. This loop
//! fetches each one from the Bluesky CDN, recognises its text with Vision,
//! and stores the text, which adds it to the post's searchable image text.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bsearch_core::db::Database;
use bsearch_core::models::PendingImage;
use reqwest::StatusCode;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::ocr;

/// Full-size images, re-encoded as JPEG by the CDN. The path continues
/// with the author's DID and the blob CID.
const CDN_BASE: &str = "https://cdn.bsky.app/img/feed_fullsize/plain";
/// How many images to process per pass.
const BATCH_SIZE: usize = 10;
/// How many failed attempts an image gets before it leaves the queue
/// without text.
const MAX_ATTEMPTS: i64 = 5;
/// How long to wait after the CDN is unreachable or reports an error.
const RETRY_DELAY: Duration = Duration::from_secs(60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Why an image could not be fetched.
#[derive(Debug, thiserror::Error)]
enum FetchError {
    /// The CDN does not have the image, because the blob or its post was
    /// deleted. Retrying cannot succeed.
    #[error("image not available (HTTP {0})")]
    Gone(StatusCode),
    /// The CDN refused the request for some other reason.
    #[error("request refused (HTTP {0})")]
    Refused(StatusCode),
    /// The CDN is rate-limiting requests or failing.
    #[error("CDN unavailable (HTTP {0})")]
    Unavailable(StatusCode),
    /// No response at all: the network or the CDN is down.
    #[error("CDN unreachable: {0}")]
    Unreachable(reqwest::Error),
}

impl FetchError {
    fn from_status(status: StatusCode) -> Self {
        match status {
            StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND | StatusCode::GONE => {
                Self::Gone(status)
            }
            StatusCode::TOO_MANY_REQUESTS => Self::Unavailable(status),
            s if s.is_server_error() => Self::Unavailable(status),
            s => Self::Refused(s),
        }
    }
}

/// What to do with an image after one attempt at it.
enum Outcome {
    /// Store this text; an empty string records that there is none.
    Done(String),
    /// Count a failed attempt and go on to the next image.
    Failed(anyhow::Error),
    /// Count a failed attempt and pause the loop.
    FailedAndPause(anyhow::Error),
    /// Pause the loop without counting an attempt, because the failure
    /// says nothing about the image.
    Pause(anyhow::Error),
}

fn image_url(image: &PendingImage) -> String {
    format!("{CDN_BASE}/{}/{}@jpeg", image.author_did, image.blob_cid)
}

async fn fetch(client: &reqwest::Client, image: &PendingImage) -> Result<Vec<u8>, FetchError> {
    let response = client
        .get(image_url(image))
        .send()
        .await
        .map_err(FetchError::Unreachable)?;
    let status = response.status();
    if !status.is_success() {
        return Err(FetchError::from_status(status));
    }
    let bytes = response.bytes().await.map_err(FetchError::Unreachable)?;
    Ok(bytes.to_vec())
}

/// Fetch one image and recognise its text.
async fn process(client: &reqwest::Client, image: &PendingImage) -> Outcome {
    let bytes = match fetch(client, image).await {
        Ok(bytes) => bytes,
        Err(FetchError::Gone(status)) => {
            tracing::info!(cid = %image.blob_cid, %status, "Image no longer available");
            return Outcome::Done(String::new());
        }
        Err(e @ FetchError::Refused(_)) => return Outcome::Failed(e.into()),
        Err(e @ FetchError::Unavailable(_)) => return Outcome::FailedAndPause(e.into()),
        Err(e @ FetchError::Unreachable(_)) => return Outcome::Pause(e.into()),
    };
    // Recognition is CPU-bound and blocks, so it runs on the blocking pool
    // rather than on the single runtime thread.
    match tokio::task::spawn_blocking(move || ocr::recognise_text(&bytes)).await {
        Ok(Ok(text)) => Outcome::Done(text),
        Ok(Err(e)) => Outcome::Failed(e),
        Err(e) => Outcome::Failed(anyhow::anyhow!("text recognition task failed: {e}")),
    }
}

/// Periodically recognise the text in queued images.
///
/// The queue is drained like the others: an image is read, processed, and
/// only then marked as done, so a restart leaves it queued. An image that
/// fails is retried on later passes, up to [`MAX_ATTEMPTS`] times, after
/// which it leaves the queue without text; its post is then embedded with
/// the text it has. When the CDN cannot be reached the loop pauses without
/// counting an attempt, so a period offline does not use up the retries.
///
/// Like the Instapaper loop, this never returns early on error: a failure
/// is logged and waited out.
pub async fn run(
    interval: Duration,
    db: Arc<Mutex<Database>>,
    token: CancellationToken,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .user_agent(concat!("bsearch/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let mut delay = interval;

    loop {
        tokio::select! {
            () = token.cancelled() => break,
            () = tokio::time::sleep(delay) => {}
        }
        delay = interval;

        let pending = match db.lock().await.take_pending_images(BATCH_SIZE) {
            Ok(pending) => pending,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to read the image queue");
                delay = RETRY_DELAY;
                continue;
            }
        };

        let mut recognised = 0;
        let mut pause = false;
        for image in &pending {
            if token.is_cancelled() {
                break;
            }
            let outcome = process(&client, image).await;
            let db = db.lock().await;
            let stored = match outcome {
                Outcome::Done(text) => {
                    if !text.is_empty() {
                        recognised += 1;
                    }
                    db.store_image_text(image.post_id, image.position, &text)
                }
                Outcome::Failed(e) => {
                    tracing::warn!(cid = %image.blob_cid, error = %e, "Failed to read image text");
                    record_failure(&db, image)
                }
                Outcome::FailedAndPause(e) => {
                    tracing::warn!(error = %e, "Image fetch failed; pausing");
                    pause = true;
                    record_failure(&db, image)
                }
                Outcome::Pause(e) => {
                    tracing::warn!(error = %e, "Image fetch failed; pausing");
                    pause = true;
                    Ok(())
                }
            };
            if let Err(e) = stored {
                tracing::error!(error = ?e, post_id = image.post_id, "Failed to update the image queue");
                pause = true;
            }
            if pause {
                delay = RETRY_DELAY;
                break;
            }
        }
        if recognised > 0 {
            tracing::info!("Recognised text in {recognised} images");
        }
    }
    Ok(())
}

fn record_failure(db: &Database, image: &PendingImage) -> Result<()> {
    if db.record_image_failure(image.post_id, image.position, MAX_ATTEMPTS)? {
        tracing::warn!(cid = %image.blob_cid, "Giving up on image after {MAX_ATTEMPTS} attempts");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_url_names_author_and_blob() {
        let image = PendingImage {
            post_id: 1,
            position: 0,
            author_did: "did:plc:abc".to_string(),
            blob_cid: "bafkreiabc".to_string(),
        };
        assert_eq!(
            image_url(&image),
            "https://cdn.bsky.app/img/feed_fullsize/plain/did:plc:abc/bafkreiabc@jpeg"
        );
    }

    /// The variant name of the error a status maps to.
    fn classify(status: StatusCode) -> &'static str {
        match FetchError::from_status(status) {
            FetchError::Gone(_) => "gone",
            FetchError::Refused(_) => "refused",
            FetchError::Unavailable(_) => "unavailable",
            FetchError::Unreachable(_) => "unreachable",
        }
    }

    #[test]
    fn test_status_classification() {
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::NOT_FOUND,
            StatusCode::GONE,
        ] {
            assert_eq!(classify(status), "gone", "{status}");
        }
        for status in [
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
        ] {
            assert_eq!(classify(status), "unavailable", "{status}");
        }
        assert_eq!(classify(StatusCode::FORBIDDEN), "refused");
    }
}
