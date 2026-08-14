//! The Jetstream v2 archive HTTP client: `planSnapshot`, `getSegment` and
//! `getBlock`.
//!
//! These three calls carry the API key as a bearer token; nothing else
//! does. Whole-segment downloads spool to disk and resume with HTTP Range
//! requests, because a segment can run to hundreds of megabytes and the
//! server meters transfer by byte. Single blocks are small and fetched
//! into memory.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncWriteExt;

/// Filters and bounds for one `planSnapshot` call. `after_seq` is
/// exclusive, `before_seq` inclusive; pin `before_seq` to the first page's
/// `sealedTipSeq` so the snapshot does not move while it is downloaded.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotRequest {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub collections: Vec<String>,
    pub after_seq: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_seq: Option<u64>,
}

/// One page of a snapshot plan.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotPlan {
    /// Highest sealed seq this page accounts for; the next page's
    /// `after_seq`. Planning is complete when it reaches `sealed_tip_seq`.
    pub planned_through_seq: u64,
    pub sealed_tip_seq: u64,
    pub segments: Vec<PlanSegment>,
    pub stats: PlanStats,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanSegment {
    pub name: String,
    pub index: u64,
    /// xxh3 metadata checksum, 16 hex characters; used as a cache key.
    pub checksum: String,
    pub min_seq: u64,
    pub max_seq: u64,
    /// `segment` (download the whole file) or `blocks` (download the
    /// listed ranges).
    pub mode: String,
    #[serde(default)]
    pub blocks: Vec<BlockRange>,
}

/// An inclusive range of block indices.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub struct BlockRange {
    pub first: u64,
    pub last: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanStats {
    pub segments_examined: u64,
    pub segments_matched: u64,
    pub blocks_matched: u64,
    pub entries: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// The key is missing, malformed or revoked. Retrying cannot help.
    #[error("archive request unauthorised: {0}")]
    Unauthorized(String),
    /// Byte quota exhausted; retry after the given delay.
    #[error("archive request rate limited")]
    RateLimited { retry_after: Option<Duration> },
    #[error("archive request failed with HTTP status {status}: {body}")]
    Http { status: u16, body: String },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// The archive operations replay needs, as a trait so the replay driver
/// can be tested without a network.
pub trait Fetcher {
    async fn plan_snapshot(&self, req: &SnapshotRequest) -> Result<SnapshotPlan, ArchiveError>;
    /// Fetch one raw zstd block frame.
    async fn fetch_block(&self, segment: &str, block_index: u64) -> Result<Vec<u8>, ArchiveError>;
    /// Download a whole segment file to the spool, resuming a previous
    /// partial download if one exists. Returns the path to the complete
    /// file; the caller deletes it once its events are ingested.
    async fn fetch_segment(&self, name: &str) -> Result<PathBuf, ArchiveError>;
}

/// Delay before retrying a failed archive call: 1 s doubling to a minute.
pub fn next_delay(attempt: u32) -> Duration {
    Duration::from_secs(1 << attempt.min(6)).min(Duration::from_secs(60))
}

/// The Range header for resuming a download that already holds `len` bytes.
fn range_from(len: u64) -> Option<String> {
    (len > 0).then(|| format!("bytes={len}-"))
}

pub struct HttpArchive {
    client: reqwest::Client,
    hostname: String,
    key: String,
    spool_dir: PathBuf,
}

impl HttpArchive {
    pub fn new(hostname: &str, key: &str, spool_dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(spool_dir)?;
        let client = reqwest::Client::builder()
            .user_agent(concat!("bsearch/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            // A read timeout rather than a whole-request deadline, so a
            // large but flowing segment download is never cut off.
            .read_timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            client,
            hostname: hostname.to_string(),
            key: key.to_string(),
            spool_dir: spool_dir.to_path_buf(),
        })
    }

    fn url(&self, method: &str) -> String {
        format!(
            "https://{}/xrpc/network.bsky.jetstream.{method}",
            self.hostname
        )
    }

    /// Map non-success statuses onto the typed errors the driver acts on.
    async fn check(response: reqwest::Response) -> Result<reqwest::Response, ArchiveError> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs);
        let body = response.text().await.unwrap_or_default();
        match status.as_u16() {
            401 => Err(ArchiveError::Unauthorized(body)),
            429 => Err(ArchiveError::RateLimited { retry_after }),
            s => Err(ArchiveError::Http { status: s, body }),
        }
    }
}

impl Fetcher for HttpArchive {
    async fn plan_snapshot(&self, req: &SnapshotRequest) -> Result<SnapshotPlan, ArchiveError> {
        let response = self
            .client
            .post(self.url("planSnapshot"))
            .bearer_auth(&self.key)
            .json(req)
            .send()
            .await
            .map_err(|e| ArchiveError::Other(e.into()))?;
        let response = Self::check(response).await?;
        response
            .json::<SnapshotPlan>()
            .await
            .map_err(|e| ArchiveError::Other(e.into()))
    }

    async fn fetch_block(&self, segment: &str, block_index: u64) -> Result<Vec<u8>, ArchiveError> {
        let response = self
            .client
            .get(self.url("getBlock"))
            .query(&[
                ("segment", segment),
                ("blockIndex", &block_index.to_string()),
            ])
            .bearer_auth(&self.key)
            .send()
            .await
            .map_err(|e| ArchiveError::Other(e.into()))?;
        let response = Self::check(response).await?;
        Ok(response
            .bytes()
            .await
            .map_err(|e| ArchiveError::Other(e.into()))?
            .to_vec())
    }

    async fn fetch_segment(&self, name: &str) -> Result<PathBuf, ArchiveError> {
        let done_path = self.spool_dir.join(name);
        if done_path.exists() {
            // Downloaded fully on an earlier attempt; the crash happened
            // between download and ingest.
            return Ok(done_path);
        }
        let part_path = self.spool_dir.join(format!("{name}.part"));
        let etag_path = self.spool_dir.join(format!("{name}.etag"));
        let existing = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);

        let mut request = self
            .client
            .get(self.url("getSegment"))
            .query(&[("name", name)])
            .bearer_auth(&self.key);
        if let Some(range) = range_from(existing) {
            request = request.header(reqwest::header::RANGE, range);
            // If-Range makes a changed file come back whole (200) instead
            // of as a mismatched suffix.
            if let Ok(etag) = std::fs::read_to_string(&etag_path) {
                request = request.header(reqwest::header::IF_RANGE, etag.trim());
            }
        }
        let response = request
            .send()
            .await
            .map_err(|e| ArchiveError::Other(e.into()))?;

        // A complete spool answered with 416: nothing left to fetch.
        if existing > 0 && response.status().as_u16() == 416 {
            std::fs::rename(&part_path, &done_path).map_err(anyhow::Error::from)?;
            let _ = std::fs::remove_file(&etag_path);
            return Ok(done_path);
        }
        let mut response = Self::check(response).await?;

        let resumed = response.status().as_u16() == 206;
        if let Some(etag) = response
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
        {
            std::fs::write(&etag_path, etag).map_err(anyhow::Error::from)?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(resumed)
            .write(true)
            .truncate(!resumed)
            .open(&part_path)
            .await
            .map_err(anyhow::Error::from)?;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| ArchiveError::Other(e.into()))?
        {
            file.write_all(&chunk).await.map_err(anyhow::Error::from)?;
        }
        file.flush().await.map_err(anyhow::Error::from)?;
        drop(file);
        std::fs::rename(&part_path, &done_path).map_err(anyhow::Error::from)?;
        let _ = std::fs::remove_file(&etag_path);
        Ok(done_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_deserialises_both_modes() {
        // Shapes from the planSnapshot lexicon: a whole-segment unit and a
        // sparse blocks unit.
        let json = r#"{
          "plannedThroughSeq": 900,
          "sealedTipSeq": 900,
          "segments": [
            {"name": "seg_0000000001.jss", "index": 1,
             "checksum": "0123456789abcdef", "minSeq": 1, "maxSeq": 500,
             "mode": "segment"},
            {"name": "seg_0000000002.jss", "index": 2,
             "checksum": "fedcba9876543210", "minSeq": 501, "maxSeq": 900,
             "mode": "blocks", "blocks": [{"first": 3, "last": 4}, {"first": 9, "last": 9}]}
          ],
          "stats": {"segmentsExamined": 40, "segmentsMatched": 2,
                    "blocksMatched": 3, "entries": 2}
        }"#;
        let plan: SnapshotPlan = serde_json::from_str(json).expect("deserialise");
        assert_eq!(plan.planned_through_seq, 900);
        assert_eq!(plan.sealed_tip_seq, 900);
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[0].mode, "segment");
        assert!(plan.segments[0].blocks.is_empty());
        assert_eq!(plan.segments[1].mode, "blocks");
        assert_eq!(
            plan.segments[1].blocks,
            vec![
                BlockRange { first: 3, last: 4 },
                BlockRange { first: 9, last: 9 }
            ]
        );
        assert_eq!(plan.stats.segments_examined, 40);
    }

    #[test]
    fn test_request_serialises_with_lexicon_field_names() {
        let req = SnapshotRequest {
            dids: vec!["did:plc:abc".to_string()],
            collections: vec!["app.bsky.feed.post".to_string()],
            after_seq: 0,
            before_seq: None,
        };
        let json = serde_json::to_value(&req).expect("serialise");
        assert_eq!(
            json,
            serde_json::json!({
                "dids": ["did:plc:abc"],
                "collections": ["app.bsky.feed.post"],
                "afterSeq": 0
            })
        );
    }

    #[test]
    fn test_request_pins_before_seq_on_later_pages() {
        let req = SnapshotRequest {
            dids: vec![],
            collections: vec![],
            after_seq: 900,
            before_seq: Some(1_000),
        };
        let json = serde_json::to_value(&req).expect("serialise");
        assert_eq!(
            json,
            serde_json::json!({"afterSeq": 900, "beforeSeq": 1000})
        );
    }

    #[test]
    fn test_next_delay_doubles_to_a_minute() {
        assert_eq!(next_delay(0), Duration::from_secs(1));
        assert_eq!(next_delay(1), Duration::from_secs(2));
        assert_eq!(next_delay(5), Duration::from_secs(32));
        assert_eq!(next_delay(6), Duration::from_secs(60));
        assert_eq!(next_delay(60), Duration::from_secs(60));
    }

    #[test]
    fn test_range_from_spool_length() {
        assert_eq!(range_from(0), None);
        assert_eq!(range_from(1024), Some("bytes=1024-".to_string()));
    }
}
