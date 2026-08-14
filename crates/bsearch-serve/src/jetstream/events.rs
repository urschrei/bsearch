//! The Jetstream v2 event model, shared by the live tail and archive replay.
//!
//! Wire and format references: the `subscribeEvents` lexicon and
//! `docs/README.md` sections 3.2 and 5.2 in
//! <https://github.com/bluesky-social/jetstream>.
#![allow(dead_code)]

use anyhow::Context;
use anyhow::Result;
use cid::Cid;
use ipld_core::ipld::Ipld;
use multihash::Multihash;
use serde::Deserialize;
use sha2::Digest;
use sha2::Sha256;

/// Cursor values at or above this are unix-microsecond timestamps; below it
/// they are v2 sequence numbers. The split is the server's
/// (`CursorSeqMaxThreshold`): at current network throughput seqs cannot
/// reach 1e15 for centuries, and microsecond timestamps passed 1e15 in 2001.
pub const CURSOR_SEQ_MAX_THRESHOLD: i64 = 1_000_000_000_000_000;

/// What the single stored cursor value means, decided by magnitude.
///
/// The same `meta` row holds a v1 `time_us` before the migration and a v2
/// seq after it. A legacy value also appears if the Python tooling ever
/// writes a timestamp cursor again; it is simply re-migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredCursor {
    Seq(u64),
    LegacyTimeUs(i64),
}

pub fn classify_cursor(value: i64) -> StoredCursor {
    if value >= CURSOR_SEQ_MAX_THRESHOLD {
        StoredCursor::LegacyTimeUs(value)
    } else {
        StoredCursor::Seq(value.max(0) as u64)
    }
}

/// One Jetstream event, whichever transport delivered it.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// Jetstream's monotonic sequence number; the stream cursor.
    pub seq: u64,
    pub did: String,
    /// Display timestamp in unix microseconds. Informational only: the
    /// cursor is `seq`.
    pub time_us: i64,
    pub payload: Payload,
}

/// The kind-specific payload. Identity, account and sync events are
/// delivered even under a collection filter; the daemon ignores their
/// bodies but still advances the cursor past them.
#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    Commit(Commit),
    Identity,
    Account,
    Sync,
}

/// A record mutation. `operation` is the wire string (`create`, `update`,
/// `delete`); `record` and `cid` are absent on deletes.
#[derive(Debug, Clone, PartialEq)]
pub struct Commit {
    pub operation: String,
    pub collection: String,
    pub rkey: String,
    pub rev: String,
    pub record: Option<serde_json::Value>,
    pub cid: Option<String>,
}

/// One decoded live-tail frame.
///
/// `Unknown` covers envelope or payload `$type`s this client does not know;
/// the protocol requires skipping them for forward compatibility.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Event(Event),
    /// A seq-less advisory, e.g. `OutdatedCursor` after a clamped
    /// timestamp-cursor resume.
    Info {
        name: String,
        message: Option<String>,
    },
    /// Terminal: the server closes the connection after sending it.
    Error {
        error: String,
        message: Option<String>,
    },
    Unknown,
}

const PAYLOAD_TYPE_PREFIX: &str = "network.bsky.jetstream.subscribeEvents#";

#[derive(Deserialize)]
struct WireEnvelope {
    #[serde(rename = "$type")]
    kind: String,
    payload: Option<serde_json::Value>,
    error: Option<String>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct WireEventPayload {
    seq: u64,
    did: String,
    time: String,
    #[serde(default)]
    rev: String,
    #[serde(default)]
    operation: String,
    #[serde(default)]
    collection: String,
    #[serde(default)]
    rkey: String,
    record: Option<serde_json::Value>,
    cid: Option<String>,
}

#[derive(Deserialize)]
struct WireInfoPayload {
    name: String,
    message: Option<String>,
}

/// Decode one `xrpc.v1.json` text frame.
pub fn decode_frame(text: &str) -> Result<Frame> {
    let envelope: WireEnvelope = serde_json::from_str(text).context("undecodable frame")?;
    match envelope.kind.as_str() {
        "message" => {
            let payload = envelope.payload.context("message frame without payload")?;
            decode_message_payload(payload)
        }
        "error" => Ok(Frame::Error {
            error: envelope.error.unwrap_or_default(),
            message: envelope.message,
        }),
        _ => Ok(Frame::Unknown),
    }
}

fn decode_message_payload(payload: serde_json::Value) -> Result<Frame> {
    let type_name = payload
        .get("$type")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let Some(fragment) = type_name.strip_prefix(PAYLOAD_TYPE_PREFIX) else {
        return Ok(Frame::Unknown);
    };
    if fragment == "info" {
        let info: WireInfoPayload = serde_json::from_value(payload).context("bad info payload")?;
        return Ok(Frame::Info {
            name: info.name,
            message: info.message,
        });
    }
    let make_payload = match fragment {
        "commit" => None,
        "identity" => Some(Payload::Identity),
        "account" => Some(Payload::Account),
        "sync" => Some(Payload::Sync),
        _ => return Ok(Frame::Unknown),
    };
    let wire: WireEventPayload =
        serde_json::from_value(payload).with_context(|| format!("bad {fragment} payload"))?;
    let payload = match make_payload {
        Some(p) => p,
        None => Payload::Commit(Commit {
            operation: wire.operation,
            collection: wire.collection,
            rkey: wire.rkey,
            rev: wire.rev,
            record: wire.record,
            cid: wire.cid,
        }),
    };
    Ok(Frame::Event(Event {
        seq: wire.seq,
        did: wire.did,
        time_us: parse_time_us(&wire.time),
        payload,
    }))
}

/// Parse the wire `time` (RFC 3339, microsecond precision) to unix
/// microseconds. The value is informational, so a malformed one becomes 0
/// rather than an error.
fn parse_time_us(time: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(time)
        .map(|dt| dt.timestamp_micros())
        .unwrap_or(0)
}

/// The CID of a record, computed from its DAG-CBOR bytes: CIDv1, dag-cbor
/// codec, sha2-256, rendered in the canonical base32-lower multibase.
///
/// Segment rows do not carry the CID the live wire shows, but it is fully
/// determined by the record bytes, so archive replay reconstructs it.
pub fn record_cid(dag_cbor: &[u8]) -> String {
    const DAG_CBOR_CODEC: u64 = 0x71;
    const SHA2_256: u64 = 0x12;
    let digest = Sha256::digest(dag_cbor);
    let hash = Multihash::<64>::wrap(SHA2_256, &digest).expect("32-byte digest fits");
    Cid::new_v1(DAG_CBOR_CODEC, hash).to_string()
}

/// Decode a record's DAG-CBOR bytes into the same JSON value the live wire
/// carries, following the atproto JSON data model: bytes become
/// `{"$bytes": base64}` and links become `{"$link": cid}`.
pub fn record_json(dag_cbor: &[u8]) -> Result<serde_json::Value> {
    let ipld: Ipld = serde_ipld_dagcbor::from_slice(dag_cbor).context("undecodable record CBOR")?;
    Ok(ipld_to_json(ipld))
}

fn ipld_to_json(ipld: Ipld) -> serde_json::Value {
    use serde_json::json;
    use serde_json::Value;
    match ipld {
        Ipld::Null => Value::Null,
        Ipld::Bool(b) => Value::Bool(b),
        Ipld::Integer(i) => Value::from(i as i64),
        Ipld::Float(f) => json!(f),
        Ipld::String(s) => Value::String(s),
        Ipld::Bytes(b) => json!({ "$bytes": data_encoding::BASE64_NOPAD.encode(&b) }),
        Ipld::List(items) => Value::Array(items.into_iter().map(ipld_to_json).collect()),
        Ipld::Map(map) => {
            Value::Object(map.into_iter().map(|(k, v)| (k, ipld_to_json(v))).collect())
        }
        Ipld::Link(cid) => json!({ "$link": cid.to_string() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_cursor_boundaries() {
        assert_eq!(classify_cursor(0), StoredCursor::Seq(0));
        assert_eq!(classify_cursor(123_456), StoredCursor::Seq(123_456));
        assert_eq!(
            classify_cursor(CURSOR_SEQ_MAX_THRESHOLD - 1),
            StoredCursor::Seq((CURSOR_SEQ_MAX_THRESHOLD - 1) as u64)
        );
        assert_eq!(
            classify_cursor(CURSOR_SEQ_MAX_THRESHOLD),
            StoredCursor::LegacyTimeUs(CURSOR_SEQ_MAX_THRESHOLD)
        );
        // A cursor stored by the v1 daemon in 2026.
        assert_eq!(
            classify_cursor(1_785_009_595_019_071),
            StoredCursor::LegacyTimeUs(1_785_009_595_019_071)
        );
    }

    #[test]
    fn test_decode_commit_frame() {
        // The example frame from docs/README.md section 5.2.
        let text = r#"{
          "$type": "message",
          "payload": {
            "$type": "network.bsky.jetstream.subscribeEvents#commit",
            "seq": 12345,
            "did": "did:plc:eygmaihciaxprqvxpfvl6flk",
            "time": "2024-09-09T19:46:02.329308Z",
            "rev": "3l3qo2vutsw2b",
            "operation": "create",
            "collection": "app.bsky.feed.like",
            "rkey": "3l3qo2vuowo2b",
            "cid": "bafyreidwaivazkwu67xztlmuobx35hs2lnfh3kolmgfmucldvhd3sgzcqi",
            "record": { "$type": "app.bsky.feed.like" }
          }
        }"#;
        let Frame::Event(event) = decode_frame(text).expect("decode") else {
            panic!("expected an event frame");
        };
        assert_eq!(event.seq, 12345);
        assert_eq!(event.did, "did:plc:eygmaihciaxprqvxpfvl6flk");
        assert_eq!(event.time_us, 1_725_911_162_329_308);
        let Payload::Commit(commit) = event.payload else {
            panic!("expected a commit payload");
        };
        assert_eq!(commit.operation, "create");
        assert_eq!(commit.collection, "app.bsky.feed.like");
        assert_eq!(commit.rkey, "3l3qo2vuowo2b");
        assert_eq!(commit.rev, "3l3qo2vutsw2b");
        assert!(commit.record.is_some());
        assert!(commit.cid.is_some());
    }

    #[test]
    fn test_decode_delete_commit_without_record_or_cid() {
        let text = r#"{"$type":"message","payload":{
            "$type":"network.bsky.jetstream.subscribeEvents#commit",
            "seq":7,"did":"did:plc:abc","time":"2026-01-01T00:00:00.000000Z",
            "rev":"3abc","operation":"delete",
            "collection":"app.bsky.feed.post","rkey":"gone"}}"#;
        let Frame::Event(event) = decode_frame(text).expect("decode") else {
            panic!("expected an event frame");
        };
        let Payload::Commit(commit) = event.payload else {
            panic!("expected a commit payload");
        };
        assert_eq!(commit.operation, "delete");
        assert_eq!(commit.record, None);
        assert_eq!(commit.cid, None);
    }

    #[test]
    fn test_decode_identity_frame_ignores_wrapped_upstream_event() {
        // The wrapped upstream event carries its own seq/time; only the
        // envelope fields matter here.
        let text = r#"{"$type":"message","payload":{
            "$type":"network.bsky.jetstream.subscribeEvents#identity",
            "seq":99,"did":"did:plc:abc","time":"2026-01-01T00:00:00.000000Z",
            "identity":{"seq":1,"did":"did:plc:abc","handle":"a.example",
                        "time":"2020-01-01T00:00:00.000Z"}}}"#;
        let Frame::Event(event) = decode_frame(text).expect("decode") else {
            panic!("expected an event frame");
        };
        assert_eq!(event.seq, 99);
        assert_eq!(event.payload, Payload::Identity);
    }

    #[test]
    fn test_decode_info_frame() {
        let text = r#"{"$type":"message","payload":{
            "$type":"network.bsky.jetstream.subscribeEvents#info",
            "name":"OutdatedCursor","message":"resumed from seq 42"}}"#;
        assert_eq!(
            decode_frame(text).expect("decode"),
            Frame::Info {
                name: "OutdatedCursor".to_string(),
                message: Some("resumed from seq 42".to_string()),
            }
        );
    }

    #[test]
    fn test_decode_error_frame() {
        let text = r#"{"$type":"error","error":"ConsumerTooSlow","message":"too slow"}"#;
        assert_eq!(
            decode_frame(text).expect("decode"),
            Frame::Error {
                error: "ConsumerTooSlow".to_string(),
                message: Some("too slow".to_string()),
            }
        );
    }

    #[test]
    fn test_unknown_types_are_skipped_not_errors() {
        let unknown_payload = r#"{"$type":"message","payload":{
            "$type":"network.bsky.jetstream.subscribeEvents#brandnew","seq":1}}"#;
        assert_eq!(
            decode_frame(unknown_payload).expect("decode"),
            Frame::Unknown
        );
        let unknown_envelope = r#"{"$type":"heartbeat"}"#;
        assert_eq!(
            decode_frame(unknown_envelope).expect("decode"),
            Frame::Unknown
        );
    }

    #[test]
    fn test_record_json_round_trips_a_post_record() {
        // Encode a post-shaped record to DAG-CBOR and decode it back the way
        // archive replay will.
        let record = serde_json::json!({
            "$type": "app.bsky.feed.post",
            "text": "a post about ferrets",
            "createdAt": "2026-07-25T21:01:36.005Z",
            "langs": ["en"],
        });
        let ipld: Ipld = ipld_core::serde::from_ipld::<Ipld>(
            serde_ipld_dagcbor::from_slice(&serde_ipld_dagcbor::to_vec(&record).expect("encode"))
                .expect("decode"),
        )
        .expect("ipld");
        let bytes = serde_ipld_dagcbor::to_vec(&ipld).expect("re-encode");
        assert_eq!(record_json(&bytes).expect("record_json"), record);
    }

    #[test]
    fn test_record_cid_shape_and_determinism() {
        let bytes = serde_ipld_dagcbor::to_vec(&serde_json::json!({"text": "hi"})).expect("cbor");
        let cid = record_cid(&bytes);
        // CIDv1 + dag-cbor + sha2-256 in base32-lower always starts this way.
        assert!(cid.starts_with("bafyrei"), "unexpected cid {cid}");
        assert_eq!(cid, record_cid(&bytes));
        let other = serde_ipld_dagcbor::to_vec(&serde_json::json!({"text": "bye"})).expect("cbor");
        assert_ne!(cid, record_cid(&other));
    }
}
