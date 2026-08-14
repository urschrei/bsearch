//! Decoder for Jetstream sealed segment files (`.jss`) and their block
//! frames.
//!
//! Layout per `docs/README.md` sections 3.1.2 and 3.2 of
//! <https://github.com/bluesky-social/jetstream>: a 256-byte fixed header,
//! 8-byte-length-prefixed zstd block frames, then a footer of indexes this
//! reader never needs (a sequential walk stops at `footer_offset`). Each
//! block frame decompresses to a columnar body: a `u32` event count, the
//! fixed-width columns, then the concatenated variable-length columns.
//! zstd frames are written with content checksums, so corruption surfaces
//! as a decompression error.
#![allow(dead_code)]

use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;

use super::events::record_cid;
use super::events::record_json;
use super::events::Commit;
use super::events::Event;
use super::events::Payload;

pub const RESERVED_HEADER_BYTES: u64 = 256;
const SEGMENT_MAGIC: &[u8; 4] = b"jss0";
const HEADER_VERSION: u16 = 1;
/// Hard ceiling on a block's event count, mirroring the reference
/// reader's guard against a lying header.
const MAX_BLOCK_EVENTS: u64 = 1 << 18;
/// Decompression bound per block; well above any real block, well below
/// a decompression bomb.
const MAX_BLOCK_UNCOMPRESSED: u64 = 256 << 20;

/// The parsed 256-byte fixed header of a sealed segment.
#[derive(Debug, Clone, Copy)]
pub struct SegmentHeader {
    pub version: u16,
    pub block_count: u32,
    pub event_count: u32,
    pub min_seq: u64,
    pub max_seq: u64,
    pub footer_offset: u64,
}

/// One row of a segment block, still in on-disk terms.
#[derive(Debug, Clone, PartialEq)]
pub struct RawEvent {
    pub seq: u64,
    pub witnessed_at: i64,
    pub indexed_at: i64,
    pub kind: u8,
    pub collection: String,
    pub did: String,
    pub rkey: String,
    pub rev: String,
    /// Raw DAG-CBOR record bytes; empty for kinds that carry none.
    pub payload: Vec<u8>,
}

pub const KIND_CREATE: u8 = 1;
pub const KIND_UPDATE: u8 = 2;
pub const KIND_DELETE: u8 = 3;
pub const KIND_IDENTITY: u8 = 4;
pub const KIND_ACCOUNT: u8 = 5;
pub const KIND_SYNC: u8 = 6;
/// A Sync 1.1 resync replacement row; commit-shaped, materialises the
/// record like a create.
pub const KIND_CREATE_RESYNC: u8 = 7;

impl RawEvent {
    /// The display timestamp handed to clients as `time_us`: the imported
    /// `indexed_at` when set, else the witnessed time.
    pub fn time_us(&self) -> i64 {
        if self.indexed_at != 0 {
            self.indexed_at
        } else {
            self.witnessed_at
        }
    }

    /// Convert to the shared event model. `decode_record` skips the CBOR
    /// work for rows the caller will discard anyway; commits without it
    /// carry no record or cid, like wire deletes.
    pub fn to_event(&self, decode_record: bool) -> Result<Event> {
        let payload = match self.kind {
            KIND_CREATE | KIND_UPDATE | KIND_DELETE | KIND_CREATE_RESYNC => {
                let operation = match self.kind {
                    KIND_UPDATE => "update",
                    KIND_DELETE => "delete",
                    _ => "create",
                };
                let materialises = self.kind != KIND_DELETE && !self.payload.is_empty();
                let (record, cid) = if decode_record && materialises {
                    (
                        Some(record_json(&self.payload)?),
                        Some(record_cid(&self.payload)),
                    )
                } else {
                    (None, None)
                };
                Payload::Commit(Commit {
                    operation: operation.to_string(),
                    collection: self.collection.clone(),
                    rkey: self.rkey.clone(),
                    rev: self.rev.clone(),
                    record,
                    cid,
                })
            }
            KIND_IDENTITY => Payload::Identity,
            KIND_ACCOUNT => Payload::Account,
            KIND_SYNC => Payload::Sync,
            k => bail!("unknown event kind {k}"),
        };
        Ok(Event {
            seq: self.seq,
            did: self.did.clone(),
            time_us: self.time_us(),
            payload,
        })
    }
}

/// Decompress and decode one raw block frame, exactly the bytes
/// `getBlock` returns or a length-prefixed frame inside a segment file.
pub fn decode_block_frame(frame: &[u8]) -> Result<Vec<RawEvent>> {
    let content_size = zstd::zstd_safe::get_frame_content_size(frame)
        .map_err(|e| anyhow::anyhow!("bad zstd frame: {e:?}"))?
        .context("zstd frame does not declare its content size")?;
    if content_size > MAX_BLOCK_UNCOMPRESSED {
        bail!("block frame declares {content_size} uncompressed bytes; refusing");
    }
    let body = zstd::bulk::decompress(frame, content_size as usize)
        .context("zstd decompression failed")?;
    decode_block_body(&body)
}

/// Decode the uncompressed columnar body of a block.
fn decode_block_body(buf: &[u8]) -> Result<Vec<RawEvent>> {
    const FIXED_PER_EVENT: u64 = 8 + 8 + 8 + 1 + 1 + 2 + 1 + 1 + 4;

    let mut cursor = ByteCursor::new(buf);
    let n64 = cursor.u32().context("event count")? as u64;
    if n64 > MAX_BLOCK_EVENTS {
        bail!("block claims {n64} events; refusing");
    }
    if (cursor.remaining() as u64) < n64 * FIXED_PER_EVENT {
        bail!(
            "truncated block: {} bytes left for {n64} events",
            cursor.remaining()
        );
    }
    let n = n64 as usize;
    if n == 0 {
        if cursor.remaining() != 0 {
            bail!("zero-event block with trailing bytes");
        }
        return Ok(Vec::new());
    }

    let seqs: Vec<u64> = cursor.u64_column(n)?;
    let witnessed: Vec<i64> = cursor
        .u64_column(n)?
        .into_iter()
        .map(|v| v as i64)
        .collect();
    let indexed: Vec<i64> = cursor
        .u64_column(n)?
        .into_iter()
        .map(|v| v as i64)
        .collect();
    let kinds = cursor.bytes(n)?.to_vec();
    if let Some(bad) = kinds
        .iter()
        .find(|k| **k < KIND_CREATE || **k > KIND_CREATE_RESYNC)
    {
        bail!("invalid event kind {bad}");
    }
    let collection_lens = cursor.bytes(n)?.to_vec();
    let did_lens: Vec<u16> = cursor.u16_column(n)?;
    let rkey_lens = cursor.bytes(n)?.to_vec();
    let rev_lens = cursor.bytes(n)?.to_vec();
    let payload_lens: Vec<u32> = cursor.u32_column(n)?;

    let collections = cursor.string_column(collection_lens.iter().map(|l| *l as usize))?;
    let dids = cursor.string_column(did_lens.iter().map(|l| *l as usize))?;
    let rkeys = cursor.string_column(rkey_lens.iter().map(|l| *l as usize))?;
    let revs = cursor.string_column(rev_lens.iter().map(|l| *l as usize))?;
    let mut payloads = Vec::with_capacity(n);
    for len in &payload_lens {
        payloads.push(cursor.bytes(*len as usize)?.to_vec());
    }
    if cursor.remaining() != 0 {
        bail!("block has {} trailing bytes", cursor.remaining());
    }

    let mut events = Vec::with_capacity(n);
    for i in 0..n {
        events.push(RawEvent {
            seq: seqs[i],
            witnessed_at: witnessed[i],
            indexed_at: indexed[i],
            kind: kinds[i],
            collection: collections[i].clone(),
            did: dids[i].clone(),
            rkey: rkeys[i].clone(),
            rev: revs[i].clone(),
            payload: std::mem::take(&mut payloads[i]),
        });
    }
    Ok(events)
}

/// Parse and validate the fixed header of a sealed segment file.
pub fn read_header(bytes: &[u8; 256]) -> Result<SegmentHeader> {
    if &bytes[0..4] != SEGMENT_MAGIC {
        bail!("not a segment file: bad magic");
    }
    let le_u64 = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
    let le_u32 = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    let checksum = le_u64(4);
    if checksum == 0 {
        bail!("segment is active (unsealed); refusing to read");
    }
    let version = u16::from_le_bytes(bytes[12..14].try_into().unwrap());
    if version != HEADER_VERSION {
        bail!("unsupported segment version {version}");
    }
    Ok(SegmentHeader {
        version,
        block_count: le_u32(14),
        event_count: le_u32(18),
        min_seq: le_u64(26),
        max_seq: le_u64(34),
        footer_offset: le_u64(58),
    })
}

/// Walk a sealed segment file sequentially, calling `visit` with each
/// block's events in order. The footer is never read.
pub fn for_each_block(
    path: &Path,
    mut visit: impl FnMut(Vec<RawEvent>) -> Result<()>,
) -> Result<()> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut header_bytes = [0u8; 256];
    file.read_exact(&mut header_bytes).context("read header")?;
    let header = read_header(&header_bytes)?;

    let mut offset = RESERVED_HEADER_BYTES;
    file.seek(SeekFrom::Start(offset))?;
    let mut blocks_seen: u32 = 0;
    while offset < header.footer_offset {
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .context("read block length")?;
        let block_len = u64::from_le_bytes(len_bytes);
        if offset + 8 + block_len > header.footer_offset {
            bail!("block at offset {offset} runs past the footer");
        }
        let mut frame = vec![0u8; block_len as usize];
        file.read_exact(&mut frame).context("read block frame")?;
        visit(decode_block_frame(&frame)?)?;
        offset += 8 + block_len;
        blocks_seen += 1;
    }
    if blocks_seen != header.block_count {
        bail!(
            "segment header promises {} blocks but the file holds {blocks_seen}",
            header.block_count
        );
    }
    Ok(())
}

/// A bounds-checked little-endian reader over a byte slice.
struct ByteCursor<'a> {
    buf: &'a [u8],
    off: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, off: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.off
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.off + n > self.buf.len() {
            bail!(
                "truncated block: wanted {n} bytes, {} left",
                self.remaining()
            );
        }
        let s = &self.buf[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u16_column(&mut self, n: usize) -> Result<Vec<u16>> {
        let chunk = self.bytes(n * 2)?;
        Ok(chunk
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn u32_column(&mut self, n: usize) -> Result<Vec<u32>> {
        let chunk = self.bytes(n * 4)?;
        Ok(chunk
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn u64_column(&mut self, n: usize) -> Result<Vec<u64>> {
        let chunk = self.bytes(n * 8)?;
        Ok(chunk
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn string_column(&mut self, lens: impl Iterator<Item = usize>) -> Result<Vec<String>> {
        lens.map(|len| {
            let bytes = self.bytes(len)?;
            String::from_utf8(bytes.to_vec()).context("non-UTF-8 column value")
        })
        .collect()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Encode the uncompressed columnar body the way the segment writer
    /// does; the test-side mirror of `decode_block_body`.
    pub fn encode_block_body(events: &[RawEvent]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(events.len() as u32).to_le_bytes());
        for e in events {
            buf.extend_from_slice(&e.seq.to_le_bytes());
        }
        for e in events {
            buf.extend_from_slice(&(e.witnessed_at as u64).to_le_bytes());
        }
        for e in events {
            buf.extend_from_slice(&(e.indexed_at as u64).to_le_bytes());
        }
        for e in events {
            buf.push(e.kind);
        }
        for e in events {
            buf.push(e.collection.len() as u8);
        }
        for e in events {
            buf.extend_from_slice(&(e.did.len() as u16).to_le_bytes());
        }
        for e in events {
            buf.push(e.rkey.len() as u8);
        }
        for e in events {
            buf.push(e.rev.len() as u8);
        }
        for e in events {
            buf.extend_from_slice(&(e.payload.len() as u32).to_le_bytes());
        }
        for e in events {
            buf.extend_from_slice(e.collection.as_bytes());
        }
        for e in events {
            buf.extend_from_slice(e.did.as_bytes());
        }
        for e in events {
            buf.extend_from_slice(e.rkey.as_bytes());
        }
        for e in events {
            buf.extend_from_slice(e.rev.as_bytes());
        }
        for e in events {
            buf.extend_from_slice(&e.payload);
        }
        buf
    }

    pub fn compress_block(events: &[RawEvent]) -> Vec<u8> {
        zstd::bulk::compress(&encode_block_body(events), 3).expect("compress")
    }

    pub fn sample_events() -> Vec<RawEvent> {
        let record = serde_json::json!({
            "$type": "app.bsky.feed.post",
            "text": "a segment post about ferrets",
            "createdAt": "2026-07-25T21:01:36.005Z",
        });
        let payload = serde_ipld_dagcbor::to_vec(&record).expect("cbor");
        vec![
            RawEvent {
                seq: 100,
                witnessed_at: 1_785_009_595_019_071,
                indexed_at: 0,
                kind: KIND_CREATE,
                collection: "app.bsky.feed.post".to_string(),
                did: "did:plc:abc123".to_string(),
                rkey: "3lxyz123".to_string(),
                rev: "3mrir7qvgmk2o".to_string(),
                payload,
            },
            RawEvent {
                seq: 101,
                witnessed_at: 1_785_009_595_019_072,
                indexed_at: 0,
                kind: KIND_IDENTITY,
                collection: String::new(),
                did: "did:plc:abc123".to_string(),
                rkey: String::new(),
                rev: String::new(),
                payload: Vec::new(),
            },
            RawEvent {
                seq: 102,
                witnessed_at: 1_785_009_595_019_073,
                indexed_at: 0,
                kind: KIND_DELETE,
                collection: "app.bsky.feed.post".to_string(),
                did: "did:plc:abc123".to_string(),
                rkey: "gone".to_string(),
                rev: "3mrir7qvgmk2p".to_string(),
                payload: Vec::new(),
            },
        ]
    }

    #[test]
    fn test_block_frame_round_trips() {
        let events = sample_events();
        let decoded = decode_block_frame(&compress_block(&events)).expect("decode");
        assert_eq!(decoded, events);
    }

    #[test]
    fn test_zero_event_block_is_empty_not_an_error() {
        // Compaction leaves zero-event blocks behind.
        assert_eq!(
            decode_block_frame(&compress_block(&[])).expect("decode"),
            Vec::new()
        );
    }

    #[test]
    fn test_truncated_body_is_an_error() {
        let mut body = encode_block_body(&sample_events());
        body.truncate(body.len() - 10);
        let frame = zstd::bulk::compress(&body, 3).expect("compress");
        assert!(decode_block_frame(&frame).is_err());
    }

    #[test]
    fn test_invalid_kind_is_an_error() {
        let mut events = sample_events();
        events[0].kind = 42;
        assert!(decode_block_frame(&compress_block(&events)).is_err());
    }

    #[test]
    fn test_corrupt_frame_is_an_error() {
        let mut frame = compress_block(&sample_events());
        let mid = frame.len() / 2;
        frame[mid] ^= 0xff;
        assert!(decode_block_frame(&frame).is_err());
    }

    #[test]
    fn test_raw_create_converts_with_record_and_cid() {
        let raw = &sample_events()[0];
        let event = raw.to_event(true).expect("convert");
        assert_eq!(event.seq, 100);
        assert_eq!(event.time_us, 1_785_009_595_019_071);
        let Payload::Commit(commit) = event.payload else {
            panic!("expected commit");
        };
        assert_eq!(commit.operation, "create");
        assert_eq!(
            commit
                .record
                .as_ref()
                .and_then(|r| r.get("text"))
                .and_then(|t| t.as_str()),
            Some("a segment post about ferrets")
        );
        assert!(commit.cid.expect("cid").starts_with("bafyrei"));
    }

    #[test]
    fn test_raw_delete_converts_without_record() {
        let raw = &sample_events()[2];
        let event = raw.to_event(true).expect("convert");
        let Payload::Commit(commit) = event.payload else {
            panic!("expected commit");
        };
        assert_eq!(commit.operation, "delete");
        assert_eq!(commit.record, None);
        assert_eq!(commit.cid, None);
    }

    #[test]
    fn test_indexed_at_overrides_witnessed_at() {
        let mut raw = sample_events()[0].clone();
        assert_eq!(raw.time_us(), raw.witnessed_at);
        raw.indexed_at = 42;
        assert_eq!(raw.time_us(), 42);
    }

    /// Write a sealed segment fixture from per-block event lists; the
    /// entry point sibling test modules use.
    pub fn write_segment_fixture(path: &Path, blocks: &[Vec<RawEvent>]) {
        let frames: Vec<Vec<u8>> = blocks.iter().map(|b| compress_block(b)).collect();
        write_test_segment(path, &frames);
    }

    /// Build a minimal sealed segment file: header, one block, footer
    /// stub. Offsets and the non-zero checksum are what `read_header`
    /// validates; the footer content is never read.
    fn write_test_segment(path: &Path, frames: &[Vec<u8>]) {
        let mut body = Vec::new();
        for frame in frames {
            body.extend_from_slice(&(frame.len() as u64).to_le_bytes());
            body.extend_from_slice(frame);
        }
        let footer_offset = RESERVED_HEADER_BYTES + body.len() as u64;
        let mut header = vec![0u8; 256];
        header[0..4].copy_from_slice(b"jss0");
        header[4..12].copy_from_slice(&1u64.to_le_bytes()); // non-zero checksum
        header[12..14].copy_from_slice(&1u16.to_le_bytes()); // version
        header[14..18].copy_from_slice(&(frames.len() as u32).to_le_bytes());
        header[58..66].copy_from_slice(&footer_offset.to_le_bytes());
        let mut bytes = header;
        bytes.extend_from_slice(&body);
        bytes.extend_from_slice(b"footer bytes the reader must ignore");
        std::fs::write(path, bytes).expect("write segment");
    }

    #[test]
    fn test_for_each_block_walks_a_sealed_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seg_0000000001.jss");
        let events = sample_events();
        write_test_segment(
            &path,
            &[compress_block(&events), compress_block(&events[..1])],
        );

        let mut seen = Vec::new();
        for_each_block(&path, |block| {
            seen.push(block.len());
            Ok(())
        })
        .expect("walk");
        assert_eq!(seen, vec![3, 1]);
    }

    #[test]
    fn test_for_each_block_rejects_an_active_segment() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("active.jss");
        // Zero checksum marks an active file whose header is unfinalised.
        let mut bytes = vec![0u8; 256];
        bytes[0..4].copy_from_slice(b"jss0");
        std::fs::write(&path, bytes).expect("write");
        assert!(for_each_block(&path, |_| Ok(())).is_err());
    }

    #[test]
    fn test_for_each_block_rejects_block_count_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seg_bad_count.jss");
        let events = sample_events();
        write_test_segment(&path, &[compress_block(&events)]);
        // Corrupt the promised block count.
        let mut bytes = std::fs::read(&path).expect("read");
        bytes[14..18].copy_from_slice(&9u32.to_le_bytes());
        std::fs::write(&path, bytes).expect("rewrite");
        assert!(for_each_block(&path, |_| Ok(())).is_err());
    }
}
