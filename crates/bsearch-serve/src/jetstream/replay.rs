//! The archive replay driver: page a snapshot plan, download and decode
//! its units, and feed matching events to the sink in seq order.
//!
//! The stored cursor is the only checkpoint. `planSnapshot` is stateless,
//! so a crash mid-replay costs nothing: the next run plans a smaller
//! window from wherever the cursor reached. All ingestion downstream is
//! idempotent, and the driver additionally skips rows at or below its
//! floor, so overlap at every boundary is absorbed.

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use tokio_util::sync::CancellationToken;

use super::archive::next_delay;
use super::archive::ArchiveError;
use super::archive::Fetcher;
use super::archive::PlanSegment;
use super::archive::SnapshotRequest;
use super::events::Event;
use super::jss;
use super::jss::RawEvent;

/// Where replayed events go. Implemented by the ingest handler; the sink
/// owns cursor persistence, exactly as it does for live events.
pub trait EventSink {
    async fn apply(&self, event: &Event) -> Result<()>;
}

/// What to replay: one DID's commits in the given collections, plus the
/// DID-level marker events the collection filter never suppresses.
#[derive(Debug, Clone)]
pub struct ReplayFilters {
    pub did: String,
    pub collections: Vec<String>,
}

impl ReplayFilters {
    /// The exact row selector. The plan over-approximates (whole blocks
    /// or files), so every row is checked here before it is surfaced.
    fn matches(&self, raw: &RawEvent) -> bool {
        if raw.did != self.did {
            return false;
        }
        match raw.kind {
            jss::KIND_CREATE | jss::KIND_UPDATE | jss::KIND_DELETE | jss::KIND_CREATE_RESYNC => {
                self.collections.contains(&raw.collection)
            }
            // Identity, account and sync markers are gated by DID only.
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayOutcome {
    /// Seq of the last event applied, if any survived the filter.
    pub last_seq: Option<u64>,
    /// The pinned sealed-archive tip. Connect the live tail at
    /// `max(sealed_tip_seq, last_seq)`.
    pub sealed_tip_seq: u64,
    pub events_applied: u64,
}

/// Sweep the sealed archive from `after_seq` (exclusive) and feed every
/// matching event to `sink` in seq order.
///
/// Returns early with the progress so far when `token` is cancelled. A
/// disappearing segment (compaction renamed the files mid-sweep) causes a
/// fresh plan from the current floor rather than an error.
pub async fn replay<F: Fetcher>(
    fetcher: &F,
    filters: &ReplayFilters,
    after_seq: u64,
    sink: &impl EventSink,
    token: &CancellationToken,
) -> Result<ReplayOutcome> {
    let mut floor = after_seq;
    let mut last_seq = None;
    let mut events_applied = 0u64;
    let mut pinned_tip: Option<u64> = None;

    'sweep: loop {
        if token.is_cancelled() {
            break;
        }
        let request = SnapshotRequest {
            dids: vec![filters.did.clone()],
            collections: filters.collections.clone(),
            after_seq: floor,
            before_seq: pinned_tip,
        };
        let plan = match retry(token, || fetcher.plan_snapshot(&request)).await {
            Ok(plan) => plan,
            Err(RetryOutcome::Cancelled) => break,
            Err(RetryOutcome::Fatal(e)) => return Err(e).context("planSnapshot"),
        };
        // Pin the tip on the first page so the snapshot does not move
        // while it is downloaded; segments sealed mid-sweep are covered
        // by the live tail's replay after cutover.
        let tip = *pinned_tip.get_or_insert(plan.sealed_tip_seq);
        tracing::debug!(
            units = plan.segments.len(),
            examined = plan.stats.segments_examined,
            matched = plan.stats.segments_matched,
            blocks = plan.stats.blocks_matched,
            entries = plan.stats.entries,
            planned_through = plan.planned_through_seq,
            "Snapshot plan page"
        );

        for unit in &plan.segments {
            tracing::debug!(
                name = %unit.name,
                index = unit.index,
                mode = %unit.mode,
                min_seq = unit.min_seq,
                max_seq = unit.max_seq,
                checksum = %unit.checksum,
                "Applying plan unit"
            );
            match apply_unit(
                fetcher,
                filters,
                unit,
                &mut floor,
                &mut last_seq,
                &mut events_applied,
                sink,
                token,
            )
            .await?
            {
                UnitOutcome::Done => {}
                UnitOutcome::Cancelled => break 'sweep,
                UnitOutcome::Replan => continue 'sweep,
            }
        }

        if plan.planned_through_seq >= tip {
            break;
        }
        if plan.planned_through_seq <= floor.min(request.after_seq) && plan.segments.is_empty() {
            bail!(
                "snapshot plan does not advance: plannedThroughSeq {} at floor {}",
                plan.planned_through_seq,
                request.after_seq
            );
        }
        floor = floor.max(plan.planned_through_seq);
    }

    Ok(ReplayOutcome {
        last_seq,
        sealed_tip_seq: pinned_tip.unwrap_or(after_seq),
        events_applied,
    })
}

enum UnitOutcome {
    Done,
    Cancelled,
    Replan,
}

#[allow(clippy::too_many_arguments)]
async fn apply_unit<F: Fetcher>(
    fetcher: &F,
    filters: &ReplayFilters,
    unit: &PlanSegment,
    floor: &mut u64,
    last_seq: &mut Option<u64>,
    events_applied: &mut u64,
    sink: &impl EventSink,
    token: &CancellationToken,
) -> Result<UnitOutcome> {
    match unit.mode.as_str() {
        "blocks" => {
            for range in &unit.blocks {
                for index in range.first..=range.last {
                    if token.is_cancelled() {
                        return Ok(UnitOutcome::Cancelled);
                    }
                    let frame = match retry(token, || fetcher.fetch_block(&unit.name, index)).await
                    {
                        Ok(frame) => frame,
                        Err(RetryOutcome::Cancelled) => return Ok(UnitOutcome::Cancelled),
                        Err(RetryOutcome::Fatal(e)) => {
                            if segment_vanished(&e) {
                                tracing::info!(segment = %unit.name, "segment gone; re-planning");
                                return Ok(UnitOutcome::Replan);
                            }
                            return Err(e)
                                .with_context(|| format!("getBlock {} {index}", unit.name));
                        }
                    };
                    let rows = jss::decode_block_frame(&frame)
                        .with_context(|| format!("decode block {index} of {}", unit.name))?;
                    apply_rows(filters, rows, floor, last_seq, events_applied, sink).await?;
                }
            }
            Ok(UnitOutcome::Done)
        }
        "segment" => {
            let path = match retry(token, || fetcher.fetch_segment(&unit.name)).await {
                Ok(path) => path,
                Err(RetryOutcome::Cancelled) => return Ok(UnitOutcome::Cancelled),
                Err(RetryOutcome::Fatal(e)) => {
                    if segment_vanished(&e) {
                        tracing::info!(segment = %unit.name, "segment gone; re-planning");
                        return Ok(UnitOutcome::Replan);
                    }
                    return Err(e).with_context(|| format!("getSegment {}", unit.name));
                }
            };
            // The walk is synchronous; collect per block and apply after,
            // so the async sink is not called from the closure.
            let mut blocks: Vec<Vec<RawEvent>> = Vec::new();
            jss::for_each_block(&path, |rows| {
                blocks.push(rows);
                Ok(())
            })
            .with_context(|| format!("decode segment {}", unit.name))?;
            for rows in blocks {
                apply_rows(filters, rows, floor, last_seq, events_applied, sink).await?;
            }
            // Ingested fully; the spool copy has served its purpose.
            std::fs::remove_file(&path).ok();
            Ok(UnitOutcome::Done)
        }
        other => bail!("unknown plan unit mode {other:?} for segment {}", unit.name),
    }
}

async fn apply_rows(
    filters: &ReplayFilters,
    rows: Vec<RawEvent>,
    floor: &mut u64,
    last_seq: &mut Option<u64>,
    events_applied: &mut u64,
    sink: &impl EventSink,
) -> Result<()> {
    for raw in rows {
        // The plan prunes by seq overlap one-sidedly: a unit straddling
        // the floor arrives whole, so rows at or below it reappear here
        // and are dropped, along with anything already applied.
        if raw.seq <= *floor {
            continue;
        }
        if !filters.matches(&raw) {
            continue;
        }
        let wants_record = matches!(
            raw.kind,
            jss::KIND_CREATE | jss::KIND_UPDATE | jss::KIND_CREATE_RESYNC
        );
        let event = raw.to_event(wants_record)?;
        sink.apply(&event).await?;
        *floor = raw.seq;
        *last_seq = Some(raw.seq);
        *events_applied += 1;
    }
    Ok(())
}

/// The segment named by the plan no longer exists: compaction rewrote the
/// archive under the sweep. The plan, not the request, is stale.
fn segment_vanished(error: &ArchiveError) -> bool {
    matches!(error, ArchiveError::Http { status: 404, .. })
}

enum RetryOutcome {
    Cancelled,
    Fatal(ArchiveError),
}

/// Run an archive call until it succeeds, backing off on transient
/// failures and honouring `Retry-After` on rate limits. Unauthorised and
/// unexpected-HTTP errors are surfaced to the caller; everything else
/// retries until cancelled.
async fn retry<T, Fut>(
    token: &CancellationToken,
    mut call: impl FnMut() -> Fut,
) -> Result<T, RetryOutcome>
where
    Fut: std::future::Future<Output = Result<T, ArchiveError>>,
{
    let mut attempt = 0u32;
    loop {
        if token.is_cancelled() {
            return Err(RetryOutcome::Cancelled);
        }
        let delay = match call().await {
            Ok(value) => return Ok(value),
            Err(ArchiveError::RateLimited { retry_after }) => {
                let delay = retry_after.unwrap_or_else(|| next_delay(attempt));
                tracing::info!(seconds = delay.as_secs(), "archive rate limited; waiting");
                delay
            }
            Err(e @ ArchiveError::Unauthorized(_)) => {
                return Err(RetryOutcome::Fatal(e));
            }
            Err(e @ ArchiveError::Http { .. }) => {
                return Err(RetryOutcome::Fatal(e));
            }
            Err(ArchiveError::Other(e)) => {
                let delay = next_delay(attempt);
                tracing::warn!(error = ?e, seconds = delay.as_secs(), "archive call failed; retrying");
                delay
            }
        };
        attempt = attempt.saturating_add(1);
        tokio::select! {
            () = token.cancelled() => return Err(RetryOutcome::Cancelled),
            () = tokio::time::sleep(delay) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jetstream::archive::BlockRange;
    use crate::jetstream::archive::PlanStats;
    use crate::jetstream::archive::SnapshotPlan;
    use crate::jetstream::jss::tests::compress_block;
    use crate::jetstream::jss::KIND_CREATE;
    use crate::jetstream::jss::KIND_IDENTITY;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::path::PathBuf;

    const DID: &str = "did:plc:me";

    fn filters() -> ReplayFilters {
        ReplayFilters {
            did: DID.to_string(),
            collections: vec![
                "app.bsky.feed.post".to_string(),
                "app.bsky.feed.like".to_string(),
            ],
        }
    }

    fn post_row(seq: u64, did: &str) -> RawEvent {
        let record = serde_json::json!({
            "$type": "app.bsky.feed.post",
            "text": format!("post {seq}"),
            "createdAt": "2026-07-25T21:01:36.005Z",
        });
        RawEvent {
            seq,
            witnessed_at: 1_785_000_000_000_000 + seq as i64,
            indexed_at: 0,
            kind: KIND_CREATE,
            collection: "app.bsky.feed.post".to_string(),
            did: did.to_string(),
            rkey: format!("rkey{seq}"),
            rev: "3rev".to_string(),
            payload: serde_ipld_dagcbor::to_vec(&record).expect("cbor"),
        }
    }

    fn identity_row(seq: u64) -> RawEvent {
        RawEvent {
            seq,
            witnessed_at: 1_785_000_000_000_000 + seq as i64,
            indexed_at: 0,
            kind: KIND_IDENTITY,
            collection: String::new(),
            did: DID.to_string(),
            rkey: String::new(),
            rev: String::new(),
            payload: Vec::new(),
        }
    }

    fn stats() -> PlanStats {
        PlanStats {
            segments_examined: 0,
            segments_matched: 0,
            blocks_matched: 0,
            entries: 0,
        }
    }

    fn blocks_unit(name: &str, ranges: &[(u64, u64)]) -> PlanSegment {
        PlanSegment {
            name: name.to_string(),
            index: 0,
            checksum: "0123456789abcdef".to_string(),
            min_seq: 0,
            max_seq: 0,
            mode: "blocks".to_string(),
            blocks: ranges
                .iter()
                .map(|(first, last)| BlockRange {
                    first: *first,
                    last: *last,
                })
                .collect(),
        }
    }

    /// A programmable fetcher. Plans are consumed in order; blocks are
    /// served from a map keyed by (segment, index).
    #[derive(Default)]
    struct MockFetcher {
        plans: RefCell<Vec<SnapshotPlan>>,
        blocks: HashMap<(String, u64), Vec<u8>>,
        block_errors: RefCell<HashMap<(String, u64), Vec<ArchiveError>>>,
        segments: HashMap<String, PathBuf>,
        plan_calls: RefCell<u64>,
    }

    impl MockFetcher {
        fn add_plan(&self, plan: SnapshotPlan) {
            self.plans.borrow_mut().push(plan);
        }

        fn add_block(&mut self, segment: &str, index: u64, rows: &[RawEvent]) {
            self.blocks
                .insert((segment.to_string(), index), compress_block(rows));
        }

        fn fail_block_once(&self, segment: &str, index: u64, error: ArchiveError) {
            self.block_errors
                .borrow_mut()
                .entry((segment.to_string(), index))
                .or_default()
                .push(error);
        }
    }

    impl Fetcher for MockFetcher {
        async fn plan_snapshot(
            &self,
            _req: &SnapshotRequest,
        ) -> Result<SnapshotPlan, ArchiveError> {
            *self.plan_calls.borrow_mut() += 1;
            let mut plans = self.plans.borrow_mut();
            if plans.is_empty() {
                panic!("unexpected planSnapshot call");
            }
            Ok(plans.remove(0))
        }

        async fn fetch_block(
            &self,
            segment: &str,
            block_index: u64,
        ) -> Result<Vec<u8>, ArchiveError> {
            let key = (segment.to_string(), block_index);
            if let Some(errors) = self.block_errors.borrow_mut().get_mut(&key) {
                if !errors.is_empty() {
                    return Err(errors.remove(0));
                }
            }
            self.blocks
                .get(&key)
                .cloned()
                .ok_or_else(|| ArchiveError::Http {
                    status: 404,
                    body: "BlockNotFound".to_string(),
                })
        }

        async fn fetch_segment(&self, name: &str) -> Result<PathBuf, ArchiveError> {
            self.segments
                .get(name)
                .cloned()
                .ok_or_else(|| ArchiveError::Http {
                    status: 404,
                    body: "SegmentNotFound".to_string(),
                })
        }
    }

    /// Records applied seqs; can be told to fail at a given seq to model
    /// a crash mid-replay.
    #[derive(Default)]
    struct RecordingSink {
        seqs: RefCell<Vec<u64>>,
        fail_at: Option<u64>,
    }

    impl EventSink for RecordingSink {
        async fn apply(&self, event: &Event) -> Result<()> {
            if self.fail_at == Some(event.seq) {
                bail!("sink failure at seq {}", event.seq);
            }
            self.seqs.borrow_mut().push(event.seq);
            Ok(())
        }
    }

    fn plan(planned_through: u64, tip: u64, segments: Vec<PlanSegment>) -> SnapshotPlan {
        SnapshotPlan {
            planned_through_seq: planned_through,
            sealed_tip_seq: tip,
            segments,
            stats: stats(),
        }
    }

    #[tokio::test]
    async fn test_empty_plan_goes_straight_to_handoff() {
        let fetcher = MockFetcher::default();
        fetcher.add_plan(plan(500, 500, vec![]));
        let sink = RecordingSink::default();
        let outcome = replay(&fetcher, &filters(), 120, &sink, &CancellationToken::new())
            .await
            .expect("replay");
        assert_eq!(
            outcome,
            ReplayOutcome {
                last_seq: None,
                sealed_tip_seq: 500,
                events_applied: 0
            }
        );
        assert!(sink.seqs.borrow().is_empty());
    }

    #[tokio::test]
    async fn test_replays_pages_in_order_with_boundary_dedupe() {
        let mut fetcher = MockFetcher::default();
        // Block 0 straddles the floor: seqs 90 and 100 are already
        // applied, 110 is new. Others' events (seq 115) never surface.
        fetcher.add_block(
            "seg_a",
            0,
            &[
                post_row(90, DID),
                post_row(100, DID),
                post_row(110, DID),
                post_row(115, "did:plc:other"),
            ],
        );
        fetcher.add_block("seg_a", 1, &[identity_row(120), post_row(130, DID)]);
        fetcher.add_block("seg_b", 0, &[post_row(200, DID)]);
        fetcher.add_plan(plan(150, 250, vec![blocks_unit("seg_a", &[(0, 1)])]));
        fetcher.add_plan(plan(250, 250, vec![blocks_unit("seg_b", &[(0, 0)])]));

        let sink = RecordingSink::default();
        let outcome = replay(&fetcher, &filters(), 100, &sink, &CancellationToken::new())
            .await
            .expect("replay");

        assert_eq!(*sink.seqs.borrow(), vec![110, 120, 130, 200]);
        assert_eq!(outcome.last_seq, Some(200));
        assert_eq!(outcome.sealed_tip_seq, 250);
        assert_eq!(outcome.events_applied, 4);
        assert_eq!(*fetcher.plan_calls.borrow(), 2);
    }

    #[tokio::test]
    async fn test_sink_failure_surfaces_and_rerun_does_not_duplicate() {
        let build = || {
            let mut fetcher = MockFetcher::default();
            fetcher.add_block("seg_a", 0, &[post_row(110, DID), post_row(120, DID)]);
            fetcher
        };

        // First run: the sink dies mid-block, exactly like a crash after
        // seq 110 was durably applied.
        let fetcher = build();
        fetcher.add_plan(plan(200, 200, vec![blocks_unit("seg_a", &[(0, 0)])]));
        let crashing = RecordingSink {
            fail_at: Some(120),
            ..Default::default()
        };
        let err = replay(
            &fetcher,
            &filters(),
            100,
            &crashing,
            &CancellationToken::new(),
        )
        .await;
        assert!(err.is_err());
        assert_eq!(*crashing.seqs.borrow(), vec![110]);

        // Restart replans from the stored cursor; 110 is not re-applied.
        let fetcher = build();
        fetcher.add_plan(plan(200, 200, vec![blocks_unit("seg_a", &[(0, 0)])]));
        let sink = RecordingSink::default();
        replay(&fetcher, &filters(), 110, &sink, &CancellationToken::new())
            .await
            .expect("replay");
        assert_eq!(*sink.seqs.borrow(), vec![120]);
    }

    #[tokio::test(start_paused = true)]
    async fn test_rate_limit_waits_and_retries() {
        let mut fetcher = MockFetcher::default();
        fetcher.add_block("seg_a", 0, &[post_row(110, DID)]);
        fetcher.fail_block_once(
            "seg_a",
            0,
            ArchiveError::RateLimited {
                retry_after: Some(std::time::Duration::from_secs(30)),
            },
        );
        fetcher.add_plan(plan(200, 200, vec![blocks_unit("seg_a", &[(0, 0)])]));

        let sink = RecordingSink::default();
        let started = tokio::time::Instant::now();
        replay(&fetcher, &filters(), 100, &sink, &CancellationToken::new())
            .await
            .expect("replay");
        assert_eq!(*sink.seqs.borrow(), vec![110]);
        assert!(
            started.elapsed() >= std::time::Duration::from_secs(30),
            "the Retry-After pause must be honoured"
        );
    }

    #[tokio::test]
    async fn test_cancellation_wins_over_rate_limit_pause() {
        let mut fetcher = MockFetcher::default();
        fetcher.add_block("seg_a", 0, &[post_row(110, DID)]);
        fetcher.fail_block_once(
            "seg_a",
            0,
            ArchiveError::RateLimited {
                retry_after: Some(std::time::Duration::from_secs(3600)),
            },
        );
        fetcher.add_plan(plan(200, 200, vec![blocks_unit("seg_a", &[(0, 0)])]));

        let token = CancellationToken::new();
        let sink = RecordingSink::default();
        let cancel = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            replay(&fetcher, &filters(), 100, &sink, &token),
        )
        .await
        .expect("must not wait out the rate limit")
        .expect("cancellation is not an error");
        assert_eq!(outcome.events_applied, 0);
    }

    #[tokio::test]
    async fn test_vanished_segment_triggers_replan() {
        let mut fetcher = MockFetcher::default();
        // The first plan names a segment that compaction has removed;
        // only the second plan's block exists.
        fetcher.add_block("seg_new", 0, &[post_row(110, DID)]);
        fetcher.add_plan(plan(200, 200, vec![blocks_unit("seg_gone", &[(0, 0)])]));
        fetcher.add_plan(plan(200, 200, vec![blocks_unit("seg_new", &[(0, 0)])]));

        let sink = RecordingSink::default();
        replay(&fetcher, &filters(), 100, &sink, &CancellationToken::new())
            .await
            .expect("replay");
        assert_eq!(*sink.seqs.borrow(), vec![110]);
        assert_eq!(*fetcher.plan_calls.borrow(), 2);
    }

    #[tokio::test]
    async fn test_unauthorised_is_fatal() {
        let fetcher = MockFetcher::default();
        fetcher.add_plan(plan(200, 200, vec![blocks_unit("seg_a", &[(0, 0)])]));
        fetcher.fail_block_once(
            "seg_a",
            0,
            ArchiveError::Unauthorized("invalid bearer credential".to_string()),
        );
        let sink = RecordingSink::default();
        let err = replay(&fetcher, &filters(), 100, &sink, &CancellationToken::new())
            .await
            .expect_err("must fail");
        assert!(err.to_string().contains("getBlock"));
    }

    #[tokio::test]
    async fn test_whole_segment_unit_is_decoded_and_spool_removed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seg_whole.jss");
        // Reuse the jss test helpers to write a real sealed file.
        crate::jetstream::jss::tests::write_segment_fixture(
            &path,
            &[vec![post_row(110, DID), post_row(120, DID)]],
        );

        let mut fetcher = MockFetcher::default();
        fetcher
            .segments
            .insert("seg_whole.jss".to_string(), path.clone());
        let mut unit = blocks_unit("seg_whole.jss", &[]);
        unit.mode = "segment".to_string();
        unit.blocks.clear();
        fetcher.add_plan(plan(200, 200, vec![unit]));

        let sink = RecordingSink::default();
        replay(&fetcher, &filters(), 100, &sink, &CancellationToken::new())
            .await
            .expect("replay");
        assert_eq!(*sink.seqs.borrow(), vec![110, 120]);
        assert!(!path.exists(), "the spool file is deleted after ingest");
    }

    #[tokio::test]
    async fn test_unknown_unit_mode_is_an_error() {
        let fetcher = MockFetcher::default();
        let mut unit = blocks_unit("seg_a", &[]);
        unit.mode = "torrent".to_string();
        fetcher.add_plan(plan(200, 200, vec![unit]));
        let sink = RecordingSink::default();
        assert!(
            replay(&fetcher, &filters(), 100, &sink, &CancellationToken::new())
                .await
                .is_err()
        );
    }
}
