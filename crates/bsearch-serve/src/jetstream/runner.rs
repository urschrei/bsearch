//! The consumption loop: archive catch-up when there is ground to make
//! up, then the live tail, reconnecting until cancelled.
//!
//! With an API key the daemon is self-healing: any gap -- downtime, a
//! cursor below the live retention floor, even a fresh database -- is
//! swept from the sealed archive before the live tail resumes at the
//! sealed tip. Without a key it degrades to the live tail alone.

use std::sync::Arc;
use std::time::Duration;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use bsearch_core::db::Database;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::archive::ArchiveError;
use super::archive::HttpArchive;
use super::events::classify_cursor;
use super::events::Frame;
use super::events::StoredCursor;
use super::ingest::IngestHandler;
use super::ingest::COLLECTION_LIKE;
use super::ingest::COLLECTION_POST;
use super::live;
use super::live::ConnectError;
use super::live::LiveParams;
use super::replay;
use super::replay::ReplayFilters;
use super::replay::ReplayOutcome;
use crate::config::Config;
use crate::notify;

const RECONNECT_DELAY_SECS: u64 = 5;

/// Catch-up cycles that fail to advance the cursor before the daemon
/// gives up; matches the reference client's bound on pathological loops.
const MAX_STALLED_CATCHUPS: u32 = 5;

/// Replays applying at least this many events merit a notification; the
/// routine trickle after a recycle does not.
const NOTIFY_REPLAY_EVENTS: u64 = 100;

/// Where the archive sweep starts, given the stored cursor.
///
/// A legacy time_us value (or no cursor at all) triggers a full-history
/// sweep: the archive reaches back to the beginning, ingestion is
/// idempotent, and a plan filtered to one DID keeps the transfer small.
/// This is also what retires the Python backfill.
fn catchup_after_seq(stored: Option<i64>) -> u64 {
    match stored.map(classify_cursor) {
        Some(StoredCursor::Seq(seq)) => seq,
        Some(StoredCursor::LegacyTimeUs(_)) | None => 0,
    }
}

/// The live cursor when no archive is available. Any stored value passes
/// through unchanged: the server tells seqs from unix-microsecond
/// timestamps by magnitude and translates the latter.
fn live_cursor_without_archive(stored: Option<i64>) -> Option<u64> {
    stored.map(|v| v.max(0) as u64)
}

/// The live cursor after a completed sweep: `max(sealed tip, last
/// applied)`, per the cutover contract. An empty archive yields no cursor
/// at all -- a cursor of 0 would mean "replay everything", not "tip".
fn cutover_cursor(outcome: &ReplayOutcome) -> Option<u64> {
    let cursor = outcome.sealed_tip_seq.max(outcome.last_seq.unwrap_or(0));
    (cursor > 0).then_some(cursor)
}

/// Cancel `connection_token` once `lifetime` has elapsed.
///
/// A socket that dies without a close frame leaves the reader parked on a
/// read that never completes, and a subscription filtered to one DID is
/// silent whenever the account is idle, so nothing but time distinguishes
/// the two. Recycling on a timer bounds the stall. The caller aborts the
/// returned handle when the connection ends for any other reason.
fn spawn_connection_watchdog(
    connection_token: CancellationToken,
    lifetime: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        tokio::time::sleep(lifetime).await;
        connection_token.cancel();
    })
}

/// Run the Jetstream consumer, reconnecting until cancelled.
pub async fn run(
    config: &Config,
    db: Arc<Mutex<Database>>,
    handler: Arc<IngestHandler>,
    token: CancellationToken,
) -> Result<()> {
    let filters = ReplayFilters {
        did: config.did.clone(),
        collections: vec![COLLECTION_POST.to_string(), COLLECTION_LIKE.to_string()],
    };
    let mut archive = match &config.jetstream_key {
        Some(key) => Some(
            HttpArchive::new(&config.jetstream_hostname, key, &config.segment_spool_dir)
                .context("archive client")?,
        ),
        None => {
            tracing::warn!(
                "No BSEARCH_JETSTREAM_KEY configured; history beyond the live \
                 replay buffer cannot be recovered"
            );
            None
        }
    };

    // Catch up from the archive at startup and after any abnormal end;
    // a quiet watchdog recycle skips straight back to the live tail.
    let mut needs_catchup = true;
    let mut stalled_catchups = 0u32;
    let mut last_rejected_cursor: Option<Option<u64>> = None;

    while !token.is_cancelled() {
        let stored = {
            let db = db.lock().await;
            db.get_cursor()?
        };

        let live_cursor = match &archive {
            Some(fetcher) if needs_catchup => {
                let after_seq = catchup_after_seq(stored);
                tracing::info!(after_seq, "Sweeping the sealed archive");
                match replay::replay(fetcher, &filters, after_seq, handler.as_ref(), &token).await {
                    Ok(outcome) => {
                        needs_catchup = false;
                        if outcome.events_applied > 0 {
                            tracing::info!(
                                events = outcome.events_applied,
                                sealed_tip = outcome.sealed_tip_seq,
                                "Archive sweep complete"
                            );
                        }
                        if outcome.events_applied >= NOTIFY_REPLAY_EVENTS {
                            notify(
                                "bsearch: history recovered",
                                &format!(
                                    "Replayed {} events from the Jetstream archive.",
                                    outcome.events_applied
                                ),
                            );
                        }
                        cutover_cursor(&outcome)
                    }
                    Err(e) if is_unauthorized(&e) => {
                        // A revoked key cannot heal itself; run degraded
                        // rather than hammering the endpoint.
                        tracing::error!(error = ?e, "Jetstream API key rejected");
                        notify(
                            "bsearch: invalid Jetstream API key",
                            "Archive replay is disabled until the key is fixed.",
                        );
                        archive = None;
                        live_cursor_without_archive(stored)
                    }
                    Err(e) => {
                        tracing::error!(error = ?e, "Archive sweep failed; will retry");
                        live_cursor_without_archive(stored)
                    }
                }
            }
            Some(_) => stored.map(|v| v.max(0) as u64),
            None => live_cursor_without_archive(stored),
        };

        if token.is_cancelled() {
            break;
        }

        let params = LiveParams {
            hostname: config.jetstream_hostname.clone(),
            dids: vec![config.did.clone()],
            collections: filters.collections.clone(),
            cursor: live_cursor,
        };
        tracing::info!(cursor = ?live_cursor, host = %config.jetstream_hostname, "Connecting to Jetstream");

        match live::connect(&params).await {
            Ok(connection) => {
                last_rejected_cursor = None;
                stalled_catchups = 0;

                // A child token so the watchdog can end this connection
                // without stopping the service; cancelling the parent
                // still cancels the child.
                let connection_token = token.child_token();
                let lifetime = Duration::from_secs(config.max_connection_seconds);
                let watchdog = spawn_connection_watchdog(connection_token.clone(), lifetime);
                let end = read_until_end(connection, handler.as_ref(), &connection_token).await;
                // Whether the connection ended on its own or on the
                // timer, the watchdog has no further work.
                watchdog.abort();

                match end {
                    ConnectionEnd::Recycled if token.is_cancelled() => break,
                    ConnectionEnd::Recycled => {
                        tracing::debug!(
                            seconds = lifetime.as_secs(),
                            "Recycling Jetstream connection"
                        );
                    }
                    ConnectionEnd::Closed => {
                        tracing::warn!("Jetstream connection closed");
                        notify("bsearch: disconnected", "Jetstream connection closed.");
                        needs_catchup = archive.is_some();
                    }
                    ConnectionEnd::Errored(e) => {
                        tracing::error!(error = ?e, "Jetstream connection error");
                        notify("bsearch: error", "Jetstream connection error.");
                        needs_catchup = archive.is_some();
                    }
                }
            }
            Err(ConnectError::Rejected { status: 400 }) if archive.is_some() => {
                // The filters are constant and correct, so a 400 on a
                // well-formed URL is CursorTooOld: the cursor fell below
                // the live retention floor. Sweep the gap from the
                // archive and try again -- bounded, in case the rejection
                // is something a sweep cannot fix.
                if last_rejected_cursor == Some(live_cursor) {
                    stalled_catchups += 1;
                } else {
                    stalled_catchups = 1;
                    last_rejected_cursor = Some(live_cursor);
                }
                if stalled_catchups > MAX_STALLED_CATCHUPS {
                    notify(
                        "bsearch: stalled",
                        "Jetstream keeps rejecting the cursor; the daemon gave up.",
                    );
                    bail!(
                        "live cursor {live_cursor:?} rejected {stalled_catchups} times \
                         without progress"
                    );
                }
                tracing::warn!(
                    cursor = ?live_cursor,
                    "Live cursor rejected (CursorTooOld); re-entering archive catch-up"
                );
                needs_catchup = true;
            }
            Err(ConnectError::Rejected { status }) => {
                tracing::warn!(status, "Jetstream refused the connection");
                if status == 400 {
                    notify(
                        "bsearch: gap in history",
                        "The cursor is too old for the live buffer and no API key \
                         is configured to recover the gap.",
                    );
                    // Nothing can serve the gap; resume from the tip
                    // rather than failing forever.
                    let db = db.lock().await;
                    db.clear_cursor()?;
                }
            }
            Err(ConnectError::Other(e)) => {
                tracing::error!(error = ?e, "Jetstream connection failed");
            }
        }

        if token.is_cancelled() {
            break;
        }
        tracing::info!("Reconnecting in {} seconds...", RECONNECT_DELAY_SECS);
        tokio::select! {
            () = token.cancelled() => break,
            () = tokio::time::sleep(Duration::from_secs(RECONNECT_DELAY_SECS)) => {}
        }
    }

    Ok(())
}

enum ConnectionEnd {
    /// The watchdog (or shutdown) cancelled the connection token.
    Recycled,
    /// The server closed the stream.
    Closed,
    Errored(anyhow::Error),
}

async fn read_until_end(
    mut connection: live::Live,
    handler: &IngestHandler,
    connection_token: &CancellationToken,
) -> ConnectionEnd {
    loop {
        tokio::select! {
            () = connection_token.cancelled() => {
                connection.close().await;
                return ConnectionEnd::Recycled;
            }
            frame = connection.next_frame() => match frame {
                Ok(Some(Frame::Event(event))) => {
                    if let Err(e) = handler.handle_event(&event).await {
                        // The cursor was not advanced, so reconnecting
                        // replays this event rather than skipping it.
                        return ConnectionEnd::Errored(e);
                    }
                }
                Ok(Some(Frame::Info { name, message })) => {
                    tracing::warn!(name, message = ?message, "Jetstream advisory");
                }
                Ok(Some(Frame::Error { error, message })) => {
                    return ConnectionEnd::Errored(anyhow::anyhow!(
                        "server error frame {error}: {message:?}"
                    ));
                }
                Ok(Some(Frame::Unknown)) => {}
                Ok(None) => return ConnectionEnd::Closed,
                Err(e) => return ConnectionEnd::Errored(e),
            }
        }
    }
}

fn is_unauthorized(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| matches!(cause.downcast_ref(), Some(ArchiveError::Unauthorized(_))))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY_TIME_US: i64 = 1_785_009_595_019_071;

    #[test]
    fn test_catchup_starts_from_a_seq_cursor() {
        assert_eq!(catchup_after_seq(Some(42)), 42);
    }

    #[test]
    fn test_catchup_sweeps_full_history_for_legacy_or_missing_cursors() {
        // The v1 daemon stored time_us; the sweep that migrates it doubles
        // as the backfill replacement, so it starts from the beginning.
        assert_eq!(catchup_after_seq(Some(LEGACY_TIME_US)), 0);
        assert_eq!(catchup_after_seq(None), 0);
    }

    #[test]
    fn test_live_cursor_without_archive_passes_values_through() {
        assert_eq!(live_cursor_without_archive(None), None);
        assert_eq!(live_cursor_without_archive(Some(42)), Some(42));
        // A legacy timestamp is translated by the server, not by us.
        assert_eq!(
            live_cursor_without_archive(Some(LEGACY_TIME_US)),
            Some(LEGACY_TIME_US as u64)
        );
    }

    #[test]
    fn test_cutover_takes_the_later_of_tip_and_last_applied() {
        let outcome = ReplayOutcome {
            last_seq: Some(900),
            sealed_tip_seq: 500,
            events_applied: 10,
        };
        assert_eq!(cutover_cursor(&outcome), Some(900));
        let outcome = ReplayOutcome {
            last_seq: Some(400),
            sealed_tip_seq: 500,
            events_applied: 10,
        };
        assert_eq!(cutover_cursor(&outcome), Some(500));
    }

    #[test]
    fn test_cutover_on_an_empty_archive_starts_at_the_tip() {
        // A cursor of 0 would mean "replay everything"; no cursor means
        // "start at the live tip", which is what an empty archive implies.
        let outcome = ReplayOutcome {
            last_seq: None,
            sealed_tip_seq: 0,
            events_applied: 0,
        };
        assert_eq!(cutover_cursor(&outcome), None);
    }

    #[tokio::test(start_paused = true)]
    async fn test_watchdog_ends_connection_but_not_service() {
        let service = CancellationToken::new();
        let connection = service.child_token();

        let handle = spawn_connection_watchdog(connection.clone(), Duration::from_secs(300));

        tokio::time::advance(Duration::from_secs(299)).await;
        assert!(
            !connection.is_cancelled(),
            "should still be within lifetime"
        );

        tokio::time::advance(Duration::from_secs(2)).await;
        handle.await.expect("watchdog panicked");

        assert!(
            connection.is_cancelled(),
            "a connection that outlives its lifetime must be torn down"
        );
        assert!(
            !service.is_cancelled(),
            "recycling one connection must not stop the service"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_aborted_watchdog_leaves_connection_alone() {
        // The usual case: the connection ends by itself and the caller
        // aborts the timer. Nothing should cancel the token afterwards.
        let service = CancellationToken::new();
        let connection = service.child_token();

        let handle = spawn_connection_watchdog(connection.clone(), Duration::from_secs(300));
        handle.abort();

        tokio::time::advance(Duration::from_secs(600)).await;
        assert!(!connection.is_cancelled());
        assert!(!service.is_cancelled());
    }

    #[tokio::test(start_paused = true)]
    async fn test_cancelling_service_cancels_live_connection() {
        // Shutdown must propagate through the child token, or the reader
        // would keep running after the service was told to stop.
        let service = CancellationToken::new();
        let connection = service.child_token();
        let _watchdog = spawn_connection_watchdog(connection.clone(), Duration::from_secs(300));

        service.cancel();
        assert!(connection.is_cancelled());
    }
}
