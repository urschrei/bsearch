mod backfill_links;
mod config;
mod instapaper;
mod jetstream;
mod links;
mod resolver;

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use bsearch_core::db::Database;
use bsearch_core::embed::Embedder;
use clap::Parser;
use clap::Subcommand;
use jiff::civil::Date;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Bluesky post and like monitor: the ingest daemon.
///
/// With no subcommand, follows the account's Jetstream and indexes posts
/// and likes until stopped.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    mode: Option<Mode>,
}

#[derive(Subcommand)]
enum Mode {
    /// Queue the links in already-indexed liked posts for Instapaper.
    ///
    /// Fetches the posts again to read their link facets and cards, and
    /// leaves the links in the queue the running daemon drains.
    BackfillLinks {
        /// Only posts created on or after this date (YYYY-MM-DD).
        #[arg(long, value_parser = parse_date)]
        since: Date,
        /// Fetch and list the links without queueing them.
        #[arg(long)]
        dry_run: bool,
    },
}

fn parse_date(raw: &str) -> Result<Date, String> {
    Date::strptime("%Y-%m-%d", raw).map_err(|e| format!("{raw}: {e}"))
}

/// Send a macOS notification, as `_notify` does in `src/bsearch/jetstream.py`.
///
/// Failures are deliberately swallowed: losing a notification must never take
/// the daemon down.
pub fn notify(title: &str, message: &str) {
    let script = format!("display notification \"{message}\" with title \"{title}\"");
    if let Err(e) = Command::new("osascript").arg("-e").arg(&script).spawn() {
        tracing::debug!(error = ?e, "Failed to send notification");
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // The dependency graph carries both rustls crypto providers (ring via
    // tokio-websockets, aws-lc-rs via its rustls-native-roots feature), and
    // rustls panics at first use unless exactly one is selected.
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("a rustls crypto provider was already installed"))?;

    // ONNX Runtime logs its entire graph-optimisation pass at INFO, which is
    // hundreds of lines every time a session is loaded, so it is pinned to
    // warn unless RUST_LOG says otherwise.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn")),
        )
        .init();

    // A current-thread runtime is plenty: the workload is a single WebSocket
    // and two timers, and it keeps the thread-stack overhead down.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run(cli.mode))
}

async fn run(mode: Option<Mode>) -> Result<()> {
    let mut config = config::Config::from_env(None)?;

    if let Some(Mode::BackfillLinks { since, dry_run }) = mode {
        return backfill_links::run(&config, since, dry_run).await;
    }

    let resolver = Arc::new(resolver::Resolver::new(&config));
    let did = resolver.login(&config).await?;
    if config.did.is_empty() {
        config.did = did;
    }

    let db = Database::open_read_write(&config.db_path)
        .with_context(|| format!("Failed to open database at {}", config.db_path.display()))?;
    let db = Arc::new(Mutex::new(db));

    let token = CancellationToken::new();
    spawn_signal_handler(token.clone());

    tracing::info!(handle = %config.handle, did = %config.did, "Starting service");

    let handler = Arc::new(jetstream::IngestHandler::new(
        db.clone(),
        config.handle.clone(),
    ));

    let likes = tokio::spawn(resolve_likes_loop(
        config.clone(),
        db.clone(),
        resolver,
        token.clone(),
    ));
    let embeddings = tokio::spawn(embedding_loop(config.clone(), db.clone(), token.clone()));
    let instapaper = config.instapaper.as_ref().map(|settings| {
        tracing::info!(folder = %settings.folder, "Filing links from liked posts in Instapaper");
        tokio::spawn(instapaper_loop(
            settings.clone(),
            config.instapaper_batch_interval,
            db.clone(),
            token.clone(),
        ))
    });

    jetstream::run(&config, db, handler, token).await?;
    likes.await??;
    embeddings.await??;
    if let Some(instapaper) = instapaper {
        instapaper.await??;
    }

    tracing::info!("Service stopped");
    Ok(())
}

/// How many posts to embed per batch, matching the Python default.
const EMBEDDING_BATCH_SIZE: usize = 100;

/// Periodically embed any posts that lack an embedding.
///
/// Port of `Service._generate_embeddings_loop`, with one addition: the ONNX
/// session is loaded on demand and dropped again once the loop has been idle
/// for `embedder_idle_timeout`. Reloading takes well under a second, and this
/// is the difference between an idle daemon holding roughly 20 MB and 90 MB.
async fn embedding_loop(
    config: config::Config,
    db: Arc<Mutex<Database>>,
    token: CancellationToken,
) -> Result<()> {
    let interval = Duration::from_secs(config.embedding_batch_interval);
    let idle_timeout = Duration::from_secs(config.embedder_idle_timeout);
    let mut embedder: Option<Embedder> = None;
    let mut idle_since: Option<Instant> = None;

    loop {
        tokio::select! {
            () = token.cancelled() => break,
            () = tokio::time::sleep(interval) => {}
        }

        let pending = {
            let db = db.lock().await;
            db.get_posts_without_embeddings(EMBEDDING_BATCH_SIZE)?
        };

        if pending.is_empty() {
            let idle_long_enough = idle_since.is_some_and(|since| since.elapsed() >= idle_timeout);
            if embedder.is_some() && idle_long_enough {
                embedder = None;
                idle_since = None;
                tracing::debug!("Dropped idle ONNX session");
            }
            continue;
        }

        if embedder.is_none() {
            match Embedder::load(&config.model_dir) {
                Ok(e) => embedder = Some(e),
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to load embedding model");
                    continue;
                }
            }
        }

        // Inference is CPU-bound and would otherwise stall the Jetstream
        // reader on this single-threaded runtime, so it runs on the blocking
        // pool. The embedder is moved in and handed back out again.
        let count = pending.len();
        let owned = embedder.take().expect("embedder loaded above");
        let result = tokio::task::spawn_blocking(move || {
            let mut owned = owned;
            let mut out = Vec::with_capacity(pending.len());
            for (id, text) in &pending {
                match owned.encode(text) {
                    Ok(vector) => out.push((*id, vector)),
                    Err(e) => tracing::error!(error = ?e, id, "Failed to embed post"),
                }
            }
            (owned, out)
        })
        .await;

        let (returned, vectors) = match result {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!(error = ?e, "Embedding task panicked");
                continue;
            }
        };
        embedder = Some(returned);

        if !vectors.is_empty() {
            let mut db = db.lock().await;
            match db.store_embeddings(&vectors) {
                Ok(()) => tracing::info!("Generated embeddings for {} posts", vectors.len()),
                Err(e) => tracing::error!(error = ?e, "Failed to store embeddings"),
            }
        }

        // Start the idle clock only once the backlog is drained.
        idle_since = if count < EMBEDDING_BATCH_SIZE {
            Some(Instant::now())
        } else {
            None
        };
    }
    Ok(())
}

/// Periodically drain the queued like URIs and index the posts behind them.
///
/// Port of `Service._resolve_likes_loop`, except that the queue lives in the
/// database rather than in memory. URIs are read, not removed, and deleted only
/// once the batch has been resolved; a failure -- or a crash, or a restart --
/// therefore leaves them queued for the next pass instead of losing them. The
/// retry is safe because inserting a post already present is ignored.
///
/// A resolved batch is cleared in full, including URIs the API returned nothing
/// for. Those are posts that have since been deleted, and keeping them would
/// mean retrying them forever.
async fn resolve_likes_loop(
    config: config::Config,
    db: Arc<Mutex<Database>>,
    resolver: Arc<resolver::Resolver>,
    token: CancellationToken,
) -> Result<()> {
    let interval = Duration::from_secs(config.like_batch_interval);
    let file_links = config.instapaper.is_some();
    loop {
        tokio::select! {
            () = token.cancelled() => break,
            () = tokio::time::sleep(interval) => {}
        }

        let uris = {
            let db = db.lock().await;
            db.take_pending_likes(resolver::MAX_URIS_PER_CALL)?
        };
        if uris.is_empty() {
            continue;
        }

        match resolver.resolve_post_uris(&uris).await {
            Ok(posts) => {
                let db = db.lock().await;
                for resolved in &posts {
                    let post = &resolved.post;
                    // Links are collected only when there is somewhere to
                    // send them; otherwise the queue would grow unread.
                    let links: Vec<_> = if file_links {
                        resolved
                            .links
                            .iter()
                            .map(|link| links::to_pending(post, link))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    match db.insert_post_with_links(post, &links) {
                        Ok(Some(_)) => tracing::info!(uri = %post.uri, "Indexed liked post"),
                        Ok(None) => {}
                        Err(e) => {
                            tracing::error!(error = ?e, uri = %post.uri, "Failed to index liked post")
                        }
                    }
                }
                if let Err(e) = db.remove_pending_likes(&uris) {
                    // Left queued, so the batch is simply resolved again.
                    tracing::error!(error = ?e, "Failed to clear resolved likes from the queue");
                }
            }
            Err(e) => {
                tracing::error!(error = ?e, "Failed to resolve like batch; leaving it queued");
            }
        }
    }
    Ok(())
}

/// How many queued links to submit per pass.
const INSTAPAPER_BATCH_SIZE: usize = 10;
/// Pause between consecutive submissions, so a burst of likes does not
/// become a burst of requests.
const INSTAPAPER_PACING: Duration = Duration::from_secs(1);
/// How long to wait after a failure before trying again.
const INSTAPAPER_RETRY_DELAY: Duration = Duration::from_secs(60);
/// How long to wait after Instapaper reports its rate limit exceeded.
const INSTAPAPER_RATE_LIMIT_DELAY: Duration = Duration::from_secs(300);

/// Periodically send queued links to Instapaper.
///
/// The queue is drained the way the like queue is: a link is read, sent,
/// and only then removed, so a failure or a restart leaves it for the next
/// pass. Resending is harmless, since Instapaper treats a URL it already
/// holds as an update rather than a duplicate. A link Instapaper rejects
/// outright -- an invalid URL, a publisher that has opted out -- is dropped,
/// because retrying it could never succeed.
///
/// Nothing in here returns early on error: a failure of any kind is logged
/// and waited out, since the alternative is a task that dies quietly while
/// the daemon carries on and the queue fills. The session is made on the
/// first pass, before there is anything to send, so that bad credentials
/// show up in the log at startup rather than at the first like; it is then
/// kept for as long as it works. It is discarded if the folder it
/// was bound to disappears or the token stops being accepted, so that the
/// next pass rebuilds it; and a failure to connect is reported once by
/// notification and thereafter only logged, so a revoked token does not
/// produce a notification every minute.
async fn instapaper_loop(
    settings: config::InstapaperConfig,
    interval_seconds: u64,
    db: Arc<Mutex<Database>>,
    token: CancellationToken,
) -> Result<()> {
    let interval = Duration::from_secs(interval_seconds);
    let mut session: Option<instapaper::Session> = None;
    let mut connect_failure_notified = false;
    let mut delay = interval;

    loop {
        tokio::select! {
            () = token.cancelled() => break,
            () = tokio::time::sleep(delay) => {}
        }
        delay = interval;

        let current = match &session {
            Some(current) => current,
            None => match instapaper::Session::connect(&settings).await {
                Ok(connected) => {
                    tracing::info!(
                        folder = %connected.folder_title(),
                        folder_id = connected.folder_id(),
                        "Connected to Instapaper"
                    );
                    connect_failure_notified = false;
                    session.insert(connected)
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to connect to Instapaper");
                    if !connect_failure_notified {
                        notify("bsearch", &format!("Instapaper connection failed: {e}"));
                        connect_failure_notified = true;
                    }
                    delay = INSTAPAPER_RETRY_DELAY;
                    continue;
                }
            },
        };

        let links = match db.lock().await.take_pending_links(INSTAPAPER_BATCH_SIZE) {
            Ok(links) => links,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to read the Instapaper queue");
                delay = INSTAPAPER_RETRY_DELAY;
                continue;
            }
        };
        if links.is_empty() {
            continue;
        }

        let mut reset_session = false;
        for (i, link) in links.iter().enumerate() {
            if i > 0 {
                tokio::select! {
                    () = token.cancelled() => return Ok(()),
                    () = tokio::time::sleep(INSTAPAPER_PACING) => {}
                }
            }
            match current.add_bookmark(link).await {
                Ok(bookmark_id) => {
                    tracing::info!(url = %link.url, bookmark_id, "Saved link to Instapaper");
                }
                Err(e) => match e.disposition() {
                    instapaper::Disposition::DropLink => {
                        tracing::warn!(url = %link.url, error = %e, "Instapaper rejected link; dropping it");
                    }
                    instapaper::Disposition::ReconnectFolder => {
                        tracing::warn!(error = %e, "Instapaper folder is gone; reconnecting");
                        reset_session = true;
                        break;
                    }
                    instapaper::Disposition::ResetSession => {
                        tracing::warn!(error = %e, "Instapaper no longer accepts the session; reconnecting");
                        reset_session = true;
                        delay = INSTAPAPER_RETRY_DELAY;
                        break;
                    }
                    instapaper::Disposition::RateLimited => {
                        tracing::warn!(error = %e, "Instapaper rate limit reached; pausing");
                        delay = INSTAPAPER_RATE_LIMIT_DELAY;
                        break;
                    }
                    instapaper::Disposition::Retry => {
                        tracing::error!(url = %link.url, error = %e, "Failed to save link to Instapaper; leaving it queued");
                        delay = INSTAPAPER_RETRY_DELAY;
                        break;
                    }
                },
            }
            // Reached only for a link that is finished with, saved or
            // dropped. Failing to forget it means it is sent once more.
            if let Err(e) = db.lock().await.remove_pending_link(&link.url) {
                tracing::error!(error = ?e, url = %link.url, "Failed to remove link from the Instapaper queue");
                delay = INSTAPAPER_RETRY_DELAY;
                break;
            }
        }
        if reset_session {
            session = None;
        }
    }
    Ok(())
}

/// Trip the cancellation token on SIGTERM or SIGINT, replacing the
/// `loop.add_signal_handler` calls in `src/bsearch/service.py`.
fn spawn_signal_handler(token: CancellationToken) {
    tokio::spawn(async move {
        let mut sigterm =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to install SIGTERM handler");
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
        tracing::info!("Received shutdown signal");
        token.cancel();
    });
}
