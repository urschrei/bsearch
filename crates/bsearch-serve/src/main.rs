mod config;
mod jetstream;
mod resolver;

use std::collections::VecDeque;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use atproto_jetstream::CancellationToken;
use bsearch_core::db::Database;
use tokio::sync::Mutex;

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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // A current-thread runtime is plenty: the workload is a single WebSocket
    // and two timers, and it keeps the thread-stack overhead down.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run())
}

async fn run() -> Result<()> {
    let mut config = config::Config::from_env(None)?;

    let resolver = Arc::new(resolver::Resolver::new(&config));
    let did = resolver.login(&config).await?;
    if config.did.is_empty() {
        config.did = did;
    }

    let db = Database::open_read_write(&config.db_path)
        .with_context(|| format!("Failed to open database at {}", config.db_path.display()))?;
    let db = Arc::new(Mutex::new(db));
    let pending_likes = Arc::new(Mutex::new(VecDeque::new()));

    let token = CancellationToken::new();
    spawn_signal_handler(token.clone());

    tracing::info!(handle = %config.handle, did = %config.did, "Starting service");

    let handler = Arc::new(jetstream::IngestHandler::new(
        db.clone(),
        pending_likes.clone(),
        config.handle.clone(),
    ));

    let likes = tokio::spawn(resolve_likes_loop(
        config.clone(),
        db.clone(),
        pending_likes,
        resolver,
        token.clone(),
    ));

    jetstream::run(&config, db, handler, token).await?;
    likes.await??;

    tracing::info!("Service stopped");
    Ok(())
}

/// Periodically drain the queued like URIs and index the posts behind them.
///
/// Port of `Service._resolve_likes_loop`, including its re-queueing of a batch
/// that failed to resolve.
async fn resolve_likes_loop(
    config: config::Config,
    db: Arc<Mutex<Database>>,
    pending_likes: Arc<Mutex<VecDeque<String>>>,
    resolver: Arc<resolver::Resolver>,
    token: CancellationToken,
) -> Result<()> {
    let interval = Duration::from_secs(config.like_batch_interval);
    loop {
        tokio::select! {
            () = token.cancelled() => break,
            () = tokio::time::sleep(interval) => {}
        }

        let uris: Vec<String> = {
            let mut queue = pending_likes.lock().await;
            let take = queue.len().min(resolver::MAX_URIS_PER_CALL);
            queue.drain(..take).collect()
        };
        if uris.is_empty() {
            continue;
        }

        match resolver.resolve_post_uris(&uris).await {
            Ok(posts) => {
                let db = db.lock().await;
                for post in &posts {
                    match db.insert_post(post) {
                        Ok(Some(_)) => tracing::info!(uri = %post.uri, "Indexed liked post"),
                        Ok(None) => {}
                        Err(e) => {
                            tracing::error!(error = ?e, uri = %post.uri, "Failed to index liked post")
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = ?e, "Failed to resolve like batch");
                let mut queue = pending_likes.lock().await;
                for uri in uris.into_iter().rev() {
                    queue.push_front(uri);
                }
            }
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
