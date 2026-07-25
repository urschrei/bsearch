mod config;
mod jetstream;

use std::collections::VecDeque;
use std::process::Command;
use std::sync::Arc;

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
    let config = config::Config::from_env(None)?;

    anyhow::ensure!(
        !config.did.is_empty(),
        "No DID configured. Set BSEARCH_DID in .env."
    );

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

    jetstream::run(&config, db, handler, token).await?;

    tracing::info!("Service stopped");
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
