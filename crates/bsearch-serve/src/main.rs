mod config;
mod jetstream;
mod resolver;

use std::collections::VecDeque;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use atproto_jetstream::CancellationToken;
use bsearch_core::db::Database;
use bsearch_core::embed::Embedder;
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
    let embeddings = tokio::spawn(embedding_loop(config.clone(), db.clone(), token.clone()));

    jetstream::run(&config, db, handler, token).await?;
    likes.await??;
    embeddings.await??;

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
