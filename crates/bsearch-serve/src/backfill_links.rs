//! One-off queueing of the links in liked posts that were indexed before
//! Instapaper filing existed.
//!
//! The database holds a post's text but not its facets or link card, so the
//! records are fetched again and put through the same extraction as a live
//! like. The result goes into `pending_links`, which the running daemon
//! drains as usual; nothing is sent to Instapaper from here.

use std::collections::HashSet;

use anyhow::Context;
use anyhow::Result;
use bsearch_core::db::Database;
use bsearch_core::models::PendingLink;
use jiff::civil::Date;

use crate::config::Config;
use crate::links::to_pending;
use crate::resolver::Resolver;

/// Posts per progress report; the resolver splits this into API calls.
const FETCH_BATCH: usize = 100;

pub async fn run(config: &Config, since: Date, dry_run: bool) -> Result<()> {
    anyhow::ensure!(
        dry_run || config.instapaper.is_some(),
        "Instapaper is not configured in .env, so nothing would drain the queue"
    );

    let db = Database::open_read_write(&config.db_path)
        .with_context(|| format!("Failed to open database at {}", config.db_path.display()))?;
    let uris = db.liked_post_uris_since(&since.to_string())?;
    println!("Liked posts created since {since}: {}", uris.len());

    let resolver = Resolver::new(config);
    resolver.login(config).await?;

    let mut resolved = 0;
    let mut posts_with_links = 0;
    let mut links: Vec<PendingLink> = Vec::new();
    for (i, chunk) in uris.chunks(FETCH_BATCH).enumerate() {
        let posts = resolver.resolve_post_uris(chunk).await?;
        resolved += posts.len();
        for post in &posts {
            if !post.links.is_empty() {
                posts_with_links += 1;
            }
            links.extend(post.links.iter().map(|link| to_pending(&post.post, link)));
        }
        eprintln!(
            "  fetched {}/{} posts, {} links so far",
            (i * FETCH_BATCH + chunk.len()).min(uris.len()),
            uris.len(),
            links.len()
        );
    }

    let mut seen = HashSet::new();
    let distinct: Vec<PendingLink> = links
        .into_iter()
        .filter(|link| seen.insert(link.url.clone()))
        .collect();

    println!(
        "Posts still available: {resolved} ({} deleted since)",
        uris.len() - resolved
    );
    println!("Posts carrying links:  {posts_with_links}");
    println!("Distinct links:        {}", distinct.len());
    println!(
        "  with a card title:   {}",
        distinct.iter().filter(|l| l.title.is_some()).count()
    );

    if dry_run {
        println!();
        for link in &distinct {
            match &link.title {
                Some(title) => println!("{}\t{title}", link.url),
                None => println!("{}", link.url),
            }
        }
        println!("\nDry run: nothing queued.");
        return Ok(());
    }

    let queued = db.queue_pending_links(&distinct)?;
    println!(
        "Queued {queued} links for Instapaper ({} were already queued).",
        distinct.len() - queued
    );
    Ok(())
}
