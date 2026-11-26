//! One-off recording of the images in posts that were indexed before
//! images were recorded.
//!
//! The database holds a post's text but not its embed, so the records are
//! fetched again and put through the same extraction as a live post. The
//! images go into `post_images`, which the running daemon drains as usual;
//! no image is fetched or read from here.

use anyhow::Context;
use anyhow::Result;
use bsearch_core::db::Database;
use jiff::civil::Date;

use crate::config::Config;
use crate::resolver::Resolver;

/// Posts per progress report; the resolver splits this into API calls.
const FETCH_BATCH: usize = 100;

pub async fn run(config: &Config, since: Option<Date>, dry_run: bool) -> Result<()> {
    let db = Database::open_read_write(&config.db_path)
        .with_context(|| format!("Failed to open database at {}", config.db_path.display()))?;
    let since = since.map(|date| date.to_string()).unwrap_or_default();
    let uris = db.post_uris_since(&since)?;
    if since.is_empty() {
        println!("Indexed posts: {}", uris.len());
    } else {
        println!("Indexed posts created since {since}: {}", uris.len());
    }

    let resolver = Resolver::new(config);
    resolver.login(config).await?;

    let mut resolved = 0;
    let mut posts_with_images = 0;
    let mut images = 0;
    let mut updated = 0;
    for (i, chunk) in uris.chunks(FETCH_BATCH).enumerate() {
        let posts = resolver.resolve_post_uris(chunk).await?;
        resolved += posts.len();
        for post in posts.iter().map(|resolved| &resolved.post) {
            if post.images.is_empty() {
                continue;
            }
            posts_with_images += 1;
            images += post.images.len();
            if !dry_run && db.add_images_to_post(&post.uri, &post.images)? {
                updated += 1;
            }
        }
        eprintln!(
            "  fetched {}/{} posts, {images} images so far",
            (i * FETCH_BATCH + chunk.len()).min(uris.len()),
            uris.len(),
        );
    }

    println!(
        "Posts still available: {resolved} ({} deleted since)",
        uris.len() - resolved
    );
    println!("Posts carrying images: {posts_with_images}");
    println!("Images:                {images}");

    if dry_run {
        println!("\nDry run: nothing recorded.");
        return Ok(());
    }
    println!(
        "Recorded images for {updated} posts ({} already had them).",
        posts_with_images - updated
    );
    println!("The running daemon reads their text and embeds the posts again.");
    Ok(())
}
