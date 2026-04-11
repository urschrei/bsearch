mod config;
mod db;
mod embed;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;

#[derive(Parser)]
#[command(
    name = "bsearch-search",
    version,
    about = "Fast search across indexed Bluesky posts"
)]
struct Cli {
    /// Search query text
    query: Option<String>,

    /// Number of results
    #[arg(short = 'n', long, default_value_t = 10)]
    limit: usize,

    /// Filter by source type
    #[arg(short, long, value_parser = ["own_post", "like", "backfill_post", "backfill_like"])]
    source: Option<String>,

    /// Search mode
    #[arg(short, long, default_value = "hybrid", value_parser = ["hybrid", "keyword", "semantic"])]
    mode: String,

    /// Filter by author handle
    #[arg(short = 'a', long)]
    handle: Option<String>,

    /// Database path
    #[arg(long, env = "BSEARCH_DB_PATH")]
    db: Option<PathBuf>,

    /// Model directory (containing model.onnx and tokenizer.json)
    #[arg(long, env = "BSEARCH_MODEL_DIR")]
    model: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::Config::resolve(cli.db, cli.model)?;

    if cli.query.is_none() && cli.handle.is_none() {
        anyhow::bail!("Provide a query and/or --handle to search.");
    }

    let database = db::Database::open(&config.db_path)
        .with_context(|| format!("Failed to open database at {}", config.db_path.display()))?;

    // No query: list posts by handle
    if cli.query.is_none() {
        let handle = cli.handle.as_deref().unwrap();
        let results = database.list_by_handle(handle, cli.limit, cli.source.as_deref())?;
        if results.is_empty() {
            eprintln!("No results found.");
            return Ok(());
        }
        for (i, r) in results.iter().enumerate() {
            print_result(i + 1, r);
        }
        return Ok(());
    }

    let query = cli.query.as_deref().unwrap();

    // Load embedder only for hybrid/semantic modes
    let query_embedding = if cli.mode == "keyword" {
        None
    } else {
        let mut embedder =
            embed::Embedder::load(&config.model_dir).context("Failed to load embedding model")?;
        Some(embedder.encode(query).context("Failed to encode query")?)
    };

    let results = match cli.mode.as_str() {
        "keyword" => database.search_fts(
            query,
            cli.limit,
            cli.source.as_deref(),
            cli.handle.as_deref(),
        )?,
        "semantic" => {
            let emb = query_embedding.as_ref().unwrap();
            database.search_vec(emb, cli.limit, cli.source.as_deref(), cli.handle.as_deref())?
        }
        _ => database.search_hybrid(
            query,
            query_embedding.as_ref(),
            cli.limit,
            cli.source.as_deref(),
            cli.handle.as_deref(),
        )?,
    };

    if results.is_empty() {
        eprintln!("No results found.");
        return Ok(());
    }

    for (i, r) in results.iter().enumerate() {
        print_result(i + 1, r);
    }

    Ok(())
}

fn at_uri_to_web_url(uri: &str) -> String {
    if let Some(rest) = uri.strip_prefix("at://") {
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() >= 3 && parts[1] == "app.bsky.feed.post" {
            return format!("https://bsky.app/profile/{}/post/{}", parts[0], parts[2]);
        }
    }
    uri.to_string()
}

fn print_result(index: usize, r: &db::SearchResult) {
    let web_url = at_uri_to_web_url(&r.uri);

    let score_info = if let Some(rrf) = r.rrf_score {
        let mt = r.match_type.as_deref().unwrap_or("");
        format!("score: {rrf:.4}, match: {mt}")
    } else if let Some(bm25) = r.bm25_rank {
        format!("bm25: {bm25:.4}")
    } else if let Some(dist) = r.distance {
        format!("distance: {dist:.4}")
    } else {
        String::new()
    };

    println!(
        "\n{} {} {} {}",
        "---".bold(),
        format!("Result {index}").white().bold(),
        format!("({score_info})").yellow(),
        "---".bold(),
    );
    println!("{}  {}", "Author:".dimmed(), r.author_handle.cyan());
    println!("{}  {}", "Date:".dimmed(), r.created_at);
    println!("{}  {}", "Source:".dimmed(), r.source.magenta());
    println!("{}  {}", "Link:".dimmed(), web_url.blue().underline());
    println!("{}  {}", "Text:".dimmed(), r.text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at_uri_to_web_url() {
        assert_eq!(
            at_uri_to_web_url("at://did:plc:abc/app.bsky.feed.post/rkey123"),
            "https://bsky.app/profile/did:plc:abc/post/rkey123"
        );
    }

    #[test]
    fn test_at_uri_to_web_url_passthrough() {
        assert_eq!(
            at_uri_to_web_url("https://example.com"),
            "https://example.com"
        );
    }
}
