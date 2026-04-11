mod config;
mod db;
mod embed;

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use colored::Colorize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SearchMode {
    Hybrid,
    Keyword,
    Semantic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SourceType {
    OwnPost,
    Like,
    BackfillPost,
    BackfillLike,
}

impl SourceType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::OwnPost => "own_post",
            Self::Like => "like",
            Self::BackfillPost => "backfill_post",
            Self::BackfillLike => "backfill_like",
        }
    }
}

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
    #[arg(short, long)]
    source: Option<SourceType>,

    /// Search mode
    #[arg(short, long, default_value = "hybrid")]
    mode: SearchMode,

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

    let source_str = cli.source.map(|s| s.as_str());
    let database = db::Database::open(&config.db_path)
        .with_context(|| format!("Failed to open database at {}", config.db_path.display()))?;

    // No query: list posts by handle
    if cli.query.is_none() {
        let handle = cli.handle.as_deref().unwrap();
        let results = database.list_by_handle(handle, cli.limit, source_str)?;
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
    let query_embedding = match cli.mode {
        SearchMode::Keyword => None,
        SearchMode::Hybrid | SearchMode::Semantic => {
            let mut embedder = embed::Embedder::load(&config.model_dir)
                .context("Failed to load embedding model")?;
            Some(embedder.encode(query).context("Failed to encode query")?)
        }
    };

    let results = match cli.mode {
        SearchMode::Keyword => {
            database.search_fts(query, cli.limit, source_str, cli.handle.as_deref())?
        }
        SearchMode::Semantic => {
            let emb = query_embedding.as_ref().unwrap();
            database.search_vec(emb, cli.limit, source_str, cli.handle.as_deref())?
        }
        SearchMode::Hybrid => database.search_hybrid(
            query,
            query_embedding.as_ref(),
            cli.limit,
            source_str,
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

fn print_result(index: usize, r: &db::SearchResult) {
    let web_url = at_uri_to_web_url(&r.uri);

    let score_info = match (r.rrf_score, r.bm25_rank, r.distance, r.match_type) {
        (Some(rrf), _, _, Some(mt)) => format!("score: {rrf:.4}, match: {mt}"),
        (_, Some(bm25), _, _) => format!("bm25: {bm25:.4}"),
        (_, _, Some(dist), _) => format!("distance: {dist:.4}"),
        _ => String::new(),
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

fn at_uri_to_web_url(uri: &str) -> String {
    if let Some(rest) = uri.strip_prefix("at://") {
        let mut parts = rest.splitn(3, '/');
        let did = parts.next();
        let collection = parts.next();
        let rkey = parts.next();
        if let (Some(did), Some("app.bsky.feed.post"), Some(rkey)) = (did, collection, rkey) {
            return format!("https://bsky.app/profile/{did}/post/{rkey}");
        }
    }
    uri.to_string()
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
