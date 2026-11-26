# Commands

bsearch has three entry points: the Python CLI for setup and maintenance,
the `bsearch-search` binary, and the `bsearch-serve` daemon.

## Python CLI

| Command | Description |
|---|---|
| `uv run bsearch init` | Verify credentials and resolve your DID |
| `uv run bsearch backfill [-n LIMIT]` | Fetch historical posts and likes via the AT Protocol API (superseded by the daemon's archive replay when a `jetstream_key` is configured) |
| `uv run bsearch serve` | Run the Jetstream listener in the foreground (superseded by `bsearch-serve`) |
| `uv run bsearch status` | Show database statistics and cursor position |
| `uv run bsearch reindex [--batch-size N]` | Clear all embeddings and regenerate from scratch |
| `uv run bsearch vacuum` | Reclaim unused database space |
| `uv run bsearch export-model` | Export the embedding model to ONNX format |
| `uv run bsearch install-service` | Install and start launchd agents for background operation and log rotation |
| `uv run bsearch uninstall-service` | Stop and remove the launchd agents |

Use `-v` before any subcommand for verbose logging, for example
`uv run bsearch -v backfill`.

Stop the daemon before running `bsearch backfill` or `bsearch reindex`. See
[Running the daemon](running-the-daemon.md#running-maintenance-commands).

## Search binary

```
bsearch-search [OPTIONS] [QUERY]
```

| Option | Description |
|---|---|
| `-n`, `--limit` | Number of results (default 10) |
| `-s`, `--source` | Filter by source: `own_post`, `like`, `backfill_post`, `backfill_like` |
| `-m`, `--mode` | Search mode: `hybrid` (default), `keyword`, `semantic` |
| `-a`, `--handle` | Filter by author handle; with no query, lists posts from that handle |
| `-d`, `--newest-first` | Return the newest matching posts, most recent first |
| `--db` | Database path (default: `./bsearch.db`, env: `BSEARCH_DB_PATH`) |
| `--model` | Model directory (default: `~/.cache/bsearch/all-MiniLM-L6-v2`, env: `BSEARCH_MODEL_DIR`) |
| `--max-semantic-distance` | L2 distance threshold for semantic-only results in hybrid mode (default: 1.05) |

Keyword search matches the post text and the text of its images: their alt
text, and the text recognised in them. Results show the image text under
the post text.

`--newest-first` replaces the relevance ranking rather than breaking ties
within it, so it changes which posts come back, not just their order: with
`-n 10` you get the ten newest matches, and a recent weak match can displace an
older strong one. Keyword search orders across every matching post. Vector
search is k-nearest-neighbour and the hybrid merge is built on rank positions,
so neither has a full match set to order; both instead draw from a pool of the
best matches twenty times the requested limit. Listing a handle with no query
is already newest-first, so the flag makes no difference there.

See [the search crate's README](../crates/bsearch-search/README.md) for
details on the embedding pipeline.

## Daemon

```
bsearch-serve [COMMAND]
```

With no command, the daemon follows the account's Jetstream and indexes
posts and likes until stopped. It reads `.env` from the working directory;
see [Configuration](configuration.md).

| Command | Description |
|---|---|
| `backfill-images [--since YYYY-MM-DD] [--dry-run]` | Record the images in already-indexed posts, for text recognition. All posts when `--since` is omitted |
| `backfill-links --since YYYY-MM-DD [--dry-run]` | Queue the links in already-indexed liked posts for Instapaper |

Both commands fetch the posts again and leave their results in a queue that
the running daemon drains. With `--dry-run` they fetch and count without
writing anything.
