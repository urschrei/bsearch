# bsearch

A local tool that monitors a Bluesky account via [Jetstream](https://github.com/bluesky-social/jetstream) (WebSocket), capturing both your own posts and posts you've liked into a SQLite database with vector search. Designed to run as a background service on macOS.

## How it works

- Connects to the Bluesky Jetstream firehose, filtered to a single account's DID
- Own posts arrive directly in the event stream with full text
- Likes arrive as references only -- the liked post's text is resolved in batches via the AT Protocol API
- Post text is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions) and stored in a `sqlite-vec` virtual table for KNN vector search
- Embeddings are generated periodically in the background, not blocking the event stream
- Search is handled by a fast Rust binary (`bsearch-search`) using ONNX Runtime, with sub-second cold start

## Setup

Requires Python 3.13+ and Rust. Python dependencies are managed with [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Create a `.env` file with your Bluesky credentials:

```
user=yourhandle.bsky.social
password=your-app-password
```

If your account is on a custom PDS (not `bsky.social`), also set:

```
pds_url=https://pds.example.com
```

Verify authentication works:

```
uv run bsearch init
```

## Commands

| Command | Description |
|---|---|
| `bsearch init` | Verify credentials and resolve your DID |
| `bsearch backfill` | Fetch historical posts and likes via the API, generate embeddings |
| `bsearch serve` | Run the Jetstream listener in the foreground |
| `bsearch status` | Show database statistics and cursor position |
| `bsearch export-model` | Export the embedding model to ONNX format for the search binary |
| `bsearch install-service` | Install and start a launchd agent for background operation |
| `bsearch uninstall-service` | Stop and remove the launchd agent |

## Search

Search is handled by a standalone Rust binary for fast startup. One-time setup:

```
uv run bsearch export-model
cargo build -p bsearch-search --release
```

Then search with:

```
bsearch-search "query" [-n LIMIT] [-s SOURCE] [-m MODE] [-a HANDLE]
```

- `-n` / `--limit`: number of results (default 10)
- `-s` / `--source`: filter by source type (`own_post`, `like`, `backfill_post`, `backfill_like`)
- `-m` / `--mode`: search mode -- `hybrid` (default), `keyword` (FTS only), or `semantic` (vector only)
- `-a` / `--handle`: filter by author handle; with no query, lists all posts from that handle

The binary requires the ONNX Runtime shared library at runtime. Set `ORT_DYLIB_PATH` to point to it (the one bundled with the Python `onnxruntime` package works).

See `crates/bsearch-search/README.md` for more details on the embedding pipeline.

### Backfill options

```
bsearch backfill [-n LIMIT]
```

- `-n` / `--limit`: maximum number of posts to fetch per category

Use `-v` before any subcommand for verbose logging, e.g. `bsearch -v backfill`.

## Service logs

When running as a launchd agent, logs are written to:

- `~/Library/Logs/bsearch/stdout.log`
- `~/Library/Logs/bsearch/stderr.log`

## Storage

The database is stored as `bsearch.db` in the working directory (configurable via `BSEARCH_DB_PATH` in `.env`).

# License
[The Blue Oak Model License 1.0](LICENSE.md)
