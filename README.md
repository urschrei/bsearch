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

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and a Rust toolchain.

### 1. Install Python dependencies

```
uv sync
```

### 2. Configure credentials

Create a `.env` file with your Bluesky credentials:

```
user=yourhandle.bsky.social
password=your-app-password
```

If your account is on a custom PDS (not `bsky.social`), also set:

```
pds_url=https://pds.example.com
```

### 3. Verify authentication

```
uv run bsearch init
```

### 4. Build the search binary

Export the embedding model to ONNX format, then build the Rust binary:

```
uv run bsearch export-model
cargo build -p bsearch-search --release
```

## Getting started

Once setup is complete, backfill your historical posts and likes:

```
uv run bsearch backfill
```

Then search:

```
ORT_DYLIB_PATH=.venv/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.1.24.4.dylib \
  ./target/release/bsearch-search "your query"
```

To avoid typing the `ORT_DYLIB_PATH` each time, add a shell alias (adjust paths if needed):

```sh
alias bss='ORT_DYLIB_PATH=/path/to/bsearch/.venv/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.1.24.4.dylib /path/to/bsearch/target/release/bsearch-search --db /path/to/bsearch/bsearch.db'
```

Then simply: `bss "your query"`.

To keep the database up to date continuously, run the Jetstream listener in the foreground or install it as a background service:

```
uv run bsearch serve
# or
uv run bsearch install-service
```

## Commands

### Python CLI

| Command | Description |
|---|---|
| `uv run bsearch init` | Verify credentials and resolve your DID |
| `uv run bsearch backfill [-n LIMIT]` | Fetch historical posts and likes, generate embeddings |
| `uv run bsearch serve` | Run the Jetstream listener in the foreground |
| `uv run bsearch status` | Show database statistics and cursor position |
| `uv run bsearch vacuum` | Reclaim unused database space |
| `uv run bsearch export-model` | Export the embedding model to ONNX format |
| `uv run bsearch install-service` | Install and start a launchd agent for background operation |
| `uv run bsearch uninstall-service` | Stop and remove the launchd agent |

Use `-v` before any subcommand for verbose logging, e.g. `uv run bsearch -v backfill`.

### Search binary

```
bsearch-search [OPTIONS] [QUERY]
```

| Option | Description |
|---|---|
| `-n`, `--limit` | Number of results (default 10) |
| `-s`, `--source` | Filter by source: `own_post`, `like`, `backfill_post`, `backfill_like` |
| `-m`, `--mode` | Search mode: `hybrid` (default), `keyword`, `semantic` |
| `-a`, `--handle` | Filter by author handle; with no query, lists posts from that handle |
| `--db` | Database path (default: `./bsearch.db`, env: `BSEARCH_DB_PATH`) |
| `--model` | Model directory (default: `~/.cache/bsearch/all-MiniLM-L6-v2`, env: `BSEARCH_MODEL_DIR`) |

The binary requires the ONNX Runtime shared library. Set `ORT_DYLIB_PATH` to point to it -- the one bundled with the Python `onnxruntime` package (installed as a dev dependency) works. See `crates/bsearch-search/README.md` for details on the embedding pipeline.

## Service logs

When running as a launchd agent, logs are written to:

- `~/Library/Logs/bsearch/stdout.log`
- `~/Library/Logs/bsearch/stderr.log`

## Storage

The database is stored as `bsearch.db` in the working directory (configurable via `BSEARCH_DB_PATH` in `.env`).

# Licence
[The Blue Oak Model Licence 1.0](LICENSE.md)
