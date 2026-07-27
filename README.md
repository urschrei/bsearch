# bsearch

A local tool that monitors a Bluesky account via [Jetstream](https://github.com/bluesky-social/jetstream) (WebSocket), capturing both your own posts and posts you've liked into a SQLite database with vector search. Designed to run as a background service on macOS.

## How it works

- Connects to the Bluesky Jetstream firehose, filtered to a single account's DID
- Own posts arrive directly in the event stream with full text
- Likes arrive as references only -- the liked post's text is resolved in batches via the AT Protocol API
- Post text is embedded with `all-MiniLM-L6-v2` (384 dimensions) and stored in a `sqlite-vec` virtual table for KNN vector search
- Embeddings are generated periodically in the background, not blocking the event stream
- Both the background daemon (`bsearch-serve`) and search (`bsearch-search`) are Rust binaries running the model through ONNX Runtime, statically linked so they have no runtime dependencies
- Python remains for one-off commands: setup, backfill, model export and maintenance

The daemon is written in Rust because it is long-lived. The equivalent Python
service settled at a 2.5 GB physical footprint -- mostly PyTorch plus the
Metal/MPS allocations `sentence-transformers` makes on Apple Silicon -- to
embed a handful of short posts every ten seconds. Measured footprints on the
same machine and model:

| | Physical footprint |
|---|---|
| Python daemon (steady state, MPS) | 2586 MB |
| Rust daemon, idle | 9-19 MB |
| Rust daemon, embedding a batch | ~70 MB |

The ONNX session is loaded only when there are posts to embed and dropped
again after five minutes idle, which is what keeps the idle figure low;
reloading takes well under a second.

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

### 4. Build the Rust binaries

Export the embedding model to ONNX format, then build:

```
uv run bsearch export-model
cargo build --release
```

This produces `target/release/bsearch-search` and `target/release/bsearch-serve`.
ONNX Runtime is linked statically, so neither needs anything else installed at
runtime.

## Getting started

Once setup is complete, backfill your historical posts and likes:

```
uv run bsearch backfill
```

Then search:

```
./target/release/bsearch-search "your query"
```

A shell alias saves typing the paths:

```sh
alias bss='/path/to/bsearch/target/release/bsearch-search --db /path/to/bsearch/bsearch.db'
```

Then simply: `bss "your query"`.

To keep the database up to date continuously, run the daemon in the foreground
or install it as a background service:

```
./target/release/bsearch-serve
# or
uv run bsearch install-service
```

`install-service` writes a launchd agent pointing at
`target/release/bsearch-serve`, so build it first.

## Commands

### Python CLI

| Command | Description |
|---|---|
| `uv run bsearch init` | Verify credentials and resolve your DID |
| `uv run bsearch backfill [-n LIMIT]` | Fetch historical posts and likes, generate embeddings |
| `uv run bsearch serve` | Run the Jetstream listener in the foreground (superseded by `bsearch-serve`) |
| `uv run bsearch status` | Show database statistics and cursor position |
| `uv run bsearch reindex [--batch-size N]` | Clear all embeddings and regenerate from scratch |
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
| `-d`, `--newest-first` | Return the newest matching posts, most recent first |
| `--db` | Database path (default: `./bsearch.db`, env: `BSEARCH_DB_PATH`) |
| `--model` | Model directory (default: `~/.cache/bsearch/all-MiniLM-L6-v2`, env: `BSEARCH_MODEL_DIR`) |
| `--max-semantic-distance` | L2 distance threshold for semantic-only results in hybrid mode (default: 1.05) |

`--newest-first` replaces the relevance ranking rather than breaking ties
within it, so it changes which posts come back, not just their order: with
`-n 10` you get the ten newest matches, and a recent weak match can displace an
older strong one. Keyword search orders across every matching post. Vector
search is k-nearest-neighbour and the hybrid merge is built on rank positions,
so neither has a full match set to order; both instead draw from a pool of the
best matches twenty times the requested limit. Listing a handle with no query
is already newest-first, so the flag makes no difference there.

See `crates/bsearch-search/README.md` for details on the embedding pipeline.

### Daemon

```
bsearch-serve
```

Takes no arguments; it reads `.env` from the working directory, using the same
keys as the Python CLI. `BSEARCH_DB_PATH` and `BSEARCH_MODEL_DIR` override the
database and model locations, and `RUST_LOG` controls logging (default
`info,ort=warn`).

## Service logs

When running as a launchd agent, logs are written to:

- `~/Library/Logs/bsearch/stdout.log`
- `~/Library/Logs/bsearch/stderr.log`

## Storage

The database is stored as `bsearch.db` in the working directory (configurable via `BSEARCH_DB_PATH` in `.env`).

# Licence
[The Blue Oak Model Licence 1.0](LICENSE.md)
