# bsearch

A local tool that monitors a Bluesky account via [Jetstream](https://github.com/bluesky-social/jetstream), capturing both your own posts and posts you've liked into a SQLite database with vector search. Designed to run as a background service on macOS.

## How it works

- Connects to the Bluesky Jetstream v2 live tail, filtered to a single account's DID
- With a Jetstream API key, gaps are closed from Jetstream's sealed archive over HTTP: on first start the daemon replays the account's full history, and after downtime it replays exactly the missed span before rejoining the live tail
- Own posts arrive directly in the event stream with full text
- Likes arrive as references only -- the liked post's text is resolved in batches via the AT Protocol API
- Post text is embedded with `all-MiniLM-L6-v2` (384 dimensions) and stored in a `sqlite-vec` virtual table for KNN vector search
- Embeddings are generated periodically in the background, not blocking the event stream
- Both the background daemon (`bsearch-serve`) and search (`bsearch-search`) are Rust binaries running the model through ONNX Runtime, statically linked so they have no runtime dependencies
- Python remains for one-off commands: setup, model export and maintenance

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

To let the daemon replay history from Jetstream's sealed archive -- the
initial sync, and recovery after downtime -- add an archive API key:

```
jetstream_key=gk_your-jetstream-api-key
```

The key is used only on the archive HTTP endpoints; the live WebSocket is
unauthenticated. Without a key the daemon still follows the live tail, but
events that fall outside the live replay buffer while it is offline are
lost.

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

With a `jetstream_key` configured, the daemon's first run replays your full
history -- posts and likes -- from the Jetstream archive; no separate
backfill step is needed. Without a key, `uv run bsearch backfill` fetches
historical posts and likes through the AT Protocol API instead.

Search with:

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
| `uv run bsearch backfill [-n LIMIT]` | Fetch historical posts and likes via the AT Protocol API (superseded by the daemon's archive replay when a `jetstream_key` is configured) |
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

The daemon speaks Jetstream v2 on two transports. The live tail is the
`subscribeEvents` WebSocket, resumed from a stored sequence number. Catching
up happens over HTTP instead: with a `jetstream_key`, the daemon plans a
snapshot of the sealed archive, downloads only the blocks that can contain
the account's events, and feeds them through the same ingestion path before
rejoining the live tail at the sealed tip. This runs at startup and after
any abnormal disconnect, so downtime of any length is recovered without
intervention. A database created by an earlier version, whose cursor is a
`time_us` timestamp rather than a sequence number, is migrated by one
full-history sweep the first time the daemon starts with a key.

The daemon still remakes its live connection periodically, so the log shows
reconnects even when nothing is wrong. This is deliberate: a WebSocket that
dies without a close frame leaves the reader waiting on a read that never
returns, and because the subscription is filtered to a single DID, no
traffic is expected anyway -- so a stalled connection is otherwise
indistinguishable from an idle account. Capping the lifetime bounds how
long ingestion can stall. The v2 cursor is inclusive and post inserts
ignore duplicate URIs, so the event replayed at each boundary is harmless.

If the live server rejects the cursor as older than its retention floor,
the daemon re-enters archive catch-up and reconnects at the new tip; with
no key configured it notifies, forgets the cursor, and resumes from the
live tip, since nothing can serve the gap.

Stop the daemon before running `bsearch backfill` or `bsearch reindex`.
They generate embeddings for posts with `has_embedding = 0`, and run
together with the daemon they collide on `vec_posts`, failing whichever
writes second.

## Service logs

When running as a launchd agent, logs are written to:

- `~/Library/Logs/bsearch/stdout.log`
- `~/Library/Logs/bsearch/stderr.log`

## Storage

The database is stored as `bsearch.db` in the working directory (configurable via `BSEARCH_DB_PATH` in `.env`).

Whole-segment archive downloads spool to `~/.cache/bsearch/segments` so an
interrupted transfer resumes from its last byte; completed files are
deleted after their events are ingested. The directory is normally empty:
plans filtered to one account almost always name individual blocks, which
are fetched in memory.

# Licence
[The Blue Oak Model Licence 1.0](LICENSE.md)
