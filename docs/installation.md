# Installing bsearch

This guide sets up bsearch, loads your history, and keeps the database up
to date.

## Prerequisites

- macOS
- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- A Rust toolchain

## Installing

1. Install the Python dependencies:

   ```
   uv sync
   ```

2. Create a `.env` file with your Bluesky credentials:

   ```
   user=yourhandle.bsky.social
   password=your-app-password
   ```

   For a custom PDS, the Jetstream archive key and Instapaper filing, see
   [Configuration](configuration.md).

3. Verify authentication:

   ```
   uv run bsearch init
   ```

4. Export the embedding model to ONNX format, then build the binaries:

   ```
   uv run bsearch export-model
   cargo build --release
   ```

   This produces `target/release/bsearch-search` and
   `target/release/bsearch-serve`. ONNX Runtime is linked statically, so
   neither needs anything else installed at runtime.

## Loading your history

With a `jetstream_key` configured, the daemon's first run replays your full
history -- posts and likes -- from the Jetstream archive; no separate
backfill step is needed. Without a key, `uv run bsearch backfill` fetches
historical posts and likes through the AT Protocol API instead.

## Searching

```
./target/release/bsearch-search "your query"
```

A shell alias saves typing the paths:

```sh
alias bss='/path/to/bsearch/target/release/bsearch-search --db /path/to/bsearch/bsearch.db'
```

Then run `bss "your query"`. For all search options, see
[Commands](commands.md#search-binary).

## Keeping the database up to date

Run the daemon in the foreground, or install it as a background service:

```
./target/release/bsearch-serve
# or
uv run bsearch install-service
```

See [Running the daemon](running-the-daemon.md) for logs and maintenance.
