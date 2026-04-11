# Rust Search Binary Design

## Problem

The `bsearch search` command takes ~10 seconds on cold start. Nearly all of that time is spent importing PyTorch (~2-3s), importing sentence-transformers (~1s), and deserialising model weights (~3-5s). The actual query encoding and database search take <10ms. This makes interactive search painful.

## Solution

A standalone Rust binary (`bsearch-search`) that handles the search command using ONNX Runtime for inference instead of PyTorch. Target cold start: under 1 second.

The Python CLI remains unchanged for `serve`, `backfill`, `init`, `status`, `vacuum`, and `install-service` -- commands where startup time is not critical or the model stays loaded in memory.

## Architecture

Two components:

### bsearch-search (new Rust binary)

Located at `crates/bsearch-search/`. A standalone CLI binary with no Python dependency at runtime.

- **Config**: reads `.env` from current directory, env vars, and CLI flags (flags override env vars override `.env`)
- **Tokenizer**: HuggingFace `tokenizers` crate, loads `tokenizer.json` from model directory
- **Model**: `ort` crate (ONNX Runtime bindings), loads `model.onnx` from model directory
- **Database**: `rusqlite` with sqlite-vec extension, opens `bsearch.db` read-only
- **Search**: FTS5 keyword, KNN vector, and hybrid (RRF) modes
- **Output**: formatted text to stdout, matching the Python output format

### bsearch export-model (new Python CLI command)

Exports the SentenceTransformer model to ONNX format and copies the tokenizer, saving both to `~/.cache/bsearch/<model-name>/`.

This is the only Python-side change. It must be run once before the Rust binary can be used.

## CLI Interface

```
bsearch-search [OPTIONS] [QUERY]

Arguments:
  [QUERY]                    Search query text

Options:
  -n, --limit <N>            Number of results [default: 10]
  -s, --source <SOURCE>      Filter by source (own_post, like, backfill_post, backfill_like)
  -m, --mode <MODE>          Search mode: hybrid, keyword, semantic [default: hybrid]
  -a, --handle <HANDLE>      Filter by author handle; with no query, lists posts from this handle
      --db <PATH>            Database path [env: BSEARCH_DB_PATH] [default: ./bsearch.db]
      --model <DIR>          Model directory [env: BSEARCH_MODEL_DIR] [default: ~/.cache/bsearch/all-MiniLM-L6-v2]
  -h, --help
  -V, --version
```

Behaviours:
- `--handle` with no query lists posts by that handle (no model loaded)
- `--mode keyword` skips model loading entirely (FTS only)
- Model is only loaded for `hybrid` and `semantic` modes

## Config Resolution

Priority order (highest first):
1. CLI flags (`--db`, `--model`)
2. Environment variables (`BSEARCH_DB_PATH`, `BSEARCH_MODEL_DIR`)
3. `.env` file in current directory (`db_path` key)
4. Defaults (`./bsearch.db`, `~/.cache/bsearch/all-MiniLM-L6-v2`)

The Rust binary does not need Bluesky credentials -- it only reads the database.

## Data Flow

### Keyword mode (no model)

1. Open SQLite database with sqlite-vec extension
2. Run FTS5 `MATCH` query with BM25 ranking
3. On FTS syntax error, retry as phrase query (`"query"`)
4. Apply source/handle filters in SQL
5. Print results

### Semantic mode

1. Load `tokenizer.json` and `model.onnx` from model directory
2. Tokenize query text
3. Run ONNX inference to get per-token embeddings
4. Mean-pool token embeddings (masking padding tokens) and L2-normalise the result vector (see Embedding Pipeline below)
5. Query `vec_posts` with `embedding MATCH ?` and `k = limit` (over-fetch 5x when source/handle filter is active)
6. Post-filter results by source/handle in Rust
7. Print results

### Hybrid mode

1. Run both keyword and semantic paths
2. Reciprocal Rank Fusion with k=60
3. Annotate each result's `match_type` as `keyword`, `semantic`, or `keyword+semantic`
4. Print results sorted by RRF score

## Embedding Pipeline

The SentenceTransformer model consists of a BERT transformer followed by a mean-pooling layer and L2 normalisation. When exported to ONNX, only the transformer is included. The pooling and normalisation must be replicated in Rust:

1. **Tokenize**: use the `tokenizers` crate with the model's `tokenizer.json` to produce `input_ids`, `attention_mask`, and `token_type_ids`
2. **ONNX inference**: feed the three tensors to the model, receive `last_hidden_state` of shape `(1, seq_len, 384)`
3. **Mean pooling**: multiply each token embedding by its attention mask value (0 or 1), sum along the sequence dimension, divide by the sum of the attention mask. This averages only non-padding token embeddings.
4. **L2 normalisation**: divide the pooled vector by its L2 norm to produce a unit vector

This produces an embedding identical to `SentenceTransformer.encode()` within float32 tolerance.

## Database Access

The Rust binary opens the database **read-only**. It uses the same schema as the Python side:

- `posts` table: post metadata (uri, cid, author, text, dates, source)
- `vec_posts` virtual table: sqlite-vec KNN index, `float[384]` embeddings
- `fts_posts` virtual table: FTS5 full-text index with porter stemmer

sqlite-vec must be loaded as an extension. The binary will look for the shared library in standard locations and accept an override via environment variable if needed. The exact loading mechanism is an implementation detail to be resolved during development.

## Python-Side Changes

A single new CLI command:

```
bsearch export-model [--output-dir DIR]
```

1. Load the SentenceTransformer model (using `config.embedding_model`)
2. Export to ONNX format
3. Copy `tokenizer.json` from the model's HuggingFace cache
4. Save both to the output directory (default: `~/.cache/bsearch/all-MiniLM-L6-v2/`)
5. Print the output path

## Crate Dependencies

```toml
[dependencies]
clap = { version = "4", features = ["derive", "env"] }
ort = { version = "2", features = ["load-dynamic"] }
tokenizers = "0.21"
rusqlite = { version = "0.34", features = ["bundled"] }
ndarray = "0.16"
dotenvy = "0.15"
dirs = "6"
```

- `rusqlite` bundles SQLite from source (no system dependency)
- `ort` with `load-dynamic` loads ONNX Runtime at runtime
- `ndarray` for mean-pooling and L2 normalisation arithmetic

## Testing

- **Unit tests**: config resolution, RRF scoring, output formatting
- **Integration test**: export model from Python, run Rust binary against a test database with known posts, verify output matches Python
- **Numerical parity**: encode the same query in both Python and Rust, compare embedding vectors (must match within float32 tolerance)

## Deliverables

- `Cargo.toml` at repo root (workspace)
- `crates/bsearch-search/` with the Rust binary
- `crates/bsearch-search/README.md` covering the embedding pipeline, model export, and relationship to the Python side
- New `export-model` command in Python `cli.py`
- Tests for both sides
