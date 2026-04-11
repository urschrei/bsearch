# bsearch-search

A fast Rust binary for searching indexed Bluesky posts. Replaces the Python
`bsearch search` command with sub-second cold start by using ONNX Runtime
instead of PyTorch for embedding inference.

## Setup

1. Export the model from the Python side (one-time):

   ```
   uv run bsearch export-model
   ```

   This saves `model.onnx` and `tokenizer.json` to
   `~/.cache/bsearch/all-MiniLM-L6-v2/`.

2. Build the binary:

   ```
   cargo build -p bsearch-search --release
   ```

3. Search:

   ```
   ./target/release/bsearch-search "your query"
   ```

## Relationship to the Python tool

The Python `bsearch` CLI handles monitoring (serve), backfill, and service
management. This Rust binary only handles search. Both read from the same
SQLite database.

## Embedding pipeline

The SentenceTransformer model (`all-MiniLM-L6-v2`) consists of three stages:

1. **Tokenisation** -- text to token IDs via the WordPiece tokenizer
2. **Transformer** -- token IDs to contextual embeddings (6-layer BERT,
   output shape: seq_len x 384)
3. **Mean pooling + L2 normalisation** -- aggregate token embeddings into a
   single 384-dimensional unit vector

When exported to ONNX, only stage 2 (the transformer) is included in
`model.onnx`. Stages 1 and 3 are handled in Rust:

- **Tokenisation**: the `tokenizers` crate loads `tokenizer.json` (the same
  Rust library that Python's `tokenizers` package wraps)
- **Mean pooling**: multiply each token embedding by its attention mask value,
  sum along the sequence axis, divide by the mask sum. This averages only
  non-padding tokens.
- **L2 normalisation**: divide by the vector's L2 norm to produce a unit
  vector suitable for cosine-distance search.

This produces embeddings identical to `SentenceTransformer.encode()` within
float32 tolerance.
