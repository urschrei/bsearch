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

## Distance metric and semantic filtering

`sqlite-vec`'s `vec0` virtual table uses L2 (Euclidean) distance. Because all
embeddings are L2-normalised to unit length, L2 distance is equivalent to
cosine distance: `L2 = sqrt(2 - 2*cos(theta))`. The ranking is identical.

In hybrid mode, results from FTS and vector search are combined using
Reciprocal Rank Fusion (RRF). Semantic-only results (those without a keyword
match) are only included if their L2 distance is below `--max-semantic-distance`
(default: 1.05). This prevents short or vague queries from being padded with
irrelevant semantic matches.

### Why not dot product?

Dot product similarity preserves vector magnitude, which in theory encodes how
much semantic content the model found in a text. Longer, more specific posts
produce higher-magnitude vectors and would score higher. This would naturally
penalise vague single-word posts that produce short vectors.

We attempted to implement dot product as an alternative distance metric. The
approach was:

1. Store unnormalised embeddings (stripping the Normalize module from the
   SentenceTransformer pipeline).
2. Use `sqlite-vec` L2 KNN as a candidate generator, then re-rank candidates
   by dot product in application code.

This failed because **L2 KNN on unnormalised vectors does not produce useful
candidates for dot product re-ranking**. The L2 distance between a query and
stored vectors is dominated by the `||v||^2` magnitude term, so `sqlite-vec`
returns vectors with similar magnitude rather than vectors pointing in a similar
direction. Normalising the query for the KNN lookup does not help: the stored
vectors still have varying magnitudes, so the distance is
`sqrt(1 + ||v||^2 - 2*q_norm . v)`, still magnitude-dominated.

Possible workarounds that were not pursued:

- **Store magnitude separately**: keep normalised embeddings in `vec_posts` for
  KNN, store the original magnitude as a column. Reconstruct dot product as
  `cos_similarity * ||query|| * ||stored||` at scoring time. Clean, but adds
  complexity for a marginal gain over the distance threshold approach.
- **Two vec tables**: one normalised (for KNN), one unnormalised (for readback
  during dot product scoring). Correct but doubles embedding storage.

The cosine distance approach with the `--max-semantic-distance` threshold solves
the original problem (irrelevant results for short queries) without these
trade-offs.
