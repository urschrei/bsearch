# How bsearch works

This page explains how the daemon collects posts, why it is written in Rust,
and how it handles likes, links and images.

## Overview

- Connects to the Bluesky Jetstream v2 live tail, filtered to a single account's DID
- With a Jetstream API key, gaps are closed from Jetstream's sealed archive over HTTP: on first start the daemon replays the account's full history, and after downtime it replays exactly the missed span before rejoining the live tail
- Own posts arrive directly in the event stream with full text
- Likes arrive as references only -- the liked post's text is resolved in batches via the AT Protocol API
- With Instapaper credentials configured, the links a liked post carries -- its link facets and link card -- are filed in an Instapaper folder
- Images in posts are indexed by their alt text at once, and by the text that the macOS Vision framework recognises in them shortly after
- Post text is embedded with `all-MiniLM-L6-v2` (384 dimensions) and stored in a `sqlite-vec` virtual table for KNN vector search
- Embeddings are generated periodically in the background, not blocking the event stream
- Both the background daemon (`bsearch-serve`) and search (`bsearch-search`) are Rust binaries running the model through ONNX Runtime, statically linked so they have no runtime dependencies
- Python remains for one-off commands: setup, model export and maintenance

## Memory use

The daemon is written in Rust because it is long-lived. The equivalent Python
service settled at a 2.5 GB physical footprint -- mostly PyTorch plus the
Metal/MPS allocations `sentence-transformers` makes on Apple Silicon -- to
embed a handful of short posts every ten seconds. Measured footprints on the
same machine and model:

| | Physical footprint |
|---|---|
| Python daemon (steady state, MPS) | 2586 MB |
| Rust daemon, idle | 9-20 MB |
| Rust daemon, embedding a batch | ~70 MB |
| Rust daemon, embedding while image text is read | ~110 MB |
| Text recognition process, one image | 30-80 MB |

The ONNX session is loaded only when there are posts to embed and dropped
again after five minutes idle, which is what keeps the idle figure low;
reloading takes well under a second. Text in images is read by a separate
process for each image, which exits when the image is done, so its memory
does not add to the daemon's idle figure. Its footprint depends on the
image: a photo with a few words needs about 30 MB, a dense screenshot about
80 MB.

## Jetstream transports

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

## Filing links in Instapaper

With Instapaper configured, each liked post the daemon indexes for the
first time has its links -- `app.bsky.richtext.facet#link` facets and
`app.bsky.embed.external` cards, not URLs scanned out of the text -- written
to a `pending_links` queue in the same transaction as the post. Links into
Bluesky itself (`bsky.app`) and to YouTube are left out, subdomains
included. A separate task drains that queue, submitting one link a second
in batches of ten, with the card's title if it has one and the post's
author and text as the bookmark description. Posts that were already
indexed, by an earlier run or by `bsearch backfill`, are not submitted. A
link is removed from the queue only once Instapaper has accepted it, so a
network failure or a restart leaves it to be sent later; resending is
harmless, as Instapaper treats a URL it already holds as an update. A link
Instapaper rejects outright -- an invalid URL, or a publisher that has
opted out -- is dropped with a warning in the log. Instapaper reports these
and a missing folder with the same status, so after a rejection the task
checks that the folder still exists; if it does not, the task finds or
creates it again and keeps the link. After any other failure the task waits
a minute before trying again, and five minutes after a rate-limit response.
Failing to connect to Instapaper at all raises one macOS notification and
is otherwise logged. `bsearch status` shows how many links are waiting.

## Text in images

Images embedded in a post (`app.bsky.embed.images`, directly or inside
`app.bsky.embed.recordWithMedia`) are written to a `post_images` queue in
the same transaction as the post. Posts with images and no text are indexed
too. The alt text is searchable at once. A separate task fetches each
queued image from the Bluesky CDN, ten per pass, recognises its text with
the Vision framework, and adds the text to the post's `image_text` column,
which full-text search reads together with the post text. A post with
images is embedded only after all of its images are done, from its text
followed by its image text. An image the CDN no longer has is marked as done
without text. Any other failure is retried on later passes, up to five
times. When the CDN cannot be reached, or the recognition process cannot
start, the task waits a minute without counting an attempt. Each image is
read by a short-lived child process, `bsearch-serve recognise-text`,
because Vision keeps about 140 MB of models in the process that uses it.
After each new build, the first image takes about 40 seconds while Vision
prepares its models.

To record the images in posts indexed before this feature existed, see
[Recording images in older posts](running-the-daemon.md#recording-images-in-older-posts).
