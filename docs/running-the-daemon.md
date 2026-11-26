# Running the daemon

This page covers running `bsearch-serve` in the foreground or as a launchd
service, reading its logs, and the maintenance tasks that involve it.

## Running in the foreground

```
./target/release/bsearch-serve
```

Stop it with Ctrl-C.

## Running as a background service

```
uv run bsearch install-service
```

`install-service` writes a launchd agent pointing at
`target/release/bsearch-serve`, so build it first. It also installs a
second agent that rotates the service logs daily; see
[Log rotation](#log-rotation). `uv run bsearch uninstall-service` removes
both.

## Service logs

When running as a launchd agent, logs are written to:

- `~/Library/Logs/bsearch/stdout.log`
- `~/Library/Logs/bsearch/stderr.log`

### Log rotation

launchd opens both files itself and hands the descriptors to the daemon,
which holds them for as long as it runs. This rules out rename-based
rotation (newsyslog, plain logrotate): the rotator would move the file
aside and create a fresh one, but the daemon would keep writing to the
renamed file until its next restart. The descriptors are append-mode,
though, so copy-then-truncate is safe -- the daemon's next write lands at
the start of the emptied file and nothing is lost, with no restart needed.

Rotation is `scripts/bsearch-logrotate`, run daily at 03:15 by the
`social.bsky.bsearch.logrotate` launchd agent that `install-service` sets
up alongside the daemon. Each run copies a non-empty log to a timestamped
file, truncates the original in place, gzips the copy, and keeps the
newest seven archives per log (`KEEP` in the script). launchd runs a
missed calendar job once on wake, so rotation still happens on a machine
that sleeps overnight. `uninstall-service` removes the rotation agent
together with the daemon's.

## Recording images in older posts

Posts indexed before images were recorded have no queued images. To record
them, run:

```
./target/release/bsearch-serve backfill-images [--since YYYY-MM-DD] [--dry-run]
```

This fetches the posts again, queues their images, and removes their
embeddings. The running daemon then reads the images and embeds the posts
again with their image text. It can run while the daemon is running.

## Running maintenance commands

Stop the daemon before running `bsearch backfill` or `bsearch reindex`.
They generate embeddings for posts with `has_embedding = 0`, and run
together with the daemon they collide on `vec_posts`, failing whichever
writes second.
