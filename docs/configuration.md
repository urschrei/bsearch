# Configuration

bsearch reads its settings from a `.env` file in the working directory. The
Python CLI and the daemon use the same keys.

## Credentials

| Key | Required | Description |
|---|---|---|
| `user` | Yes | Your Bluesky handle, for example `yourhandle.bsky.social` |
| `password` | Yes | An app password for the account |
| `pds_url` | No | Your PDS, if the account is not on `bsky.social` |

```
user=yourhandle.bsky.social
password=your-app-password
pds_url=https://pds.example.com
```

## Jetstream archive

| Key | Required | Description |
|---|---|---|
| `jetstream_key` | No | API key for Jetstream's sealed archive |

```
jetstream_key=gk_your-jetstream-api-key
```

The key lets the daemon replay history from Jetstream's sealed archive: the
initial sync, and recovery after downtime. It is used only on the archive
HTTP endpoints; the live WebSocket is unauthenticated. Without a key the
daemon still follows the live tail, but events that fall outside the live
replay buffer while it is offline are lost.

## Instapaper

| Key | Required | Description |
|---|---|---|
| `instapaper_access_token` | No | Personal access token for Instapaper API v2. Setting it turns link filing on |
| `instapaper_folder` | No | Folder for filed links. Default: `bluesky-likes` |

```
instapaper_access_token=your-access-token
instapaper_folder=bluesky-likes
```

To get a token, open your application at
<https://www.instapaper.com/developers/applications> (create one if you
have none) and select **Generate access token**. Instapaper shows the token
once; regenerating or revoking it on the same page stops the old one
working. The folder is created on first use if it does not exist.

Setting the folder without the token is an error, so a typo cannot silently
switch the feature off. The API v1 keys (`instapaper_consumer_key`,
`instapaper_consumer_secret`, `instapaper_username`, `instapaper_password`)
are no longer read, and the daemon does not start while any of them is set.

## Environment variables

| Variable | Description |
|---|---|
| `BSEARCH_DB_PATH` | Database path. Default: `bsearch.db` in the working directory |
| `BSEARCH_MODEL_DIR` | Embedding model directory. Default: `~/.cache/bsearch/all-MiniLM-L6-v2` |
| `RUST_LOG` | Daemon log filter. Default: `info,ort=warn` |

`BSEARCH_DB_PATH` can also be set in `.env`.

## Storage

The database is stored as `bsearch.db` in the working directory unless
`BSEARCH_DB_PATH` says otherwise.

Whole-segment archive downloads spool to `~/.cache/bsearch/segments` so an
interrupted transfer resumes from its last byte; completed files are
deleted after their events are ingested. The directory is normally empty:
plans filtered to one account almost always name individual blocks, which
are fetched in memory.
