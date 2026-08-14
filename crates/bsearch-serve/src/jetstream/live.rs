//! The Jetstream v2 live tail: `/xrpc/network.bsky.jetstream.subscribeEvents`.
//!
//! The endpoint is unauthenticated. Frames are `xrpc.v1.json` JSON text
//! frames; this client never negotiates compression, so binary frames are
//! unexpected and skipped.
#![allow(dead_code)]

use anyhow::Result;
use futures_util::SinkExt;
use futures_util::StreamExt;
use http::header::HeaderValue;
use http::header::SEC_WEBSOCKET_PROTOCOL;
use http::header::USER_AGENT;
use http::Uri;
use tokio_websockets::ClientBuilder;
use tokio_websockets::MaybeTlsStream;
use tokio_websockets::WebSocketStream;

use super::events::decode_frame;
use super::events::Frame;

/// What to subscribe to. `cursor` is inclusive; it may be a seq or, at or
/// above 1e15, a unix-microsecond timestamp the server translates. `None`
/// starts at the live tip.
#[derive(Debug, Clone)]
pub struct LiveParams {
    pub hostname: String,
    pub dids: Vec<String>,
    pub collections: Vec<String>,
    pub cursor: Option<u64>,
}

/// Build the subscribe URL. Filters use the v2 parameter names; the legacy
/// `wanted*` names are rejected by the server with a 400.
pub fn subscribe_url(params: &LiveParams) -> String {
    use std::fmt::Write;
    let mut url = format!(
        "wss://{}/xrpc/network.bsky.jetstream.subscribeEvents",
        params.hostname
    );
    let mut separator = '?';
    for did in &params.dids {
        write!(url, "{separator}dids={}", urlencoding::encode(did)).unwrap();
        separator = '&';
    }
    for collection in &params.collections {
        write!(
            url,
            "{separator}collections={}",
            urlencoding::encode(collection)
        )
        .unwrap();
        separator = '&';
    }
    if let Some(cursor) = params.cursor {
        write!(url, "{separator}cursor={cursor}").unwrap();
    }
    url
}

/// A connect failure the caller may act on.
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    /// The server refused the upgrade with an HTTP status. A 400 on a
    /// well-formed request means the cursor is below the retention floor
    /// (`CursorTooOld`); the response body naming the error is not
    /// available through the WebSocket library, so the status is the
    /// signal.
    #[error("server refused the connection with HTTP status {status}")]
    Rejected { status: u16 },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub struct Live {
    ws: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
}

/// Open one live-tail connection.
pub async fn connect(params: &LiveParams) -> Result<Live, ConnectError> {
    let url = subscribe_url(params);
    let uri: Uri = url
        .parse()
        .map_err(|e| ConnectError::Other(anyhow::anyhow!("bad subscribe url {url}: {e}")))?;
    let user_agent = concat!("bsearch/", env!("CARGO_PKG_VERSION"));
    let builder = ClientBuilder::from_uri(uri)
        .add_header(USER_AGENT, HeaderValue::from_static(user_agent))
        .map_err(|e| ConnectError::Other(e.into()))?
        // Offered but not verified: the lexicon declares this subprotocol
        // as the default, so an empty echo means identical framing.
        .add_header(
            SEC_WEBSOCKET_PROTOCOL,
            HeaderValue::from_static("xrpc.v1.json"),
        )
        .map_err(|e| ConnectError::Other(e.into()))?;
    match builder.connect().await {
        Ok((ws, _response)) => Ok(Live { ws }),
        Err(tokio_websockets::Error::Upgrade(upgrade)) => {
            let text = upgrade.to_string();
            match upgrade {
                tokio_websockets::upgrade::Error::DidNotSwitchProtocols(status) => {
                    Err(ConnectError::Rejected { status })
                }
                _ => Err(ConnectError::Other(anyhow::anyhow!(
                    "websocket upgrade failed: {text}"
                ))),
            }
        }
        Err(e) => Err(ConnectError::Other(e.into())),
    }
}

impl Live {
    /// Read the next frame. `Ok(None)` means the server closed the
    /// connection. Non-text frames are skipped: compression is never
    /// negotiated, so nothing meaningful arrives as binary.
    pub async fn next_frame(&mut self) -> Result<Option<Frame>> {
        loop {
            let Some(message) = self.ws.next().await else {
                return Ok(None);
            };
            let message = message?;
            if message.is_close() {
                return Ok(None);
            }
            let Some(text) = message.as_text() else {
                tracing::debug!("skipping non-text frame");
                continue;
            };
            return Ok(Some(decode_frame(text)?));
        }
    }

    /// Send a close frame and drop the connection politely.
    pub async fn close(mut self) {
        let _ = self.ws.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(cursor: Option<u64>) -> LiveParams {
        LiveParams {
            hostname: "jetstream.us-east.bsky.network".to_string(),
            dids: vec!["did:plc:abc123".to_string()],
            collections: vec![
                "app.bsky.feed.post".to_string(),
                "app.bsky.feed.like".to_string(),
            ],
            cursor,
        }
    }

    #[test]
    fn test_subscribe_url_with_seq_cursor() {
        assert_eq!(
            subscribe_url(&params(Some(42))),
            "wss://jetstream.us-east.bsky.network/xrpc/network.bsky.jetstream.subscribeEvents\
             ?dids=did%3Aplc%3Aabc123\
             &collections=app.bsky.feed.post\
             &collections=app.bsky.feed.like\
             &cursor=42"
        );
    }

    #[test]
    fn test_subscribe_url_with_legacy_timestamp_cursor() {
        // A v1 time_us cursor is passed through unchanged; the server
        // recognises the magnitude and translates it to the nearest seq.
        let url = subscribe_url(&params(Some(1_785_009_595_019_071)));
        assert!(url.ends_with("&cursor=1785009595019071"));
    }

    #[test]
    fn test_subscribe_url_without_cursor_starts_at_tip() {
        let url = subscribe_url(&params(None));
        assert!(!url.contains("cursor"));
    }

    #[tokio::test]
    #[ignore = "network: connects to the public Jetstream instance"]
    async fn test_live_tail_decodes_real_frames() {
        let p = LiveParams {
            hostname: "jetstream.us-east.bsky.network".to_string(),
            dids: vec![],
            collections: vec![],
            cursor: None,
        };
        let mut live = connect(&p).await.expect("connect");
        let mut events = 0;
        while events < 3 {
            match live.next_frame().await.expect("frame") {
                Some(Frame::Event(event)) => {
                    assert!(event.seq > 0, "live events carry a positive seq");
                    assert!(!event.did.is_empty());
                    events += 1;
                }
                Some(_) => {}
                None => panic!("connection closed before three events arrived"),
            }
        }
        live.close().await;
    }

    #[test]
    fn test_subscribe_url_without_filters_has_no_query() {
        let p = LiveParams {
            hostname: "example.test".to_string(),
            dids: vec![],
            collections: vec![],
            cursor: None,
        };
        assert_eq!(
            subscribe_url(&p),
            "wss://example.test/xrpc/network.bsky.jetstream.subscribeEvents"
        );
    }
}
