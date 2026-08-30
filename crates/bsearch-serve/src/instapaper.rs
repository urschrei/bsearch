//! Client for the Instapaper Full API, limited to what filing links in a
//! folder needs: obtaining an access token, finding or creating the folder,
//! and adding bookmarks.
//!
//! The API is OAuth 1.0a with HMAC-SHA1 signatures. Every call is a POST
//! with its parameters in a form body and the OAuth parameters in the
//! `Authorization` header. Responses are JSON arrays of typed items, except
//! the token exchange, which answers with a query string.

use std::collections::HashMap;
use std::time::Duration;

use bsearch_core::models::PendingLink;
use oauth1_request as oauth;
use reqwest::header::AUTHORIZATION;
use reqwest::header::CONTENT_TYPE;
use serde_json::Value;

use crate::config::InstapaperConfig;

const BASE_URL: &str = "https://www.instapaper.com/api/1";
const AGENT: &str = concat!("bsearch-serve/", env!("CARGO_PKG_VERSION"));
/// Bounds how long one call may hold the submission loop, and with it the
/// daemon's shutdown.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Error codes from the API documentation that matter here.
pub const RATE_LIMIT_EXCEEDED: u32 = 1040;
pub const DOMAIN_REQUIRES_CONTENT: u32 = 1220;
pub const DOMAIN_OPTED_OUT: u32 = 1221;
pub const INVALID_URL: u32 = 1240;
pub const INVALID_FOLDER_ID: u32 = 1242;
pub const PRIVATE_REQUIRES_CONTENT: u32 = 1245;
pub const FOLDER_TITLE_EXISTS: u32 = 1251;
pub const TEXT_GENERATION_FAILED: u32 = 1550;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Instapaper API error {code}: {message}")]
    Api { code: u32, message: String },
    #[error("Instapaper request failed: {0}")]
    Http(#[from] reqwest::Error),
    /// The documentation says to read anything that is not valid JSON as
    /// a 503 and retry later.
    #[error("Instapaper returned an unreadable response (HTTP {status}): {body}")]
    Unreadable { status: u16, body: String },
    /// A well-formed response that lacks the item the call should produce.
    #[error("Instapaper response has no {expected} item: {body}")]
    MissingItem {
        expected: &'static str,
        body: String,
    },
    #[error("Instapaper rejected the credentials (HTTP {status}): {body}")]
    Auth { status: u16, body: String },
    #[error("Instapaper folder '{0}' could not be found or created")]
    FolderUnavailable(String),
}

/// What the submission loop should do about a failed call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// The URL itself is at fault, so sending it again can never succeed.
    DropLink,
    /// The folder the session is bound to is gone; rebuild the session so
    /// that the folder is found or created again.
    ReconnectFolder,
    /// The token is no longer accepted; rebuild the session.
    ResetSession,
    /// Instapaper's rate limit; wait longer than usual before the next pass.
    RateLimited,
    /// Anything else: keep the link queued and try again later.
    Retry,
}

impl Error {
    pub fn disposition(&self) -> Disposition {
        match self {
            // Codes not named here are unknown or account-level; retrying
            // costs nothing and keeps the link.
            Self::Api { code, .. } => match *code {
                DOMAIN_REQUIRES_CONTENT
                | DOMAIN_OPTED_OUT
                | INVALID_URL
                | PRIVATE_REQUIRES_CONTENT
                | TEXT_GENERATION_FAILED => Disposition::DropLink,
                INVALID_FOLDER_ID => Disposition::ReconnectFolder,
                RATE_LIMIT_EXCEEDED => Disposition::RateLimited,
                _ => Disposition::Retry,
            },
            // A revoked token does not arrive as an error item but as a
            // bare 401 or 403.
            Self::Unreadable { status, .. } => match *status {
                401 | 403 => Disposition::ResetSession,
                _ => Disposition::Retry,
            },
            Self::Auth { .. } => Disposition::ResetSession,
            Self::Http(_) | Self::MissingItem { .. } | Self::FolderUnavailable(_) => {
                Disposition::Retry
            }
        }
    }
}

type Credentials = oauth::Credentials<String>;

/// A signing identity: the consumer key alone before the token exchange,
/// consumer key plus access token after it.
struct Signer {
    consumer: Credentials,
    token: Option<Credentials>,
}

impl Signer {
    fn authorization(&self, uri: &str, request: &impl oauth::Request) -> String {
        let mut builder = oauth::Builder::new(self.consumer.as_ref(), oauth::HMAC_SHA1);
        builder.token(self.token.as_ref().map(Credentials::as_ref));
        builder.post(uri, request)
    }
}

#[derive(oauth::Request)]
struct AccessTokenRequest<'a> {
    x_auth_username: &'a str,
    x_auth_password: &'a str,
    x_auth_mode: &'a str,
}

#[derive(oauth::Request)]
struct AddFolderRequest<'a> {
    title: &'a str,
}

#[derive(oauth::Request)]
struct AddBookmarkRequest<'a> {
    url: &'a str,
    title: Option<&'a str>,
    description: &'a str,
    folder_id: u64,
}

/// An authenticated connection bound to one folder.
///
/// Building one performs the token exchange and the folder lookup, so a
/// `Session` that exists is one that can file bookmarks; the loop that owns
/// it drops it and connects again if the folder later disappears.
pub struct Session {
    http: reqwest::Client,
    signer: Signer,
    folder_id: u64,
    folder_title: String,
}

impl Session {
    pub async fn connect(config: &InstapaperConfig) -> Result<Self, Error> {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .user_agent(AGENT)
            .build()?;
        let mut signer = Signer {
            consumer: Credentials::new(config.consumer_key.clone(), config.consumer_secret.clone()),
            token: None,
        };

        let token = access_token(&http, &signer, &config.username, &config.password).await?;
        signer.token = Some(token);

        let folder_id = find_or_create_folder(&http, &signer, &config.folder).await?;
        Ok(Self {
            http,
            signer,
            folder_id,
            folder_title: config.folder.clone(),
        })
    }

    pub fn folder_id(&self) -> u64 {
        self.folder_id
    }

    pub fn folder_title(&self) -> &str {
        &self.folder_title
    }

    /// File a link in the folder, returning Instapaper's bookmark id.
    ///
    /// A URL the account already holds is not an error: Instapaper moves the
    /// existing bookmark into the folder and updates its description.
    pub async fn add_bookmark(&self, link: &PendingLink) -> Result<u64, Error> {
        let request = AddBookmarkRequest {
            url: &link.url,
            title: link.title.as_deref(),
            description: &link.description,
            folder_id: self.folder_id,
        };
        let items = call(&self.http, &self.signer, "bookmarks/add", &request).await?;
        items
            .iter()
            .find(|item| item_type(item) == Some("bookmark"))
            .and_then(|item| lenient_u64(item.get("bookmark_id")))
            .ok_or_else(|| Error::MissingItem {
                expected: "bookmark",
                body: Value::Array(items).to_string(),
            })
    }
}

/// Exchange the account's username and password for an access token
/// (xAuth), the only way Instapaper issues one.
async fn access_token(
    http: &reqwest::Client,
    signer: &Signer,
    username: &str,
    password: &str,
) -> Result<Credentials, Error> {
    let request = AccessTokenRequest {
        x_auth_username: username,
        x_auth_password: password,
        x_auth_mode: "client_auth",
    };
    let (status, body) = post_signed(http, signer, "oauth/access_token", &request).await?;
    if !status.is_success() {
        return Err(Error::Auth {
            status: status.as_u16(),
            body,
        });
    }
    parse_access_token(&body).ok_or(Error::Unreadable {
        status: status.as_u16(),
        body,
    })
}

/// The id of the folder titled `title`, creating it if the account has no
/// such folder.
async fn find_or_create_folder(
    http: &reqwest::Client,
    signer: &Signer,
    title: &str,
) -> Result<u64, Error> {
    let folders = call(http, signer, "folders/list", &()).await?;
    if let Some(id) = find_folder(&folders, title) {
        return Ok(id);
    }

    let request = AddFolderRequest { title };
    match call(http, signer, "folders/add", &request).await {
        Ok(items) => find_folder(&items, title)
            .or_else(|| {
                items
                    .iter()
                    .find(|item| item_type(item) == Some("folder"))
                    .and_then(|item| lenient_u64(item.get("folder_id")))
            })
            .ok_or_else(|| Error::FolderUnavailable(title.to_string())),
        // Created by someone else between the two calls; look it up again.
        Err(Error::Api { code, .. }) if code == FOLDER_TITLE_EXISTS => {
            let folders = call(http, signer, "folders/list", &()).await?;
            find_folder(&folders, title).ok_or_else(|| Error::FolderUnavailable(title.to_string()))
        }
        Err(e) => Err(e),
    }
}

/// Make a signed call and read its items, turning an error item into
/// [`Error::Api`].
async fn call(
    http: &reqwest::Client,
    signer: &Signer,
    method: &str,
    request: &impl oauth::Request,
) -> Result<Vec<Value>, Error> {
    let (status, body) = post_signed(http, signer, method, request).await?;
    parse_items(status.as_u16(), &body)
}

async fn post_signed(
    http: &reqwest::Client,
    signer: &Signer,
    method: &str,
    request: &impl oauth::Request,
) -> Result<(reqwest::StatusCode, String), Error> {
    let uri = format!("{BASE_URL}/{method}");
    let authorization = signer.authorization(&uri, request);
    let body = oauth::to_form(request);
    let response = http
        .post(&uri)
        .header(AUTHORIZATION, authorization)
        .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    Ok((status, body))
}

/// Read `oauth_token=...&oauth_token_secret=...`.
fn parse_access_token(body: &str) -> Option<Credentials> {
    let pairs: HashMap<&str, String> = body
        .trim()
        .split('&')
        .filter_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            Some((key, urlencoding::decode(value).ok()?.into_owned()))
        })
        .collect();
    let token = pairs.get("oauth_token")?;
    let secret = pairs.get("oauth_token_secret")?;
    if token.is_empty() || secret.is_empty() {
        return None;
    }
    Some(Credentials::new(token.clone(), secret.clone()))
}

/// Read a standard response: a JSON array of typed items, an error item
/// meaning the call failed. Anything else is unreadable, which the
/// documentation says to treat as a temporary outage.
fn parse_items(status: u16, body: &str) -> Result<Vec<Value>, Error> {
    let unreadable = || Error::Unreadable {
        status,
        body: body.to_string(),
    };
    let items = match serde_json::from_str::<Value>(body) {
        Ok(Value::Array(items)) => items,
        Ok(_) | Err(_) => return Err(unreadable()),
    };
    if let Some(error) = items.iter().find(|item| item_type(item) == Some("error")) {
        return Err(Error::Api {
            code: lenient_u64(error.get("error_code"))
                .and_then(|code| u32::try_from(code).ok())
                .unwrap_or(0),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    if !(200..300).contains(&status) {
        return Err(unreadable());
    }
    Ok(items)
}

fn find_folder(items: &[Value], title: &str) -> Option<u64> {
    items
        .iter()
        .filter(|item| item_type(item) == Some("folder"))
        .find(|item| item.get("title").and_then(Value::as_str) == Some(title))
        .and_then(|item| lenient_u64(item.get("folder_id")))
}

fn item_type(item: &Value) -> Option<&str> {
    item.get("type").and_then(Value::as_str)
}

/// Ids arrive as numbers in some responses and as numeric strings in
/// others; accept either.
fn lenient_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse().ok(),
        Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_access_token_reads_both_parts() {
        let creds = parse_access_token("oauth_token=aabbccdd&oauth_token_secret=efgh1234\n")
            .expect("should parse");
        assert_eq!(creds.identifier, "aabbccdd");
        assert_eq!(creds.secret, "efgh1234");
    }

    #[test]
    fn test_parse_access_token_rejects_incomplete_bodies() {
        assert!(parse_access_token("oauth_token=aabbccdd").is_none());
        assert!(parse_access_token("oauth_token=&oauth_token_secret=x").is_none());
        assert!(parse_access_token("<html>nope</html>").is_none());
    }

    #[test]
    fn test_parse_items_returns_the_array() {
        let items = parse_items(200, r#"[{"type":"folder","folder_id":12,"title":"x"}]"#)
            .expect("should parse");
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_parse_items_surfaces_error_items() {
        let err = parse_items(
            400,
            r#"[{"type":"error","error_code":1240,"message":"Invalid URL specified"}]"#,
        )
        .expect_err("error item must fail");
        match err {
            Error::Api { code, message } => {
                assert_eq!(code, INVALID_URL);
                assert_eq!(message, "Invalid URL specified");
            }
            other => panic!("expected Api error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_items_treats_non_json_as_unreadable() {
        let err = parse_items(502, "<html>Bad Gateway</html>").expect_err("must fail");
        match err {
            Error::Unreadable { status, .. } => assert_eq!(status, 502),
            other => panic!("expected Unreadable, got {other:?}"),
        }
        assert!(matches_unreadable(parse_items(
            200,
            r#"{"not":"an array"}"#
        )));
    }

    #[test]
    fn test_parse_items_rejects_failure_status_without_error_item() {
        assert!(matches_unreadable(parse_items(500, "[]")));
    }

    fn matches_unreadable(result: Result<Vec<Value>, Error>) -> bool {
        match result {
            Err(Error::Unreadable { .. }) => true,
            Ok(_) | Err(_) => false,
        }
    }

    #[test]
    fn test_find_folder_accepts_numeric_and_string_ids() {
        let items: Vec<Value> = serde_json::from_str(
            r#"[{"type":"folder","folder_id":"7","title":"bluesky-likes"},
                {"type":"folder","folder_id":8,"title":"other"}]"#,
        )
        .unwrap();
        assert_eq!(find_folder(&items, "bluesky-likes"), Some(7));
        assert_eq!(find_folder(&items, "other"), Some(8));
        assert_eq!(find_folder(&items, "missing"), None);
    }

    #[test]
    fn test_add_bookmark_form_omits_absent_title() {
        // Instapaper looks the title up itself when the parameter is
        // absent; sending an empty one is not the same thing.
        let request = AddBookmarkRequest {
            url: "https://example.com/a?b=c",
            title: None,
            description: "@alice: read this",
            folder_id: 42,
        };
        assert_eq!(
            oauth::to_form(&request),
            "description=%40alice%3A%20read%20this&folder_id=42&url=https%3A%2F%2Fexample.com%2Fa%3Fb%3Dc"
        );
    }

    #[test]
    fn test_xauth_request_is_signed_with_consumer_only() {
        // Before the token exchange there is no token; the header must
        // still carry the consumer key and a signature.
        let signer = Signer {
            consumer: Credentials::new("ck".to_string(), "cs".to_string()),
            token: None,
        };
        let request = AccessTokenRequest {
            x_auth_username: "alice",
            x_auth_password: "",
            x_auth_mode: "client_auth",
        };
        let header = signer.authorization(
            "https://www.instapaper.com/api/1/oauth/access_token",
            &request,
        );
        assert!(header.starts_with("OAuth "));
        assert!(header.contains("oauth_consumer_key=\"ck\""));
        assert!(header.contains("oauth_signature_method=\"HMAC-SHA1\""));
        assert!(header.contains("oauth_signature="));
        assert!(!header.contains("oauth_token="));
    }

    #[test]
    fn test_error_disposition() {
        let api = |code| Error::Api {
            code,
            message: String::new(),
        };
        assert_eq!(api(INVALID_URL).disposition(), Disposition::DropLink);
        assert_eq!(api(DOMAIN_OPTED_OUT).disposition(), Disposition::DropLink);
        assert_eq!(
            api(RATE_LIMIT_EXCEEDED).disposition(),
            Disposition::RateLimited
        );
        assert_eq!(
            api(INVALID_FOLDER_ID).disposition(),
            Disposition::ReconnectFolder
        );
        assert_eq!(api(1500).disposition(), Disposition::Retry);
        assert_eq!(api(0).disposition(), Disposition::Retry);

        let unreadable = |status| Error::Unreadable {
            status,
            body: String::new(),
        };
        assert_eq!(unreadable(503).disposition(), Disposition::Retry);
        assert_eq!(unreadable(401).disposition(), Disposition::ResetSession);
        assert_eq!(
            Error::Auth {
                status: 401,
                body: String::new()
            }
            .disposition(),
            Disposition::ResetSession
        );
    }
}
