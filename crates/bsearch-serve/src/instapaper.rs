//! Client for Instapaper API v2, limited to what filing links in a folder
//! needs: finding or creating the folder, and adding bookmarks.
//!
//! Every request carries a personal access token as a bearer token. Request
//! bodies are JSON. A failure is reported by the HTTP status, with a JSON
//! body of the form `{"error": {"code": N, "message": ".."}}`.

use std::time::Duration;

use bsearch_core::models::PendingLink;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::config::InstapaperConfig;

const BASE_URL: &str = "https://www.instapaper.com/api/2";
const AGENT: &str = concat!("bsearch-serve/", env!("CARGO_PKG_VERSION"));
/// Bounds how long one call may hold the submission loop, and with it the
/// daemon's shutdown.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Instapaper API error (HTTP {status}): {message}")]
    Api { status: u16, message: String },
    #[error("Instapaper request failed: {0}")]
    Http(#[from] reqwest::Error),
    /// A body that is not the JSON the call should produce, or a failure
    /// status without an error body.
    #[error("Instapaper returned an unreadable response (HTTP {status}): {body}")]
    Unreadable { status: u16, body: String },
    #[error("Instapaper folder '{0}' could not be found or created")]
    FolderUnavailable(String),
    /// A save was rejected and the folder the session is bound to no
    /// longer exists.
    #[error("Instapaper folder {0} no longer exists")]
    FolderGone(u64),
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
            // `add_bookmark` turns a 400 caused by the folder into
            // `FolderGone`, so a 400 that arrives here is about the URL.
            Self::Api { status, .. } => match *status {
                400 => Disposition::DropLink,
                401 | 403 => Disposition::ResetSession,
                429 => Disposition::RateLimited,
                _ => Disposition::Retry,
            },
            Self::Unreadable { status, .. } => match *status {
                401 | 403 => Disposition::ResetSession,
                429 => Disposition::RateLimited,
                _ => Disposition::Retry,
            },
            Self::FolderGone(_) => Disposition::ReconnectFolder,
            Self::Http(_) | Self::FolderUnavailable(_) => Disposition::Retry,
        }
    }
}

#[derive(Serialize)]
struct NewFolder<'a> {
    title: &'a str,
}

#[derive(Serialize)]
struct NewBookmark<'a> {
    url: &'a str,
    /// Instapaper looks the title up itself when the field is absent;
    /// sending an empty one is not the same thing.
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<&'a str>,
    description: &'a str,
    folder_id: u64,
}

#[derive(Deserialize)]
struct FolderList {
    folders: Vec<Folder>,
}

#[derive(Deserialize)]
struct Folder {
    id: u64,
    title: String,
}

#[derive(Debug, Deserialize)]
struct Bookmark {
    id: u64,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

/// An authenticated connection bound to one folder.
///
/// Building one lists the account's folders, which also checks the token,
/// and finds or creates the folder, so a `Session` that exists is one that
/// can file bookmarks; the loop that owns it drops it and connects again if
/// the folder later disappears.
pub struct Session {
    client: Client,
    folder_id: u64,
    folder_title: String,
}

impl Session {
    pub async fn connect(config: &InstapaperConfig) -> Result<Self, Error> {
        let client = Client {
            http: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .user_agent(AGENT)
                .build()?,
            token: config.access_token.clone(),
        };
        let folder_id = client.find_or_create_folder(&config.folder).await?;
        Ok(Self {
            client,
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
    /// A URL the account already holds is not an error: Instapaper updates
    /// the existing bookmark with what is sent, folder included.
    pub async fn add_bookmark(&self, link: &PendingLink) -> Result<u64, Error> {
        let request = NewBookmark {
            url: &link.url,
            title: link.title.as_deref(),
            description: &link.description,
            folder_id: self.folder_id,
        };
        match self
            .client
            .send::<Bookmark>(self.client.post("bookmarks").json(&request))
            .await
        {
            Ok(bookmark) => Ok(bookmark.id),
            // A bad URL and a missing folder both answer 400, and only the
            // message text, which the documentation says not to match on,
            // tells them apart. Look the folder up instead.
            Err(e @ Error::Api { status: 400, .. }) => {
                let folders = self.client.list_folders().await?;
                if folders.iter().any(|folder| folder.id == self.folder_id) {
                    Err(e)
                } else {
                    Err(Error::FolderGone(self.folder_id))
                }
            }
            Err(e) => Err(e),
        }
    }
}

struct Client {
    http: reqwest::Client,
    token: String,
}

impl Client {
    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.get(format!("{BASE_URL}/{path}"))
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.post(format!("{BASE_URL}/{path}"))
    }

    async fn send<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, Error> {
        let response = request.bearer_auth(&self.token).send().await?;
        let status = response.status().as_u16();
        let body = response.text().await?;
        parse_response(status, &body)
    }

    async fn list_folders(&self) -> Result<Vec<Folder>, Error> {
        let list: FolderList = self.send(self.get("folders")).await?;
        Ok(list.folders)
    }

    /// The id of the folder titled `title`, creating it if the account has
    /// no such folder.
    async fn find_or_create_folder(&self, title: &str) -> Result<u64, Error> {
        if let Some(id) = find_folder(&self.list_folders().await?, title) {
            return Ok(id);
        }
        let request = NewFolder { title };
        match self
            .send::<Folder>(self.post("folders").json(&request))
            .await
        {
            Ok(folder) => Ok(folder.id),
            // Most likely created by someone else between the two calls;
            // look it up again.
            Err(Error::Api { status: 400, .. }) => find_folder(&self.list_folders().await?, title)
                .ok_or_else(|| Error::FolderUnavailable(title.to_string())),
            Err(e) => Err(e),
        }
    }
}

/// Read a response body as `T` when the status is a success, and as an
/// error body otherwise.
fn parse_response<T: DeserializeOwned>(status: u16, body: &str) -> Result<T, Error> {
    let unreadable = || Error::Unreadable {
        status,
        body: body.to_string(),
    };
    if (200..300).contains(&status) {
        return serde_json::from_str(body).map_err(|_| unreadable());
    }
    match serde_json::from_str::<ErrorBody>(body) {
        Ok(error) => Err(Error::Api {
            status,
            message: error.error.message,
        }),
        Err(_) => Err(unreadable()),
    }
}

fn find_folder(folders: &[Folder], title: &str) -> Option<u64> {
    folders
        .iter()
        .find(|folder| folder.title == title)
        .map(|folder| folder.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_response_reads_a_bookmark() {
        let bookmark: Bookmark = parse_response(
            200,
            r#"{"id":123,"url":"https://example.com/","title":null,"liked":false}"#,
        )
        .expect("should parse");
        assert_eq!(bookmark.id, 123);
    }

    #[test]
    fn test_parse_response_reads_a_folder_list() {
        let list: FolderList = parse_response(
            200,
            r#"{"folders":[{"id":99,"title":"bluesky-likes","slug":"bluesky-likes",
                "position":1,"public":false,"count":12},
               {"id":100,"title":"other","slug":"other","position":2,
                "public":false,"count":0}]}"#,
        )
        .expect("should parse");
        assert_eq!(find_folder(&list.folders, "bluesky-likes"), Some(99));
        assert_eq!(find_folder(&list.folders, "other"), Some(100));
        assert_eq!(find_folder(&list.folders, "missing"), None);
    }

    #[test]
    fn test_parse_response_surfaces_error_bodies() {
        let err = parse_response::<Bookmark>(
            400,
            r#"{"error":{"code":400,"message":"Invalid URL specified"}}"#,
        )
        .expect_err("error body must fail");
        match err {
            Error::Api { status, message } => {
                assert_eq!(status, 400);
                assert_eq!(message, "Invalid URL specified");
            }
            other => panic!("expected Api error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_treats_unexpected_bodies_as_unreadable() {
        let err =
            parse_response::<Bookmark>(502, "<html>Bad Gateway</html>").expect_err("must fail");
        match err {
            Error::Unreadable { status, .. } => assert_eq!(status, 502),
            other => panic!("expected Unreadable, got {other:?}"),
        }
        assert!(matches_unreadable(parse_response::<Bookmark>(
            200,
            r#"{"not":"a bookmark"}"#
        )));
        assert!(matches_unreadable(parse_response::<Bookmark>(500, "")));
    }

    fn matches_unreadable(result: Result<Bookmark, Error>) -> bool {
        match result {
            Err(Error::Unreadable { .. }) => true,
            Ok(_) | Err(_) => false,
        }
    }

    #[test]
    fn test_new_bookmark_omits_absent_title() {
        let request = NewBookmark {
            url: "https://example.com/a?b=c",
            title: None,
            description: "@alice: read this",
            folder_id: 42,
        };
        assert_eq!(
            serde_json::to_string(&request).unwrap(),
            r#"{"url":"https://example.com/a?b=c","description":"@alice: read this","folder_id":42}"#
        );
    }

    #[test]
    fn test_error_disposition() {
        let api = |status| Error::Api {
            status,
            message: String::new(),
        };
        assert_eq!(api(400).disposition(), Disposition::DropLink);
        assert_eq!(api(401).disposition(), Disposition::ResetSession);
        assert_eq!(api(403).disposition(), Disposition::ResetSession);
        assert_eq!(api(429).disposition(), Disposition::RateLimited);
        assert_eq!(api(500).disposition(), Disposition::Retry);
        assert_eq!(api(402).disposition(), Disposition::Retry);

        let unreadable = |status| Error::Unreadable {
            status,
            body: String::new(),
        };
        assert_eq!(unreadable(503).disposition(), Disposition::Retry);
        assert_eq!(unreadable(401).disposition(), Disposition::ResetSession);
        assert_eq!(unreadable(429).disposition(), Disposition::RateLimited);

        assert_eq!(
            Error::FolderGone(7).disposition(),
            Disposition::ReconnectFolder
        );
        assert_eq!(
            Error::FolderUnavailable("x".to_string()).disposition(),
            Disposition::Retry
        );
    }
}
