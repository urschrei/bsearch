use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;

/// Daemon configuration, loaded from `.env` and defaults.
///
/// Mirrors `Config` in `src/bsearch/config.py`, including its tolerance of
/// both `KEY=VALUE` and `key: value` lines in `.env`.
#[derive(Debug, Clone)]
pub struct Config {
    pub handle: String,
    pub app_password: String,
    pub did: String,
    pub pds_url: String,
    pub db_path: PathBuf,
    pub model_dir: PathBuf,
    pub jetstream_hostname: String,
    pub like_batch_interval: u64,
    pub embedding_batch_interval: u64,
    pub reconnect_cursor_safety_seconds: i64,
    /// How long the embedding loop may sit idle before the ONNX session is
    /// dropped. Reloading costs well under a second, and holding the session
    /// open is the difference between roughly 20 MB and 90 MB resident.
    pub embedder_idle_timeout: u64,
}

const DEFAULT_JETSTREAM_HOSTNAME: &str = "jetstream2.us-east.bsky.network";
const DEFAULT_PDS_URL: &str = "https://bsky.social";

impl Config {
    pub fn from_env(env_path: Option<&Path>) -> Result<Self> {
        let env_path = env_path.unwrap_or_else(|| Path::new(".env"));
        let _ = dotenvy::from_path(env_path);
        let file_values = parse_env_file(env_path);

        let handle = lookup(&file_values, &["BSEARCH_HANDLE", "user"]).context(
            "Missing credentials. Set BSEARCH_HANDLE and BSEARCH_APP_PASSWORD in .env \
             (or 'user' and 'password').",
        )?;
        let app_password = lookup(&file_values, &["BSEARCH_APP_PASSWORD", "password"]).context(
            "Missing credentials. Set BSEARCH_HANDLE and BSEARCH_APP_PASSWORD in .env \
             (or 'user' and 'password').",
        )?;

        let db_path = lookup(&file_values, &["BSEARCH_DB_PATH", "db_path"])
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("bsearch.db"));

        let pds_url = lookup(&file_values, &["BSEARCH_PDS_URL", "pds_url"])
            .unwrap_or_else(|| DEFAULT_PDS_URL.to_string());

        // Matches the `~/.cache/bsearch/<model>` layout that
        // `bsearch export-model` writes to and `bsearch-search` reads from.
        let default_model_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cache")
            .join("bsearch")
            .join("all-MiniLM-L6-v2");
        let model_dir = lookup(&file_values, &["BSEARCH_MODEL_DIR"])
            .map(PathBuf::from)
            .unwrap_or(default_model_dir);

        let jetstream_hostname = lookup(&file_values, &["BSEARCH_JETSTREAM_HOSTNAME"])
            .unwrap_or_else(|| DEFAULT_JETSTREAM_HOSTNAME.to_string());

        Ok(Self {
            handle,
            app_password,
            did: lookup(&file_values, &["BSEARCH_DID", "did"]).unwrap_or_default(),
            pds_url,
            db_path,
            model_dir,
            jetstream_hostname,
            like_batch_interval: 2,
            embedding_batch_interval: 10,
            reconnect_cursor_safety_seconds: 5,
            embedder_idle_timeout: 300,
        })
    }
}

/// Look a key up in the environment first, then in the parsed `.env` values.
fn lookup(file_values: &HashMap<String, String>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Ok(value) = std::env::var(key) {
            if !value.is_empty() {
                return Some(value);
            }
        }
        if let Some(value) = file_values.get(*key) {
            if !value.is_empty() {
                return Some(value.clone());
            }
        }
    }
    None
}

/// Parse a `.env` file that may use either `key=value` or `key: value`.
///
/// `dotenvy` only understands the former, but the Python loader accepts both,
/// and existing `.env` files in the wild use the YAML-ish style.
fn parse_env_file(path: &Path) -> HashMap<String, String> {
    let mut values = HashMap::new();
    let Ok(contents) = std::fs::read_to_string(path) else {
        return values;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let pair = if let Some(idx) = line.find('=') {
            Some((&line[..idx], &line[idx + 1..]))
        } else {
            line.find(": ").map(|idx| (&line[..idx], &line[idx + 2..]))
        };
        if let Some((key, value)) = pair {
            values.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_env(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(contents.as_bytes()).expect("write");
        file.flush().expect("flush");
        file
    }

    #[test]
    fn test_parses_equals_format() {
        let file = write_env("user=alice.bsky.social\npassword=hunter2\n");
        let values = parse_env_file(file.path());
        assert_eq!(values.get("user").unwrap(), "alice.bsky.social");
        assert_eq!(values.get("password").unwrap(), "hunter2");
    }

    #[test]
    fn test_parses_colon_format() {
        let file = write_env("user: alice.bsky.social\npassword: hunter2\n");
        let values = parse_env_file(file.path());
        assert_eq!(values.get("user").unwrap(), "alice.bsky.social");
        assert_eq!(values.get("password").unwrap(), "hunter2");
    }

    #[test]
    fn test_skips_comments_and_blanks() {
        let file = write_env("# a comment\n\nuser=alice\n");
        let values = parse_env_file(file.path());
        assert_eq!(values.len(), 1);
        assert_eq!(values.get("user").unwrap(), "alice");
    }

    #[test]
    fn test_value_containing_equals_is_kept_whole() {
        // App passwords and URLs can contain '='; only the first one splits.
        let file = write_env("password=abc=def=ghi\n");
        let values = parse_env_file(file.path());
        assert_eq!(values.get("password").unwrap(), "abc=def=ghi");
    }

    #[test]
    fn test_missing_file_is_not_an_error() {
        let values = parse_env_file(Path::new("/nonexistent/.env"));
        assert!(values.is_empty());
    }
}
