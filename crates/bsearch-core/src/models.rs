use chrono::DateTime;
use chrono::FixedOffset;
use chrono::Local;
use chrono::NaiveDateTime;

/// Where a post came from. Serialised into `posts.source`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    OwnPost,
    Like,
    BackfillPost,
    BackfillLike,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OwnPost => "own_post",
            Self::Like => "like",
            Self::BackfillPost => "backfill_post",
            Self::BackfillLike => "backfill_like",
        }
    }
}

/// A Bluesky post, either authored by the account or liked by it.
///
/// `created_at` and `indexed_at` are stored as pre-formatted strings rather
/// than chrono types because their exact textual form matters: rows written
/// here sit alongside rows written by the Python code, and both are read back
/// as opaque strings by the search binary. See [`format_created_at`] and
/// [`format_indexed_at`].
#[derive(Debug, Clone)]
pub struct Post {
    pub uri: String,
    pub cid: String,
    pub author_did: String,
    pub author_handle: String,
    pub text: String,
    pub created_at: String,
    pub source: &'static str,
    pub indexed_at: String,
}

impl Post {
    /// Build a post, stamping `indexed_at` with the current local time.
    pub fn new(
        uri: String,
        cid: String,
        author_did: String,
        author_handle: String,
        text: String,
        created_at: String,
        source: Source,
    ) -> Self {
        Self {
            uri,
            cid,
            author_did,
            author_handle,
            text,
            created_at,
            source: source.as_str(),
            indexed_at: format_indexed_at(Local::now().naive_local()),
        }
    }
}

/// Format a timezone-aware timestamp the way Python's `datetime.isoformat()`
/// does, e.g. `2026-03-29T03:11:21.467000+00:00`.
///
/// Note the microsecond precision and the colon in the offset: chrono's
/// `to_rfc3339` emits neither by default.
pub fn format_created_at(dt: DateTime<FixedOffset>) -> String {
    if dt.timestamp_subsec_micros() == 0 {
        dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
    } else {
        dt.format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string()
    }
}

/// Format a naive local timestamp the way Python's `datetime.now().isoformat()`
/// does, e.g. `2026-07-25T23:37:12.345678` -- no timezone offset.
pub fn format_indexed_at(dt: NaiveDateTime) -> String {
    if dt.and_utc().timestamp_subsec_micros() == 0 {
        dt.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }
}

/// Parse a `createdAt` value from an AT Protocol record, falling back to the
/// current local time when it is missing or malformed. Mirrors the
/// `datetime.fromisoformat(s.replace("Z", "+00:00"))` handling in the Python
/// code, including its fallback to `datetime.now()`.
pub fn parse_created_at(raw: Option<&str>) -> String {
    let Some(raw) = raw else {
        return format_indexed_at(Local::now().naive_local());
    };
    match DateTime::parse_from_rfc3339(raw) {
        Ok(dt) => format_created_at(dt),
        Err(_) => format_indexed_at(Local::now().naive_local()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_format_created_at_matches_python_isoformat() {
        let dt = DateTime::parse_from_rfc3339("2026-03-29T03:11:21.467Z").unwrap();
        assert_eq!(format_created_at(dt), "2026-03-29T03:11:21.467000+00:00");
    }

    #[test]
    fn test_format_created_at_preserves_offset() {
        let dt = DateTime::parse_from_rfc3339("2026-03-29T03:11:21.467123+01:00").unwrap();
        assert_eq!(format_created_at(dt), "2026-03-29T03:11:21.467123+01:00");
    }

    #[test]
    fn test_format_indexed_at_has_no_offset() {
        let dt = NaiveDate::from_ymd_opt(2026, 7, 25)
            .unwrap()
            .and_hms_micro_opt(23, 37, 12, 345678)
            .unwrap();
        assert_eq!(format_indexed_at(dt), "2026-07-25T23:37:12.345678");
    }

    #[test]
    fn test_format_created_at_omits_zero_microseconds() {
        // Python's isoformat() drops the fractional part entirely when
        // microsecond == 0, rather than emitting ".000000".
        let dt = DateTime::parse_from_rfc3339("2026-03-29T03:11:00Z").unwrap();
        assert_eq!(format_created_at(dt), "2026-03-29T03:11:00+00:00");
    }

    #[test]
    fn test_format_indexed_at_omits_zero_microseconds() {
        let dt = NaiveDate::from_ymd_opt(2026, 7, 25)
            .unwrap()
            .and_hms_opt(23, 37, 12)
            .unwrap();
        assert_eq!(format_indexed_at(dt), "2026-07-25T23:37:12");
    }

    #[test]
    fn test_parse_created_at_round_trips_z_suffix() {
        assert_eq!(
            parse_created_at(Some("2026-03-29T03:11:21.467Z")),
            "2026-03-29T03:11:21.467000+00:00"
        );
    }

    #[test]
    fn test_parse_created_at_falls_back_on_garbage() {
        // Should not panic, and should produce a naive local timestamp.
        let out = parse_created_at(Some("not a date"));
        assert!(!out.contains('+'), "fallback should be naive: {out}");
        let out = parse_created_at(None);
        assert!(!out.contains('+'), "fallback should be naive: {out}");
    }

    #[test]
    fn test_source_strings_match_python() {
        assert_eq!(Source::OwnPost.as_str(), "own_post");
        assert_eq!(Source::Like.as_str(), "like");
        assert_eq!(Source::BackfillPost.as_str(), "backfill_post");
        assert_eq!(Source::BackfillLike.as_str(), "backfill_like");
    }
}
