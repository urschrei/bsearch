use chrono::DateTime;
use chrono::FixedOffset;
use chrono::Local;
use chrono::NaiveDateTime;
use chrono::TimeZone;
use chrono::Utc;

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

/// Parse a stored `created_at` value back into an instant, for ordering.
///
/// The stored text is not in one fixed form. [`format_created_at`] preserves
/// whatever offset the source record carried, emits the fractional part only
/// when it is non-zero, and [`parse_created_at`] falls back to a naive local
/// timestamp when the record's value is missing or malformed. Comparing these
/// as strings therefore does not always give chronological order, so callers
/// that need to sort by date go through here.
///
/// Naive values are read as local time, which is what wrote them. Returns
/// `None` if neither form parses, leaving it to the caller to decide where
/// such a row belongs.
pub fn created_at_sort_key(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return Some(dt.with_timezone(&Utc));
    }
    let naive = NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S%.f").ok()?;
    // Ambiguous local times (the repeated hour when clocks go back) resolve to
    // the earlier instant; either choice is arbitrary and this one is total.
    Local
        .from_local_datetime(&naive)
        .earliest()
        .map(|dt| dt.with_timezone(&Utc))
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
    fn test_created_at_sort_key_orders_across_offsets() {
        // The whole point of parsing rather than comparing strings: this pair
        // sorts the wrong way round lexicographically, because the earlier
        // instant has the later wall-clock reading.
        let earlier = created_at_sort_key("2026-03-29T10:00:00+05:00").unwrap();
        let later = created_at_sort_key("2026-03-29T09:00:00+00:00").unwrap();
        assert!(earlier < later);
        assert!("2026-03-29T10:00:00+05:00" > "2026-03-29T09:00:00+00:00");
    }

    #[test]
    fn test_created_at_sort_key_handles_absent_fractional_part() {
        let without = created_at_sort_key("2026-03-29T03:11:00+00:00").unwrap();
        let with = created_at_sort_key("2026-03-29T03:11:00.467000+00:00").unwrap();
        assert!(without < with);
    }

    #[test]
    fn test_created_at_sort_key_accepts_naive_fallback() {
        // What parse_created_at writes when a record's timestamp is unusable.
        let naive = format_indexed_at(Local::now().naive_local());
        assert!(
            created_at_sort_key(&naive).is_some(),
            "naive fallback timestamps must still be sortable: {naive}"
        );
    }

    #[test]
    fn test_created_at_sort_key_rejects_garbage() {
        assert_eq!(created_at_sort_key("not a date"), None);
        assert_eq!(created_at_sort_key(""), None);
    }

    #[test]
    fn test_created_at_sort_key_round_trips_format_created_at() {
        let dt = DateTime::parse_from_rfc3339("2026-03-29T03:11:21.467123+01:00").unwrap();
        let key = created_at_sort_key(&format_created_at(dt)).unwrap();
        assert_eq!(key, dt.with_timezone(&Utc));
    }

    #[test]
    fn test_source_strings_match_python() {
        assert_eq!(Source::OwnPost.as_str(), "own_post");
        assert_eq!(Source::Like.as_str(), "like");
        assert_eq!(Source::BackfillPost.as_str(), "backfill_post");
        assert_eq!(Source::BackfillLike.as_str(), "backfill_like");
    }
}
