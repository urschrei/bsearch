use jiff::Timestamp;
use jiff::Zoned;
use jiff::civil;
use jiff::fmt::temporal::Pieces;
use jiff::tz::TimeZone;

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
/// than datetime types because their exact textual form matters: rows written
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
            indexed_at: format_indexed_at(Zoned::now().datetime()),
        }
    }
}

/// A link found in a liked post, waiting to be sent to the read-later
/// service. Stored in `pending_links`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingLink {
    pub url: String,
    /// The post the link was found in.
    pub post_uri: String,
    /// The page title, when the post's link card supplied one. Left to the
    /// service to look up otherwise.
    pub title: Option<String>,
    /// Shown alongside the bookmark: the author and text of the post.
    pub description: String,
}

/// Format an offset-aware timestamp the way Python's `datetime.isoformat()`
/// does, e.g. `2026-03-29T03:11:21.467000+00:00`.
///
/// Note the microsecond precision and the colon in the offset: RFC 3339
/// serialisers do not always emit these.
pub fn format_created_at(zdt: &Zoned) -> String {
    if zdt.subsec_nanosecond() == 0 {
        zdt.strftime("%Y-%m-%dT%H:%M:%S%:z").to_string()
    } else {
        zdt.strftime("%Y-%m-%dT%H:%M:%S.%6f%:z").to_string()
    }
}

/// Format a naive local timestamp the way Python's `datetime.now().isoformat()`
/// does, e.g. `2026-07-25T23:37:12.345678` -- no timezone offset.
pub fn format_indexed_at(dt: civil::DateTime) -> String {
    if dt.subsec_nanosecond() == 0 {
        dt.strftime("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        dt.strftime("%Y-%m-%dT%H:%M:%S.%6f").to_string()
    }
}

/// Parse an RFC 3339 timestamp into a fixed-offset [`Zoned`], keeping the
/// offset the text carried rather than normalising to UTC. Returns `None`
/// for text with no offset (a naive timestamp) or that does not parse.
fn parse_rfc3339(raw: &str) -> Option<Zoned> {
    let pieces = Pieces::parse(raw).ok()?;
    let offset = pieces.to_numeric_offset()?;
    let dt = pieces.date().to_datetime(pieces.time()?);
    dt.to_zoned(TimeZone::fixed(offset)).ok()
}

/// Parse a `createdAt` value from an AT Protocol record, falling back to the
/// current local time when it is missing or malformed. Mirrors the
/// `datetime.fromisoformat(s.replace("Z", "+00:00"))` handling in the Python
/// code, including its fallback to `datetime.now()`.
pub fn parse_created_at(raw: Option<&str>) -> String {
    raw.and_then(parse_rfc3339)
        .map(|zdt| format_created_at(&zdt))
        .unwrap_or_else(|| format_indexed_at(Zoned::now().datetime()))
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
pub fn created_at_sort_key(raw: &str) -> Option<Timestamp> {
    if let Some(zdt) = parse_rfc3339(raw) {
        return Some(zdt.timestamp());
    }
    let naive: civil::DateTime = raw.parse().ok()?;
    // Ambiguous local times (the repeated hour when clocks go back) resolve to
    // the earlier instant; either choice is arbitrary and this one is total.
    TimeZone::system()
        .to_ambiguous_zoned(naive)
        .earlier()
        .ok()
        .map(|zdt| zdt.timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_created_at_matches_python_isoformat() {
        let zdt = parse_rfc3339("2026-03-29T03:11:21.467Z").unwrap();
        assert_eq!(format_created_at(&zdt), "2026-03-29T03:11:21.467000+00:00");
    }

    #[test]
    fn test_format_created_at_preserves_offset() {
        let zdt = parse_rfc3339("2026-03-29T03:11:21.467123+01:00").unwrap();
        assert_eq!(format_created_at(&zdt), "2026-03-29T03:11:21.467123+01:00");
    }

    #[test]
    fn test_format_indexed_at_has_no_offset() {
        let dt = civil::date(2026, 7, 25).at(23, 37, 12, 345_678_000);
        assert_eq!(format_indexed_at(dt), "2026-07-25T23:37:12.345678");
    }

    #[test]
    fn test_format_created_at_omits_zero_microseconds() {
        // Python's isoformat() drops the fractional part entirely when
        // microsecond == 0, rather than emitting ".000000".
        let zdt = parse_rfc3339("2026-03-29T03:11:00Z").unwrap();
        assert_eq!(format_created_at(&zdt), "2026-03-29T03:11:00+00:00");
    }

    #[test]
    fn test_format_indexed_at_omits_zero_microseconds() {
        let dt = civil::date(2026, 7, 25).at(23, 37, 12, 0);
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
        let naive = format_indexed_at(Zoned::now().datetime());
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
        let zdt = parse_rfc3339("2026-03-29T03:11:21.467123+01:00").unwrap();
        let key = created_at_sort_key(&format_created_at(&zdt)).unwrap();
        assert_eq!(key, zdt.timestamp());
    }

    #[test]
    fn test_source_strings_match_python() {
        assert_eq!(Source::OwnPost.as_str(), "own_post");
        assert_eq!(Source::Like.as_str(), "like");
        assert_eq!(Source::BackfillPost.as_str(), "backfill_post");
        assert_eq!(Source::BackfillLike.as_str(), "backfill_like");
    }
}
