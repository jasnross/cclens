//! Filter primitives shared by the binary's CLI layer and the library's
//! rendering / attribution layers.
//!
//! Public API:
//! - `ThresholdsFilter` — value type holding `(min_tokens, min_cost)`.
//! - `ThresholdsFilter::matches` — predicate over `(tokens, cost)`.
//! - `SessionFilter` — value type holding
//!   `(project_name, since, until)`.
//! - `SessionFilter::accepts` — predicate over
//!   `(project_short_name, started_at)`.
//! - `QueryScope` — one of the three loaders a filter component can
//!   constrain.
//! - `HonoredBy` — which loaders honor a given component.
//! - `FilterComponent` — one rendered, flag-shaped filter component
//!   plus the loaders it constrains.
//! - `ThresholdsFilter::describe_active` /
//!   `SessionFilter::describe_active` — the active flags as
//!   `Vec<FilterComponent>`, in display order.
//! - `parse_filter_datetime` — lenient `--since` / `--until` parser
//!   (bare `YYYY-MM-DD` or full RFC 3339), shared by clap and the TUI.
//! - `render_filter_datetime` — its inverse, in the shortest spelling
//!   that reparses to the same instant.
//! - `quote_filter_value` — single-quotes a component value that
//!   would not otherwise reparse as one shell word.
//! - `parse_min_cost` — `--min-cost` parser rejecting negative and
//!   non-finite values, shared by clap and the TUI filter editor.

use std::str::FromStr;

use chrono::{DateTime, Local, NaiveDate, TimeZone, Utc};

/// One of the three loaders a filter component can constrain. Named
/// per loader rather than per view because that is the granularity
/// causation needs: `load_show` resolves its session by id and applies
/// thresholds only, so a Show empty state naming `--project` would
/// blame a filter that provably excluded nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryScope {
    Sessions,
    Show,
    Inputs,
}

/// Which loaders honor a filter component. The three constants below
/// are the whole vocabulary — a component is built from one of them,
/// never from arbitrary flags, so adding a loader-scoped filter means
/// naming its shape here rather than leaving a call site to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HonoredBy {
    sessions: bool,
    show: bool,
    inputs: bool,
}

impl HonoredBy {
    /// `--min-tokens` / `--min-cost`: thresholds every loader applies.
    pub const EVERY_LOADER: Self = Self {
        sessions: true,
        show: true,
        inputs: true,
    };
    /// `--project` / `--since` / `--until`: honored wherever sessions
    /// are *selected*. `load_show` is handed one session id, so these
    /// never narrow it.
    pub const SESSION_SCOPED: Self = Self {
        sessions: true,
        show: false,
        inputs: true,
    };
    /// `--session`: the inputs view is the only reader.
    pub const INPUTS_ONLY: Self = Self {
        sessions: false,
        show: false,
        inputs: true,
    };

    /// Whether the loader behind `scope` applies this component.
    #[must_use]
    pub fn honors(self, scope: QueryScope) -> bool {
        match scope {
            QueryScope::Sessions => self.sessions,
            QueryScope::Show => self.show,
            QueryScope::Inputs => self.inputs,
        }
    }
}

/// One rendered filter component and the loaders it constrains, in
/// display order. `text` is flag-shaped on every surface — the CLI
/// joins components with spaces, the TUI header styles each one by
/// whether it applies to the visible tab.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterComponent {
    pub text: String,
    pub honored_by: HonoredBy,
}

/// The first instant of `date` in the local zone. Not always
/// midnight: on a spring-forward day the local clock skips it, so the
/// day begins at the first hour that exists.
fn local_day_start(date: NaiveDate) -> Option<DateTime<Utc>> {
    (0..3).find_map(|hour| {
        let naive = date.and_hms_opt(hour, 0, 0)?;
        Local
            .from_local_datetime(&naive)
            .earliest()
            .map(|dt| dt.with_timezone(&Utc))
    })
}

/// Parse a filter timestamp, accepting a bare `YYYY-MM-DD` (the start
/// of that day, *locally*) in addition to full RFC 3339. Shared by
/// clap's `--since` / `--until` and the TUI filter editor so one date
/// vocabulary serves both surfaces.
///
/// Local, not UTC, because every timestamp cclens renders is local: a
/// UTC reading silently drops rows the table visibly shows on the day
/// the user named. `git log --since` resolves bare dates the same way.
///
/// A bare date means one instant on both flags — the same one,
/// whichever flag consumed it. Because `SessionFilter::accepts` is
/// inclusive at both ends, the two flags then read asymmetrically:
/// `--since 2026-04-15` admits all of the 15th, while
/// `--until 2026-04-15` admits only sessions starting at exactly that
/// instant. That asymmetry is what lets one `value_parser` serve both
/// flags, since it never learns which flag it is parsing for.
///
/// # Errors
/// Returns a human-readable message when the text matches neither
/// accepted form, or names a date with no local start instant.
pub fn parse_filter_datetime(s: &str) -> Result<DateTime<Utc>, String> {
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return local_day_start(date).ok_or_else(|| "no such local date".to_string());
    }
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|_| "expected YYYY-MM-DD or an RFC 3339 timestamp".to_string())
}

/// Parse a `--min-cost` threshold. Shared by clap's `--min-cost` and
/// the TUI filter editor, so a value one surface accepts is a value
/// the other accepts.
///
/// Rejects non-finite and negative values. `NaN` is the motivating
/// case: `ThresholdsFilter::matches` compares `cost >= NaN`, which is
/// false for every row, so `--min-cost nan` would commit as a filter
/// that silently empties every view instead of reporting an error.
///
/// # Errors
/// Returns a human-readable message when the text is unparseable,
/// non-finite, or below zero.
pub fn parse_min_cost(s: &str) -> Result<f64, String> {
    let value = f64::from_str(s).map_err(|_| "expected a number".to_string())?;
    if !value.is_finite() {
        return Err("expected a finite number".to_string());
    }
    if value < 0.0 || (value == 0.0 && value.is_sign_negative()) {
        // `-0.0` is not `< 0.0`, but it renders as `--min-cost -0`,
        // which clap rejects as a flag before this parser ever sees
        // it — the one input that would not survive its own round trip.
        return Err("expected a number at or above zero".to_string());
    }
    Ok(value)
}

/// Render an instant in the shortest spelling that
/// `parse_filter_datetime` maps back to it: a bare `2026-04-15` when
/// that date starts at exactly this instant locally, otherwise a full
/// RFC 3339 timestamp at the local offset. Never a reinterpretation — emitting a bare date for any
/// other instant would silently widen the filter on the next commit.
///
/// Defined as the literal inverse rather than by re-deriving the rule:
/// the candidate is offered to `parse_filter_datetime` and kept only
/// if it round-trips. The two cannot drift apart, whatever the local
/// zone does with midnight.
#[must_use]
pub fn render_filter_datetime(dt: DateTime<Utc>) -> String {
    let bare = dt.with_timezone(&Local).format("%Y-%m-%d").to_string();
    if parse_filter_datetime(&bare) == Ok(dt) {
        bare
    } else {
        // Local, like the bare form and like every timestamp cclens
        // displays. Reparses to the same instant either way, so this
        // is a spelling choice, not a semantic one.
        dt.with_timezone(&Local).to_rfc3339()
    }
}

/// Characters that survive a paste into a shell unquoted and un-glob-
/// expanded. Everything else — whitespace, quotes, `$`, backtick, `|`,
/// `&`, `;`, `*`, `(`, `)`, `\`, `!`, `#`, `~`, redirection — is
/// something the shell would act on.
const SHELL_SAFE_PUNCTUATION: &str = "-_./:@+=,";

/// Wrap a filter value in single quotes unless every character is
/// known safe to paste. Every surface renders components flag-shaped
/// and claims they reparse; an allowlist is what makes that true for
/// values the author never anticipated, where a denylist only covers
/// the metacharacters someone remembered to name.
#[must_use]
pub fn quote_filter_value(value: &str) -> String {
    let is_safe = |c: char| c.is_ascii_alphanumeric() || SHELL_SAFE_PUNCTUATION.contains(c);
    if !value.is_empty() && value.chars().all(is_safe) {
        return value.to_string();
    }
    // The POSIX escape for a single quote inside single quotes: close,
    // emit an escaped quote, reopen.
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ThresholdsFilter {
    pub min_tokens: Option<u64>,
    pub min_cost: Option<f64>,
}

impl ThresholdsFilter {
    /// Returns true iff `(tokens, cost)` clears every active threshold.
    /// `cost == None` (unknown model / unpriceable) fails any active
    /// `--min-cost` check; absent thresholds always pass. The two
    /// `is_none_or` calls collapse "no threshold OR threshold met" into
    /// one line each — the boolean `&&` then ANDs the per-axis decisions
    /// into a logical conjunction.
    #[must_use]
    pub fn matches(&self, tokens: u64, cost: Option<f64>) -> bool {
        let tokens_ok = self.min_tokens.is_none_or(|t| tokens >= t);
        let cost_ok = self.min_cost.is_none_or(|c| cost.is_some_and(|n| n >= c));
        tokens_ok && cost_ok
    }

    /// The active threshold flags, in flag order, as components that
    /// constrain every view. Cost is formatted with `{c}` — Rust's
    /// shortest round-trip float formatter — so a small threshold like
    /// `--min-cost 0.0001` round-trips; `{:.2}` would truncate it to
    /// `--min-cost 0.00`.
    #[must_use]
    pub fn describe_active(&self) -> Vec<FilterComponent> {
        let mut components = Vec::new();
        if let Some(t) = self.min_tokens {
            components.push(FilterComponent {
                text: format!("--min-tokens {t}"),
                honored_by: HonoredBy::EVERY_LOADER,
            });
        }
        if let Some(c) = self.min_cost {
            components.push(FilterComponent {
                text: format!("--min-cost {c}"),
                honored_by: HonoredBy::EVERY_LOADER,
            });
        }
        components
    }
}

/// Session-scope predicate covering `--project` / `--since` / `--until`.
///
/// `accepts` takes borrowed primitives instead of `&Session` /
/// `&SessionMeta` so the predicate stays decoupled from either domain
/// type — both `run_list` and `InputsFilter::accepts` delegate here
/// using fields they already carry. Inclusive bounds at both ends.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionFilter {
    pub project_name: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
}

impl SessionFilter {
    /// True iff `(project_short_name, started_at)` clears every active
    /// scope filter. Project name is exact case-sensitive equality;
    /// `since` and `until` are inclusive on both ends.
    #[must_use]
    pub fn accepts(&self, project_short_name: &str, started_at: DateTime<Utc>) -> bool {
        if let Some(name) = &self.project_name
            && project_short_name != *name
        {
            return false;
        }
        if let Some(since) = &self.since
            && started_at < *since
        {
            return false;
        }
        if let Some(until) = &self.until
            && started_at > *until
        {
            return false;
        }
        true
    }

    /// True iff at least one scope filter is active. Used by the empty-
    /// result hint to suppress the note when no filter explains an
    /// empty render.
    #[must_use]
    pub fn any_active(&self) -> bool {
        self.project_name.is_some() || self.since.is_some() || self.until.is_some()
    }

    /// The active scope flags, in flag order, as components that
    /// constrain every view. Dates render through
    /// `render_filter_datetime`, so what a surface displays always
    /// reparses to what the filter holds.
    #[must_use]
    pub fn describe_active(&self) -> Vec<FilterComponent> {
        let mut components = Vec::new();
        if let Some(p) = &self.project_name {
            components.push(FilterComponent {
                text: format!("--project {}", quote_filter_value(p)),
                honored_by: HonoredBy::SESSION_SCOPED,
            });
        }
        if let Some(s) = self.since {
            components.push(FilterComponent {
                text: format!("--since {}", render_filter_datetime(s)),
                honored_by: HonoredBy::SESSION_SCOPED,
            });
        }
        if let Some(u) = self.until {
            components.push(FilterComponent {
                text: format!("--until {}", render_filter_datetime(u)),
                honored_by: HonoredBy::SESSION_SCOPED,
            });
        }
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    #[test]
    fn matches_passes_when_no_thresholds_active() {
        let t = ThresholdsFilter::default();
        assert!(t.matches(0, None));
        assert!(t.matches(0, Some(0.0)));
        assert!(t.matches(1_000_000, Some(1.0)));
    }

    #[test]
    fn matches_min_tokens_only_passes_at_or_above_threshold() {
        let t = ThresholdsFilter {
            min_tokens: Some(100),
            min_cost: None,
        };
        // Boundary: at threshold passes (>=).
        assert!(t.matches(100, None));
        assert!(t.matches(100, Some(0.0)));
        // Above threshold passes regardless of cost.
        assert!(t.matches(101, None));
    }

    #[test]
    fn matches_min_tokens_only_fails_below_threshold() {
        let t = ThresholdsFilter {
            min_tokens: Some(100),
            min_cost: None,
        };
        assert!(!t.matches(99, Some(1.0)));
        assert!(!t.matches(0, None));
    }

    #[test]
    fn matches_min_cost_only_fails_when_cost_unknown() {
        // None cost is excluded by any active --min-cost, even a
        // 0.0 threshold (None means "unpriceable", which can't clear a
        // cost gate).
        let t = ThresholdsFilter {
            min_tokens: None,
            min_cost: Some(0.50),
        };
        assert!(!t.matches(1_000_000, None));

        let zero = ThresholdsFilter {
            min_tokens: None,
            min_cost: Some(0.0),
        };
        assert!(!zero.matches(0, None));
    }

    #[test]
    fn matches_min_cost_only_passes_at_or_above_threshold() {
        let t = ThresholdsFilter {
            min_tokens: None,
            min_cost: Some(0.50),
        };
        // Boundary: at threshold passes.
        assert!(t.matches(0, Some(0.50)));
        assert!(t.matches(1_000_000, Some(0.50)));
        // Above threshold passes.
        assert!(t.matches(0, Some(0.51)));
        // Below threshold fails.
        assert!(!t.matches(1_000_000, Some(0.49)));
        // Some(0.0) is excluded by any positive threshold.
        assert!(!t.matches(1_000_000, Some(0.0)));
        // ...but accepted by a 0.0 threshold.
        let zero = ThresholdsFilter {
            min_tokens: None,
            min_cost: Some(0.0),
        };
        assert!(zero.matches(0, Some(0.0)));
    }

    #[test]
    fn matches_both_thresholds_logical_and() {
        let t = ThresholdsFilter {
            min_tokens: Some(100),
            min_cost: Some(0.10),
        };
        // Both clear → pass.
        assert!(t.matches(100, Some(0.10)));
        assert!(t.matches(500, Some(1.00)));
        // Tokens fail, cost clears → fail.
        assert!(!t.matches(50, Some(1.00)));
        // Tokens clear, cost fails → fail.
        assert!(!t.matches(500, Some(0.05)));
        // Tokens clear, cost is None → fail.
        assert!(!t.matches(500, None));
        // Both fail → fail.
        assert!(!t.matches(0, None));
    }

    #[test]
    fn session_filter_passes_when_no_fields_active() {
        let f = SessionFilter::default();
        assert!(f.accepts("alpha", ts("2026-04-01T10:00:00Z")));
        assert!(f.accepts("", ts("1970-01-01T00:00:00Z")));
    }

    #[test]
    fn session_filter_project_name_exact_match_only() {
        let f = SessionFilter {
            project_name: Some("alpha".to_string()),
            ..Default::default()
        };
        let now = ts("2026-04-01T10:00:00Z");
        assert!(f.accepts("alpha", now));
        assert!(!f.accepts("beta", now));
        // Case-sensitive — "Alpha" != "alpha".
        assert!(!f.accepts("Alpha", now));
    }

    #[test]
    fn session_filter_since_inclusive_at_boundary() {
        let f = SessionFilter {
            since: Some(ts("2026-04-15T14:33:00Z")),
            ..Default::default()
        };
        assert!(f.accepts("any", ts("2026-04-15T14:33:00Z")));
        assert!(!f.accepts("any", ts("2026-04-15T14:32:59Z")));
        assert!(f.accepts("any", ts("2026-04-15T14:33:01Z")));
    }

    #[test]
    fn session_filter_until_inclusive_at_boundary() {
        let f = SessionFilter {
            until: Some(ts("2026-04-15T14:33:00Z")),
            ..Default::default()
        };
        assert!(f.accepts("any", ts("2026-04-15T14:33:00Z")));
        assert!(f.accepts("any", ts("2026-04-15T14:32:59Z")));
        assert!(!f.accepts("any", ts("2026-04-15T14:33:01Z")));
    }

    #[test]
    fn session_filter_combined_logical_and() {
        let f = SessionFilter {
            project_name: Some("alpha".to_string()),
            since: Some(ts("2026-04-10T00:00:00Z")),
            until: Some(ts("2026-04-20T00:00:00Z")),
        };
        // Project mismatch alone fails.
        assert!(!f.accepts("beta", ts("2026-04-15T00:00:00Z")));
        // Project match + date out of range fails.
        assert!(!f.accepts("alpha", ts("2026-04-09T23:59:59Z")));
        assert!(!f.accepts("alpha", ts("2026-04-20T00:00:01Z")));
        // Project match + date in range passes.
        assert!(f.accepts("alpha", ts("2026-04-15T00:00:00Z")));
        assert!(f.accepts("alpha", ts("2026-04-10T00:00:00Z")));
        assert!(f.accepts("alpha", ts("2026-04-20T00:00:00Z")));
    }

    #[test]
    fn session_filter_any_active_reflects_field_state() {
        assert!(!SessionFilter::default().any_active());
        assert!(
            SessionFilter {
                project_name: Some("x".to_string()),
                ..Default::default()
            }
            .any_active()
        );
        assert!(
            SessionFilter {
                since: Some(ts("2026-04-01T00:00:00Z")),
                ..Default::default()
            }
            .any_active()
        );
        assert!(
            SessionFilter {
                until: Some(ts("2026-04-01T00:00:00Z")),
                ..Default::default()
            }
            .any_active()
        );
    }

    #[test]
    fn parse_filter_datetime_resolves_a_bare_date_to_the_local_day_start() {
        // Local, not UTC: every timestamp cclens renders is local, so
        // a UTC reading drops rows the table shows on the named day.
        // Asserted as a property rather than a fixed instant, so the
        // test says the same thing in every zone it runs in.
        let parsed = parse_filter_datetime("2026-04-15").unwrap();
        let local = parsed.with_timezone(&Local);
        assert_eq!(local.format("%Y-%m-%d").to_string(), "2026-04-15");
        let one_earlier = (parsed - chrono::Duration::seconds(1)).with_timezone(&Local);
        assert_eq!(
            one_earlier.format("%Y-%m-%d").to_string(),
            "2026-04-14",
            "the instant before must fall on the previous local day",
        );
    }

    #[test]
    fn parse_filter_datetime_converts_non_utc_offset_to_utc() {
        // 09:30 at +02:00 is 07:30Z — the parser normalizes rather
        // than preserving the offset, so downstream comparisons are
        // always against a single timeline.
        assert_eq!(
            parse_filter_datetime("2026-04-15T09:30:00+02:00").unwrap(),
            ts("2026-04-15T07:30:00Z"),
        );
    }

    #[test]
    fn parse_filter_datetime_rejects_unparseable_text() {
        assert!(parse_filter_datetime("last tuesday").is_err());
        assert!(parse_filter_datetime("2026-13-45").is_err());
        assert!(parse_filter_datetime("").is_err());
    }

    #[test]
    fn filter_datetime_round_trips_through_shortest_spelling() {
        // The invariant the editor and the header both depend on:
        // whatever a surface displays reparses to the same instant.
        let day_start = parse_filter_datetime("2026-04-15").unwrap();
        assert_eq!(render_filter_datetime(day_start), "2026-04-15");

        let mid_day = day_start + chrono::Duration::minutes(893);
        let rendered = render_filter_datetime(mid_day);
        assert_ne!(
            rendered, "2026-04-15",
            "only the day start may render bare; anything else would widen the filter",
        );
        assert_eq!(parse_filter_datetime(&rendered).unwrap(), mid_day);
    }

    #[test]
    fn quote_filter_value_quotes_only_what_would_not_reparse() {
        assert_eq!(quote_filter_value("alpha"), "alpha");
        assert_eq!(quote_filter_value("my project"), "'my project'");
        assert_eq!(quote_filter_value(""), "''");
        assert_eq!(quote_filter_value("it's"), r"'it'\''s'");
        // An allowlist, so metacharacters no one thought to name are
        // quoted too — pasting these unquoted would run a second
        // command, expand a glob, or substitute a subshell.
        assert_eq!(quote_filter_value("my;project"), "'my;project'");
        assert_eq!(quote_filter_value("$(id)"), "'$(id)'");
        assert_eq!(quote_filter_value("a*b"), "'a*b'");
        // ...while the punctuation real project names carry does not
        // get quoted for nothing.
        assert_eq!(quote_filter_value("my-project_2.0"), "my-project_2.0");
    }

    #[test]
    fn session_filter_describe_active_quotes_a_project_name_with_a_space() {
        // The hint is meant to be copy-pasteable; unquoted, this one
        // reparses as `--project my` plus a stray argument.
        let spaced = SessionFilter {
            project_name: Some("my project".to_string()),
            since: None,
            until: None,
        };
        assert_eq!(spaced.describe_active()[0].text, "--project 'my project'");
    }

    #[test]
    fn thresholds_describe_active_emits_components_in_flag_order() {
        assert!(ThresholdsFilter::default().describe_active().is_empty());

        let both = ThresholdsFilter {
            min_tokens: Some(50_000),
            min_cost: Some(0.5),
        };
        let components = both.describe_active();
        assert_eq!(
            components
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            vec!["--min-tokens 50000", "--min-cost 0.5"],
        );
        assert!(
            components
                .iter()
                .all(|c| c.honored_by == HonoredBy::EVERY_LOADER)
        );

        let tokens_only = ThresholdsFilter {
            min_tokens: Some(50_000),
            min_cost: None,
        };
        assert_eq!(tokens_only.describe_active().len(), 1);
    }

    #[test]
    fn thresholds_describe_active_renders_small_cost_without_truncation() {
        // `{:.2}` would render this as `--min-cost 0.00`, which
        // reparses to a different filter than the one in effect.
        let small = ThresholdsFilter {
            min_tokens: None,
            min_cost: Some(0.0001),
        };
        assert_eq!(small.describe_active()[0].text, "--min-cost 0.0001");
    }

    #[test]
    fn session_filter_describe_active_emits_components_in_flag_order() {
        assert!(SessionFilter::default().describe_active().is_empty());

        let all = SessionFilter {
            project_name: Some("alpha".to_string()),
            // Built through the parser: the instant that renders bare
            // is the local day start, which is not midnight UTC in
            // every zone this test runs in.
            since: parse_filter_datetime("2026-04-10").ok(),
            until: Some(ts("2026-04-20T14:33:00Z")),
        };
        let components = all.describe_active();
        assert_eq!(
            components
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            vec![
                "--project alpha",
                // The local day start renders bare; any other instant
                // renders as a full timestamp at the local offset.
                "--since 2026-04-10",
                &format!(
                    "--until {}",
                    ts("2026-04-20T14:33:00Z")
                        .with_timezone(&Local)
                        .to_rfc3339()
                ),
            ],
        );
        // `load_show` is handed one session id, so a scope filter
        // cannot be what emptied it.
        assert!(
            components
                .iter()
                .all(|c| c.honored_by == HonoredBy::SESSION_SCOPED)
        );
    }

    #[test]
    fn parse_min_cost_accepts_zero_and_small_positive_values() {
        // Compared as `Option<f64>` rather than bare floats: these
        // are exact round-trips of a parse, not computed values, and
        // `float_cmp` fires on the bare form.
        assert_eq!(parse_min_cost("0").ok(), Some(0.0));
        assert_eq!(parse_min_cost("0.0001").ok(), Some(0.0001));
        assert_eq!(parse_min_cost("50").ok(), Some(50.0));
    }

    #[test]
    fn parse_min_cost_rejects_non_finite_and_negative() {
        // `NaN` would otherwise commit as a filter that empties every
        // view, since `cost >= NaN` is false for every row.
        assert!(parse_min_cost("nan").is_err());
        assert!(parse_min_cost("inf").is_err());
        assert!(parse_min_cost("-1").is_err());
        assert!(parse_min_cost("abc").is_err());
    }
}
