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

use chrono::{DateTime, Utc};

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
}
