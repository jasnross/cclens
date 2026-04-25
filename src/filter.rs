//! Threshold-pair filter primitive shared by the binary's CLI layer and
//! the library's rendering layer.
//!
//! Public API:
//! - `Thresholds` — value type holding `(min_tokens, min_cost)`.
//! - `Thresholds::matches` — predicate over `(tokens, cost)`.

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Thresholds {
    pub min_tokens: Option<u64>,
    pub min_cost: Option<f64>,
}

impl Thresholds {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_passes_when_no_thresholds_active() {
        let t = Thresholds::default();
        assert!(t.matches(0, None));
        assert!(t.matches(0, Some(0.0)));
        assert!(t.matches(1_000_000, Some(1.0)));
    }

    #[test]
    fn matches_min_tokens_only_passes_at_or_above_threshold() {
        let t = Thresholds {
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
        let t = Thresholds {
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
        let t = Thresholds {
            min_tokens: None,
            min_cost: Some(0.50),
        };
        assert!(!t.matches(1_000_000, None));

        let zero = Thresholds {
            min_tokens: None,
            min_cost: Some(0.0),
        };
        assert!(!zero.matches(0, None));
    }

    #[test]
    fn matches_min_cost_only_passes_at_or_above_threshold() {
        let t = Thresholds {
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
        let zero = Thresholds {
            min_tokens: None,
            min_cost: Some(0.0),
        };
        assert!(zero.matches(0, Some(0.0)));
    }

    #[test]
    fn matches_both_thresholds_logical_and() {
        let t = Thresholds {
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
}
