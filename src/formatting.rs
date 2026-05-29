//! Shared formatting helpers for human-readable display values.
//!
//! Public API:
//! - `coverage_line(CoverageStats) -> String` — per-tier coverage line
//! - `coverage_half(&str, &TierCoverage) -> String` — single-tier half
//! - `display_path(&Path) -> String` — home-dir replacement without
//!   truncation
//! - `format_cost_opt(Option<f64>) -> String` — cost with `$` prefix
//!   and 4 decimal places, or `—` for `None`
//! - `format_local(DateTime<Utc>) -> String` — `YYYY-MM-DD HH:MM`
//!   in the system's local timezone
//! - `format_local_or_empty(Option<DateTime<Utc>>) -> String` —
//!   delegates to `format_local` or returns the empty string on `None`
//! - `format_rate_mtok(f64) -> String` — per-token rate to `$/MTok`
//!   display string
//! - `format_tokens(u64) -> String` — compact `0.12k` / `999.99k`
//!   token count
//! - `kind_label(&ContextFileKind) -> String` — context-file kind to
//!   display label
//! - `tiers_differ(&ClaudePricing) -> bool` — whether any rate's
//!   first-200k and above-200k tiers differ
//!
//! These helpers are consumed primarily by `views` (shared
//! view builders), with `rendering` and `tui` importing
//! directly for presentation-specific needs.

use std::path::Path;

use chrono::{DateTime, Utc};

use crate::attribution::{CoverageStats, TierCoverage};
use crate::inventory::ContextFileKind;
use crate::pricing::ClaudePricing;

#[must_use]
pub fn format_cost_opt(c: Option<f64>) -> String {
    c.map_or_else(|| "—".to_string(), |n| format!("${n:.4}"))
}

#[must_use]
pub fn format_local(ts: DateTime<Utc>) -> String {
    ts.with_timezone(&chrono::Local)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

#[must_use]
pub fn format_local_or_empty(ts: Option<DateTime<Utc>>) -> String {
    ts.map_or_else(String::new, format_local)
}

#[must_use]
pub fn format_tokens(count: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let count_float = count as f64;
    format!("{:.2}k", count_float / 1000.0)
}

#[must_use]
pub fn display_path(path: &Path) -> String {
    if let Some(home) = dirs::home_dir()
        && let Ok(rel) = path.strip_prefix(&home)
    {
        format!("~/{}", rel.display())
    } else {
        path.display().to_string()
    }
}

/// Enumerates every variant (no wildcard) so adding a new variant
/// forces a label decision at compile time via `wildcard_enum_match_arm`.
#[must_use]
pub fn kind_label(kind: &ContextFileKind) -> String {
    match kind {
        ContextFileKind::GlobalClaudeMd => "global".to_string(),
        ContextFileKind::UserRule => "rule".to_string(),
        ContextFileKind::UserSkill => "skill".to_string(),
        ContextFileKind::UserAgent => "agent".to_string(),
        ContextFileKind::PluginSkill { plugin, .. } => format!("plugin:{plugin}:skill"),
        ContextFileKind::PluginRule { plugin, .. } => format!("plugin:{plugin}:rule"),
        ContextFileKind::PluginAgent { plugin, .. } => format!("plugin:{plugin}:agent"),
        ContextFileKind::ProjectClaudeMd => "project".to_string(),
        ContextFileKind::ProjectLocalSkill => "project:skill".to_string(),
        ContextFileKind::ProjectLocalCommand => "project:command".to_string(),
        ContextFileKind::ProjectLocalRule => "project:rule".to_string(),
        ContextFileKind::ProjectLocalAgent => "project:agent".to_string(),
    }
}

#[must_use]
pub fn coverage_line(coverage: &CoverageStats) -> String {
    let one_h = coverage_half("1h", &coverage.long_1h);
    let five_m = coverage_half("5m", &coverage.short_5m);
    format!("coverage: {one_h} | {five_m}")
}

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn coverage_half(label: &str, tier: &TierCoverage) -> String {
    match tier.ratio {
        None => format!("{label}: n/a"),
        Some(r) => format!(
            "{label}: {pct:.1}% ({attributed} / {observed} {label}-tokens)",
            pct = r * 100.0,
            attributed = tier.attributed_tokens,
            observed = tier.observed_tokens,
        ),
    }
}

#[must_use]
pub fn format_rate_mtok(per_token_rate: f64) -> String {
    format!("${:.2}", per_token_rate * 1_000_000.0)
}

#[allow(clippy::float_cmp)]
#[must_use]
pub fn tiers_differ(pricing: &ClaudePricing) -> bool {
    let rates = [
        &pricing.input,
        &pricing.output,
        &pricing.cache_read,
        &pricing.cache_creation_5m,
        &pricing.cache_creation_1h,
    ];
    rates.iter().any(|r| r.first_200k_rate != r.above_200k_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::TieredRate;

    fn uniform_pricing(rate: f64) -> ClaudePricing {
        let tier = TieredRate {
            first_200k_rate: rate,
            above_200k_rate: rate,
        };
        ClaudePricing {
            input: tier,
            output: tier,
            cache_creation_5m: tier,
            cache_creation_1h: tier,
            cache_read: tier,
        }
    }

    fn split_pricing() -> ClaudePricing {
        ClaudePricing {
            input: TieredRate {
                first_200k_rate: 3e-6,
                above_200k_rate: 6e-6,
            },
            output: TieredRate {
                first_200k_rate: 15e-6,
                above_200k_rate: 22.5e-6,
            },
            cache_read: TieredRate {
                first_200k_rate: 0.3e-6,
                above_200k_rate: 0.6e-6,
            },
            cache_creation_5m: TieredRate {
                first_200k_rate: 3.75e-6,
                above_200k_rate: 7.5e-6,
            },
            cache_creation_1h: TieredRate {
                first_200k_rate: 3e-6,
                above_200k_rate: 6e-6,
            },
        }
    }

    #[test]
    fn format_rate_mtok_converts_per_token_to_dollars_per_million() {
        assert_eq!(format_rate_mtok(3e-6), "$3.00");
        assert_eq!(format_rate_mtok(15e-6), "$15.00");
        assert_eq!(format_rate_mtok(0.3e-6), "$0.30");
        assert_eq!(format_rate_mtok(3.75e-6), "$3.75");
        assert_eq!(format_rate_mtok(0.0), "$0.00");
    }

    #[test]
    fn tiers_differ_returns_false_for_uniform_rates() {
        assert!(!tiers_differ(&uniform_pricing(3e-6)));
    }

    #[test]
    fn tiers_differ_returns_true_when_any_rate_differs() {
        assert!(tiers_differ(&split_pricing()));
        let mut only_output_differs = uniform_pricing(3e-6);
        only_output_differs.output.above_200k_rate = 6e-6;
        assert!(tiers_differ(&only_output_differs));
    }

    // --- display_path ---

    #[test]
    fn display_path_replaces_home_with_tilde() {
        use std::path::PathBuf;

        let Some(home) = dirs::home_dir() else {
            return;
        };
        let under_home = home.join("foo/bar");
        let displayed = display_path(&under_home);
        assert!(
            displayed.starts_with("~/"),
            "expected ~/ prefix, got: {displayed}",
        );
        let elsewhere = PathBuf::from("/var/tmp/elsewhere.md");
        let displayed = display_path(&elsewhere);
        assert!(
            displayed.starts_with('/'),
            "non-home path should render absolute, got: {displayed}",
        );
    }
}
