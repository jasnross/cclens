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
//! - `format_tokens(u64) -> String` — compact `0.12k` / `999.99k`
//!   token count
//! - `kind_label(&ContextFileKind) -> String` — context-file kind to
//!   display label
//!
//! These helpers are consumed by both `rendering` (comfy-table
//! stop-gap) and `tui` (ratatui interactive renderer).

use std::path::Path;

use chrono::{DateTime, Utc};

use crate::attribution::{CoverageStats, TierCoverage};
use crate::inventory::ContextFileKind;

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
