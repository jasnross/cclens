//! Shared formatting helpers for human-readable display values.
//!
//! Public API:
//! - `format_cost_opt(Option<f64>) -> String` — cost with `$` prefix
//!   and 4 decimal places, or `—` for `None`
//! - `format_local(DateTime<Utc>) -> String` — `YYYY-MM-DD HH:MM`
//!   in the system's local timezone
//! - `format_tokens(u64) -> String` — compact `0.12k` / `999.99k`
//!   token count
//!
//! These helpers are consumed by both `rendering` (comfy-table
//! stop-gap) and `tui` (ratatui interactive renderer).

use chrono::{DateTime, Utc};

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
pub fn format_tokens(count: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let count_float = count as f64;
    format!("{:.2}k", count_float / 1000.0)
}
