//! Display-agnostic domain types for the cclens pipeline.
//!
//! Public API:
//! - `Session` — aggregated per-session record (id, project, totals).
//! - `Turn` — single-line record (role, model, usage, content, cwd).
//! - `Role` — typed enum over Claude Code's `.type` field.
//! - `Usage` — token counts; `billable()` excludes `cache_read`.
//! - `CacheCreation` — 5m / 1h ephemeral split.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct Session {
    pub id: String,
    pub project_short_name: String,
    pub started_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub title: String,
    pub turns: Vec<Turn>,
    pub total_billable: u64,
    /// `Some(sum)` only if every assistant turn's cost resolved to
    /// `Some(_)`. A single unknown-model assistant turn collapses the
    /// whole session to `None` (strict propagation, no partial sums)
    /// — the rendered `cost` cell becomes `—`. Diverges from
    /// `total_billable` by including `cache_read` tokens.
    pub total_cost: Option<f64>,
}

impl Session {
    #[must_use]
    pub fn duration(&self) -> chrono::Duration {
        self.last_activity - self.started_at
    }
}

#[derive(Debug, Serialize)]
pub struct Turn {
    pub timestamp: Option<DateTime<Utc>>,
    pub role: Role,
    pub model: Option<String>,
    pub message_id: Option<String>,
    pub request_id: Option<String>,
    pub usage: Option<Usage>,
    pub content: Option<Value>,
    pub cwd: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
pub enum Role {
    User,
    Assistant,
    Attachment,
    System,
    Other(String),
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_creation: CacheCreation,
    pub cache_read: u64,
}

impl Usage {
    #[must_use]
    pub fn billable(&self) -> u64 {
        self.input + self.output + self.cache_creation.total()
    }
}

/// Cache-creation token counts split by ephemeral lifetime. The 5m and
/// 1h buckets are billed at distinct rates (the 1h rate is roughly
/// 1.6× the 5m rate for opus-4-7), so cclens keeps them separate from
/// `RawUsage` through to `pricing::cost_for_components`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CacheCreation {
    pub ephemeral_5m: u64,
    pub ephemeral_1h: u64,
}

impl CacheCreation {
    #[must_use]
    pub fn total(&self) -> u64 {
        self.ephemeral_5m + self.ephemeral_1h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_billable_sums_three_fields() {
        let u = Usage {
            input: 6,
            output: 1186,
            cache_creation: CacheCreation {
                ephemeral_5m: 18998,
                ephemeral_1h: 0,
            },
            cache_read: 17317,
        };
        assert_eq!(u.billable(), 20190);
    }
}
