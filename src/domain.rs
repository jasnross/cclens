//! Display-agnostic domain types for the cclens pipeline.
//!
//! Public API:
//! - `Session` — aggregated per-session record (id, project, totals).
//! - `Turn` — single-line record (role, model, usage, content, cwd,
//!   origin).
//! - `Role` — typed enum over Claude Code's `.type` field.
//! - `TurnOrigin` — discriminates a parent-session turn (`Parent`) from
//!   a subagent-transcript turn (`Subagent { agent_type, description }`).
//!   `description` carries the per-invocation summary from the
//!   `.meta.json` sidecar so the show renderer can disambiguate
//!   multiple invocations of the same `agent_type`.
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
    /// Source-provenance: `Parent` for turns parsed from the parent
    /// session's JSONL, `Subagent { agent_type, description }` for
    /// turns parsed from a subagent transcript. Defaults to `Parent`
    /// so existing turn-construction sites (parser, tests) don't
    /// need to mention it explicitly; the subagent-specific
    /// origin is set in `build_subagent_turns` after
    /// `parse_jsonl` returns.
    pub origin: TurnOrigin,
}

#[derive(Debug, Serialize)]
pub enum Role {
    User,
    Assistant,
    Attachment,
    System,
    Other(String),
}

/// Discriminator on each `Turn` so downstream consumers
/// (`render_session`, exchange-grouping, dispatch) can tell parent
/// from subagent without out-of-band info. The `Subagent` payload
/// carries both the `agent_type` (matching the `.meta.json`
/// sidecar's `agentType`) and the per-invocation `description`,
/// which the show renderer uses to disambiguate multiple
/// invocations of the same agent type.
#[derive(Debug, Clone, Default, Serialize)]
pub enum TurnOrigin {
    #[default]
    Parent,
    Subagent {
        agent_type: String,
        description: Option<String>,
    },
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

    #[test]
    fn turn_origin_default_is_parent() {
        // The `#[default]` derive lets existing turn-construction sites
        // build `Turn`s without naming the field explicitly. Pin the
        // chosen default so accidentally flipping it (or removing the
        // attribute) becomes a test failure.
        let origin: TurnOrigin = TurnOrigin::default();
        match origin {
            TurnOrigin::Parent => {}
            TurnOrigin::Subagent { .. } => panic!("default TurnOrigin must be Parent"),
        }
    }
}
