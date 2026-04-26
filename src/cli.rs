//! clap-derived CLI surface for the `cclens` binary.
//!
//! Public to the binary only (declared as `mod cli;` in `main.rs` rather
//! than in `lib.rs`) so that no library module can import clap-derived
//! types — keeps the library/CLI seam visible.
//!
//! Public API (binary-internal):
//! - `Cli` / `Command` / `PricingAction` — clap parser types.
//! - `FilterArgs` — flattened threshold flags; `.thresholds()` produces
//!   a library `Thresholds`.
//! - `InputsFilterArgs` — flattened session/project/since/until flags
//!   for the `inputs` subcommand; `.attribution_filter()` produces a
//!   library `AttributionFilter`.
//! - `emit_empty_result_hint(&FilterArgs)` — stderr hint used by
//!   `run_list` and `run_show` when a filter dropped every row.
//! - `emit_inputs_empty_hint(&InputsFilterArgs, &FilterArgs)` —
//!   sibling hint that describes both the inputs-side and threshold
//!   filters when `cclens inputs` produces no rows.

use std::path::PathBuf;

use cclens::attribution::AttributionFilter;
use cclens::filter::Thresholds;
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "cclens",
    about = "Browse Claude Code conversations (tokens + cost)"
)]
#[command(version)]
pub(super) struct Cli {
    /// Directory to scan for project conversations.
    #[arg(long, default_value_os_t = default_projects_dir())]
    pub(super) projects_dir: PathBuf,

    #[command(subcommand)]
    pub(super) command: Option<Command>,
}

#[derive(Subcommand)]
pub(super) enum Command {
    /// List sessions (default).
    ///
    /// The `cost` column includes `cache_read` tokens (priced at the
    /// discounted cache-read rate) — the `tokens` column does not.
    List {
        #[command(flatten)]
        filters: FilterArgs,
    },
    /// Show per-exchange token + cost breakdown for one session.
    ///
    /// Per-row `cost` and running `cum_cost` columns include
    /// `cache_read` tokens; the `tokens` and `cumulative` columns do
    /// not. A row's `cost` cell renders `—` when its model is unknown
    /// to the pricing catalog; once an unknown-model row appears,
    /// every subsequent `cum_cost` cell also renders `—`.
    Show {
        /// Full session UUID (matches a .jsonl filename stem under --projects-dir).
        session_id: String,
        #[command(flatten)]
        filters: FilterArgs,
    },
    /// Manage the pricing catalog cache.
    Pricing {
        #[command(subcommand)]
        action: PricingAction,
    },
    /// Rank user-controlled context files by attributed cache-creation cost.
    ///
    /// Walks `~/.claude/{CLAUDE.md,rules,skills,agents}`, the plugin
    /// cache, and per-session ancestor + project-local context, then
    /// attributes each file's tokens to the matching-tier
    /// `cache_creation_*` events observed in the JSONL stream.
    /// Long-tier files (`CLAUDE.md`, rules, agents) bill at the
    /// 1h cache-creation rate; on-demand files (skills, commands) bill
    /// at the 5m rate. The `attributed_cost` column is the per-file
    /// estimate; the rendered footer shows per-tier coverage (how
    /// much of the observed cache-creation tokens are explained by
    /// user-attributable files).
    Inputs {
        #[command(flatten)]
        inputs_filters: InputsFilterArgs,
        #[command(flatten)]
        filters: FilterArgs,
    },
}

/// Shared `--min-tokens` / `--min-cost` thresholds for `list` and
/// `show`. Flattened into both subcommands via `#[command(flatten)]`;
/// deliberately not flattened into `pricing` (so `pricing refresh
/// --min-tokens 1` is a clap parse error, not a silent no-op).
///
/// The `Copy` bound matters: a threshold pair is two scalar `Option`s,
/// and copying them around the renderer avoids borrow plumbing without
/// adding a closure.
#[derive(Args, Debug, Clone, Copy, Default)]
pub(super) struct FilterArgs {
    /// Show only rows with at least N billable tokens (e.g. --min-tokens 50000)
    #[arg(long)]
    min_tokens: Option<u64>,
    /// Show only rows costing at least USD, e.g. --min-cost 0.50; unknown-cost rows excluded
    #[arg(long)]
    min_cost: Option<f64>,
}

impl FilterArgs {
    /// Project the clap-derived flags into the library-side `Thresholds`.
    /// `FilterArgs` is binary-only (clap-derived); `Thresholds` lives in
    /// the library crate and is what `render_session` and the library's
    /// session-level filter take — keeping the library/CLI seam free of
    /// clap dependencies.
    pub(super) fn thresholds(&self) -> Thresholds {
        Thresholds {
            min_tokens: self.min_tokens,
            min_cost: self.min_cost,
        }
    }

    /// True iff at least one filter flag is active. Used to gate the
    /// empty-result stderr hint — when no filter is active, an empty
    /// result is just an empty `projects_dir` and gets no hint.
    fn any_active(&self) -> bool {
        self.min_tokens.is_some() || self.min_cost.is_some()
    }

    /// Format the active flags for the empty-result stderr hint:
    /// `--min-tokens 50000`, `--min-cost 0.50`, or both joined by a
    /// space. Cost is formatted with `{}` (Rust's default float
    /// formatter, shortest round-trip representation) so small
    /// thresholds like `--min-cost 0.0001` round-trip faithfully —
    /// `{:.2}` would truncate them to `--min-cost 0.00`.
    fn describe_active(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(t) = self.min_tokens {
            parts.push(format!("--min-tokens {t}"));
        }
        if let Some(c) = self.min_cost {
            parts.push(format!("--min-cost {c}"));
        }
        parts.join(" ")
    }
}

/// Emit `note: no rows matched <flags>` to stderr when filters dropped
/// every row. No-op when no filter is active so the pre-existing
/// "empty `projects_dir` produces no stderr" contract is preserved.
pub(super) fn emit_empty_result_hint(filters: &FilterArgs) {
    if !filters.any_active() {
        return;
    }
    eprintln!("note: no rows matched {}", filters.describe_active());
}

/// Session-level filters for `cclens inputs`. Mirrors `FilterArgs`'s
/// CLI-only / library-projection split: clap-derived flags here,
/// `.attribution_filter()` produces the library-side `AttributionFilter`.
#[derive(Args, Debug, Clone, Default)]
pub(super) struct InputsFilterArgs {
    /// Restrict attribution to one session by full UUID.
    #[arg(long)]
    session: Option<String>,
    /// Restrict attribution to one project (matches the short name
    /// shown in the `list` view's `project` column).
    #[arg(long)]
    project: Option<String>,
    /// Include only sessions whose `started_at` is at or after this
    /// ISO-8601 timestamp (inclusive).
    #[arg(long)]
    since: Option<DateTime<Utc>>,
    /// Include only sessions whose `started_at` is at or before this
    /// ISO-8601 timestamp (inclusive).
    #[arg(long)]
    until: Option<DateTime<Utc>>,
}

impl InputsFilterArgs {
    /// Project the clap-derived flags into the library-side
    /// `AttributionFilter`. Same library/CLI seam pattern as
    /// `FilterArgs::thresholds`.
    pub(super) fn attribution_filter(&self) -> AttributionFilter {
        AttributionFilter {
            session_id: self.session.clone(),
            project_name: self.project.clone(),
            since: self.since,
            until: self.until,
        }
    }

    fn any_active(&self) -> bool {
        self.session.is_some()
            || self.project.is_some()
            || self.since.is_some()
            || self.until.is_some()
    }

    fn describe_active(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(s) = &self.session {
            parts.push(format!("--session {s}"));
        }
        if let Some(p) = &self.project {
            parts.push(format!("--project {p}"));
        }
        if let Some(s) = &self.since {
            parts.push(format!("--since {}", s.to_rfc3339()));
        }
        if let Some(u) = &self.until {
            parts.push(format!("--until {}", u.to_rfc3339()));
        }
        parts.join(" ")
    }
}

/// Sibling of `emit_empty_result_hint` for `cclens inputs`: surfaces
/// both the inputs-side filter set and the row-level threshold flags
/// in one stderr line. Suppresses the hint when no filter is active.
pub(super) fn emit_inputs_empty_hint(inputs_filters: &InputsFilterArgs, filters: &FilterArgs) {
    let any = inputs_filters.any_active() || filters.any_active();
    if !any {
        return;
    }
    let mut combined = inputs_filters.describe_active();
    if filters.any_active() {
        if !combined.is_empty() {
            combined.push(' ');
        }
        combined.push_str(&filters.describe_active());
    }
    eprintln!("note: no rows matched {combined}");
}

#[derive(Subcommand, Clone, Copy)]
pub(super) enum PricingAction {
    /// Re-fetch the `LiteLLM` pricing catalog and overwrite the cache.
    Refresh,
    /// Print cache path, size, mtime, and Claude-entry count.
    Info,
}

fn default_projects_dir() -> PathBuf {
    dirs::home_dir().map_or_else(
        || PathBuf::from(".claude/projects"),
        |h| h.join(".claude/projects"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_invocation_leaves_command_as_none() {
        let cli = Cli::try_parse_from(["cclens"]).unwrap();
        assert!(cli.command.is_none());
    }

    #[test]
    fn explicit_list_parses_as_list_variant() {
        let cli = Cli::try_parse_from(["cclens", "list"]).unwrap();
        assert!(matches!(cli.command, Some(Command::List { .. })));
    }

    #[test]
    fn projects_dir_flag_overrides_default() {
        let cli = Cli::try_parse_from(["cclens", "--projects-dir", "/tmp/foo", "list"]).unwrap();
        assert_eq!(cli.projects_dir, PathBuf::from("/tmp/foo"));
    }

    #[test]
    fn default_projects_dir_ends_in_claude_projects() {
        let cli = Cli::try_parse_from(["cclens"]).unwrap();
        assert!(
            cli.projects_dir.ends_with(".claude/projects"),
            "expected default projects_dir to end in .claude/projects, got {:?}",
            cli.projects_dir,
        );
    }

    #[test]
    fn filter_args_thresholds_projects_each_field() {
        // Pin the cli↔library seam: a regression that swapped the
        // field assignments would compile and pass integration tests
        // but produce silently-wrong filter behavior.
        let f = FilterArgs {
            min_tokens: Some(50_000),
            min_cost: Some(0.50),
        };
        let t = f.thresholds();
        assert_eq!(t.min_tokens, Some(50_000));
        assert_eq!(t.min_cost, Some(0.50));

        let empty = FilterArgs::default();
        assert_eq!(empty.thresholds(), Thresholds::default());
    }

    #[test]
    fn filter_args_describe_active_formats_each_combination() {
        let tokens_only = FilterArgs {
            min_tokens: Some(50_000),
            min_cost: None,
        };
        assert_eq!(tokens_only.describe_active(), "--min-tokens 50000");

        let cost_only = FilterArgs {
            min_tokens: None,
            min_cost: Some(0.50),
        };
        assert_eq!(cost_only.describe_active(), "--min-cost 0.5");

        let both = FilterArgs {
            min_tokens: Some(50_000),
            min_cost: Some(0.50),
        };
        assert_eq!(both.describe_active(), "--min-tokens 50000 --min-cost 0.5",);

        // Regression guard: the default `{}` formatter must round-trip
        // small values faithfully — `{:.2}` would truncate this to
        // `--min-cost 0.00`.
        let small = FilterArgs {
            min_tokens: None,
            min_cost: Some(0.0001),
        };
        assert!(
            small.describe_active().contains("--min-cost 0.0001"),
            "expected 0.0001 to round-trip; got: {}",
            small.describe_active(),
        );
    }
}
