//! clap-derived CLI surface for the `cclens` binary.
//!
//! Public to the binary only (declared as `mod cli;` in `main.rs` rather
//! than in `lib.rs`) so that no library module can import clap-derived
//! types — keeps the library/CLI seam visible.
//!
//! Public API (binary-internal):
//! - `Cli` / `Command` / `PricingAction` (variants: `Refresh`, `Info`,
//!   `List`) — clap parser types.
//! - `ThresholdsFilterArgs` — flattened `--min-tokens` / `--min-cost`
//!   threshold flags; `.thresholds_filter()` produces a library
//!   `ThresholdsFilter`.
//! - `SessionFilterArgs` — flattened `--project` / `--since` /
//!   `--until` scope flags shared by `cclens list` and `cclens inputs`;
//!   `.session_filter()` produces a library `SessionFilter`.
//! - `InputsArgs` — flattened `inputs`-only `--session` flag.
//! - `emit_empty_result_hint(&SessionFilterArgs, &ThresholdsFilterArgs)`
//!   — stderr hint used by `run_list` and `run_show` when the filters
//!   dropped every row.
//! - `emit_inputs_empty_hint(&SessionFilterArgs, &InputsArgs,
//!   &ThresholdsFilterArgs)` — sibling hint that describes the inputs-
//!   side, scope, and threshold filters when `cclens inputs` produces
//!   no rows.

use std::path::PathBuf;

use cclens::filter::{SessionFilter, ThresholdsFilter, parse_filter_datetime, parse_min_cost};
use cclens::loading::Query;
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum OutputFormat {
    Plain,
    Json,
}

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

    /// Output format: `plain` for plain-text tables, `json` for
    /// machine-readable JSON. When omitted, the TUI is used if stdout
    /// is a terminal, otherwise `plain`.
    #[arg(long, global = true, value_enum)]
    pub(super) format: Option<OutputFormat>,

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
        scope: SessionFilterArgs,
        #[command(flatten)]
        thresholds: ThresholdsFilterArgs,
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
        thresholds: ThresholdsFilterArgs,
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
        // Field order shapes `--help` ordering (clap inlines flattened
        // groups in field order). `inputs` is listed first to keep
        // `--session` ahead of the scope flags, matching the
        // pre-refactor help order and `emit_inputs_empty_hint`'s
        // composition order.
        #[command(flatten)]
        inputs: InputsArgs,
        #[command(flatten)]
        scope: SessionFilterArgs,
        #[command(flatten)]
        thresholds: ThresholdsFilterArgs,
    },
}

/// Shared `--min-tokens` / `--min-cost` thresholds for `list`, `show`,
/// and `inputs`. Flattened into each subcommand via `#[command(flatten)]`;
/// deliberately not flattened into `pricing` (so `pricing refresh
/// --min-tokens 1` is a clap parse error, not a silent no-op).
///
/// The `Copy` bound matters: a threshold pair is two scalar `Option`s,
/// and copying them around the renderer avoids borrow plumbing without
/// adding a closure.
#[derive(Args, Debug, Clone, Copy, Default)]
pub(super) struct ThresholdsFilterArgs {
    /// Show only rows with at least N billable tokens (e.g. --min-tokens 50000)
    #[arg(long)]
    min_tokens: Option<u64>,
    /// Show only rows costing at least USD, e.g. --min-cost 0.50; unknown-cost rows excluded
    #[arg(long, value_parser = parse_min_cost)]
    min_cost: Option<f64>,
}

impl ThresholdsFilterArgs {
    /// Project the clap-derived flags into the library-side
    /// `ThresholdsFilter`. `ThresholdsFilterArgs` is binary-only
    /// (clap-derived); `ThresholdsFilter` lives in the library crate
    /// and is what `render_session` and the library's session-level
    /// filter take — keeping the library/CLI seam free of clap
    /// dependencies.
    pub(super) fn thresholds_filter(&self) -> ThresholdsFilter {
        ThresholdsFilter {
            min_tokens: self.min_tokens,
            min_cost: self.min_cost,
        }
    }
}

/// Shared `--project` / `--since` / `--until` scope flags for `cclens
/// list` and `cclens inputs`. Single source of truth for session-scope
/// filtering: adding a new scope flag (or adjusting an existing one's
/// help text) is a one-place change.
#[derive(Args, Debug, Clone, Default)]
pub(super) struct SessionFilterArgs {
    /// Restrict to one project (matches the short name shown in the
    /// `list` view's `project` column).
    #[arg(long)]
    project: Option<String>,
    /// Include only sessions whose `started_at` is at or after this
    /// time (inclusive). Accepts `YYYY-MM-DD`, which means midnight
    /// UTC, or a full RFC 3339 timestamp.
    #[arg(long, value_parser = parse_filter_datetime)]
    since: Option<DateTime<Utc>>,
    /// Include only sessions whose `started_at` is at or before this
    /// time (inclusive). Accepts `YYYY-MM-DD`, which means midnight
    /// UTC, or a full RFC 3339 timestamp.
    #[arg(long, value_parser = parse_filter_datetime)]
    until: Option<DateTime<Utc>>,
}

impl SessionFilterArgs {
    /// Project the clap-derived scope flags into the library-side
    /// `SessionFilter`. Same library/CLI seam pattern as
    /// `ThresholdsFilterArgs::thresholds_filter`.
    pub(super) fn session_filter(&self) -> SessionFilter {
        SessionFilter {
            project_name: self.project.clone(),
            since: self.since,
            until: self.until,
        }
    }
}

/// `cclens inputs`-only flags. Currently a single `--session` UUID.
/// Carrying it in its own struct (rather than collapsing it into
/// `SessionFilterArgs`) preserves the "shared scope flags appear on
/// both list and inputs" invariant — `--session` is genuinely
/// inputs-specific because `cclens show <id>` already covers single-
/// session navigation on the listing side.
#[derive(Args, Debug, Clone, Default)]
pub(super) struct InputsArgs {
    /// Restrict attribution to one session by full UUID.
    #[arg(long)]
    session: Option<String>,
}

impl InputsArgs {
    pub(super) fn session_id(&self) -> Option<String> {
        self.session.clone()
    }
}

/// Emit `note: no rows matched <flags>` to stderr when the `list` /
/// `show` filters dropped every row. No-op when no filter is active so
/// the pre-existing "empty `projects_dir` produces no stderr" contract
/// is preserved. Delegates the description to `Query::describe_active`,
/// the one producer both this hint and the TUI header read from.
pub(super) fn emit_empty_result_hint(scope: &SessionFilterArgs, thresholds: &ThresholdsFilterArgs) {
    emit_hint(&Query {
        sessions: scope.session_filter(),
        thresholds: thresholds.thresholds_filter(),
        inputs_session_id: None,
    });
}

/// Sibling of `emit_empty_result_hint` for `cclens inputs`: adds the
/// inputs-only `--session` constraint to the same `Query`, which puts
/// it first in the rendered order structurally rather than by
/// hand-ordered concatenation.
pub(super) fn emit_inputs_empty_hint(
    scope: &SessionFilterArgs,
    inputs: &InputsArgs,
    thresholds: &ThresholdsFilterArgs,
) {
    emit_hint(&Query {
        sessions: scope.session_filter(),
        thresholds: thresholds.thresholds_filter(),
        inputs_session_id: inputs.session_id(),
    });
}

/// Shared tail of both empty-result hints: one producer of the
/// description text, joined with spaces. Suppressed when no filter is
/// active, preserving the "empty `projects_dir` produces no stderr"
/// contract.
fn emit_hint(query: &Query) {
    let components = query.describe_active();
    if components.is_empty() {
        return;
    }
    let combined = components
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("note: no rows matched {combined}");
}

#[derive(Subcommand, Clone, Copy)]
pub(super) enum PricingAction {
    /// Re-fetch the `LiteLLM` pricing catalog and overwrite the cache.
    Refresh,
    /// Print cache path, size, mtime, and Claude-entry count.
    Info,
    /// Show per-model rates for all Claude entries in the catalog.
    List {
        /// Include provider/region-prefixed entries (`bedrock`, `vertex_ai`,
        /// etc.) in addition to bare `claude-*` keys.
        #[arg(long)]
        all: bool,
    },
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
    fn thresholds_filter_args_projection_covers_each_field() {
        // Pin the cli↔library seam: a regression that swapped the
        // field assignments would compile and pass integration tests
        // but produce silently-wrong filter behavior.
        let f = ThresholdsFilterArgs {
            min_tokens: Some(50_000),
            min_cost: Some(0.50),
        };
        let t = f.thresholds_filter();
        assert_eq!(t.min_tokens, Some(50_000));
        assert_eq!(t.min_cost, Some(0.50));

        let empty = ThresholdsFilterArgs::default();
        assert_eq!(empty.thresholds_filter(), ThresholdsFilter::default());
    }

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    #[test]
    fn session_filter_args_projection_covers_each_field() {
        // Pin the cli↔library seam: a swapped-field regression would
        // compile and silently produce wrong filter behavior.
        let all = SessionFilterArgs {
            project: Some("alpha".to_string()),
            since: Some(ts("2026-04-10T00:00:00Z")),
            until: Some(ts("2026-04-20T00:00:00Z")),
        };
        let f = all.session_filter();
        assert_eq!(f.project_name, Some("alpha".to_string()));
        assert_eq!(f.since, Some(ts("2026-04-10T00:00:00Z")));
        assert_eq!(f.until, Some(ts("2026-04-20T00:00:00Z")));

        let project_only = SessionFilterArgs {
            project: Some("beta".to_string()),
            since: None,
            until: None,
        };
        assert_eq!(
            project_only.session_filter().project_name,
            Some("beta".to_string()),
        );
        assert!(project_only.session_filter().since.is_none());
        assert!(project_only.session_filter().until.is_none());

        let since_only = SessionFilterArgs {
            project: None,
            since: Some(ts("2026-04-15T00:00:00Z")),
            until: None,
        };
        assert!(since_only.session_filter().project_name.is_none());
        assert_eq!(
            since_only.session_filter().since,
            Some(ts("2026-04-15T00:00:00Z")),
        );

        let empty = SessionFilterArgs::default();
        assert_eq!(empty.session_filter(), SessionFilter::default());
    }

    #[test]
    fn format_json_parses_before_subcommand() {
        let cli = Cli::try_parse_from(["cclens", "--format", "json", "list"]).unwrap();
        assert!(matches!(cli.format, Some(OutputFormat::Json)));
    }

    #[test]
    fn format_json_parses_after_subcommand() {
        let cli = Cli::try_parse_from(["cclens", "list", "--format", "json"]).unwrap();
        assert!(matches!(cli.format, Some(OutputFormat::Json)));
    }

    #[test]
    fn format_plain_parses() {
        let cli = Cli::try_parse_from(["cclens", "--format", "plain"]).unwrap();
        assert!(matches!(cli.format, Some(OutputFormat::Plain)));
    }

    #[test]
    fn format_defaults_to_none() {
        let cli = Cli::try_parse_from(["cclens"]).unwrap();
        assert!(cli.format.is_none());
    }

    #[test]
    fn old_plain_flag_is_rejected() {
        let result = Cli::try_parse_from(["cclens", "--plain"]);
        assert!(result.is_err(), "--plain should no longer be accepted");
    }

    #[test]
    fn output_format_variants_round_trip_through_value_enum() {
        use clap::ValueEnum;
        let variants = OutputFormat::value_variants();
        assert_eq!(variants.len(), 2);
        for v in variants {
            let s = v.to_possible_value().unwrap();
            let parsed =
                OutputFormat::from_str(s.get_name(), true).expect("round-trip must succeed");
            assert_eq!(std::mem::discriminant(&parsed), std::mem::discriminant(v),);
        }
    }

    #[test]
    fn pricing_list_parses_without_all_flag() {
        let cli = Cli::try_parse_from(["cclens", "pricing", "list"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Pricing {
                action: PricingAction::List { all: false }
            })
        ));
    }

    #[test]
    fn pricing_list_parses_with_all_flag() {
        let cli = Cli::try_parse_from(["cclens", "pricing", "list", "--all"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Pricing {
                action: PricingAction::List { all: true }
            })
        ));
    }
}
