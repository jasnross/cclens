//! clap-derived CLI surface for the `cclens` binary.
//!
//! Public to the binary only (declared as `mod cli;` in `main.rs` rather
//! than in `lib.rs`) so that no library module can import clap-derived
//! types — keeps the library/CLI seam visible.
//!
//! Public API (binary-internal):
//! - `Cli` / `Command` / `PricingAction` — clap parser types.
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

use cclens::filter::{SessionFilter, ThresholdsFilter};
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
    #[arg(long)]
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

    /// True iff at least one threshold flag is active. Used to gate
    /// the empty-result stderr hint — when no filter is active, an
    /// empty result is just an empty `projects_dir` and gets no hint.
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
    /// ISO-8601 timestamp (inclusive).
    #[arg(long)]
    since: Option<DateTime<Utc>>,
    /// Include only sessions whose `started_at` is at or before this
    /// ISO-8601 timestamp (inclusive).
    #[arg(long)]
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

    fn any_active(&self) -> bool {
        self.project.is_some() || self.since.is_some() || self.until.is_some()
    }

    fn describe_active(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
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

    fn any_active(&self) -> bool {
        self.session.is_some()
    }

    fn describe_active(&self) -> String {
        match &self.session {
            Some(s) => format!("--session {s}"),
            None => String::new(),
        }
    }
}

/// Emit `note: no rows matched <flags>` to stderr when the `list` /
/// `show` filters dropped every row. No-op when no filter is active so
/// the pre-existing "empty `projects_dir` produces no stderr" contract
/// is preserved. Composes scope-flag and threshold-flag descriptions
/// in display order: scope first, thresholds second.
pub(super) fn emit_empty_result_hint(scope: &SessionFilterArgs, thresholds: &ThresholdsFilterArgs) {
    if !scope.any_active() && !thresholds.any_active() {
        return;
    }
    let mut combined = scope.describe_active();
    if thresholds.any_active() {
        if !combined.is_empty() {
            combined.push(' ');
        }
        combined.push_str(&thresholds.describe_active());
    }
    eprintln!("note: no rows matched {combined}");
}

/// Sibling of `emit_empty_result_hint` for `cclens inputs`: surfaces
/// the inputs-side, scope, and threshold filter sets in one stderr
/// line. Suppresses the hint when no filter is active. Display order
/// is `inputs` (--session), then `scope` (--project/--since/--until),
/// then `thresholds` (--min-tokens/--min-cost) — matching how the
/// pre-split `InputsFilterArgs::describe_active` ordered them.
pub(super) fn emit_inputs_empty_hint(
    scope: &SessionFilterArgs,
    inputs: &InputsArgs,
    thresholds: &ThresholdsFilterArgs,
) {
    if !scope.any_active() && !inputs.any_active() && !thresholds.any_active() {
        return;
    }
    let mut combined = inputs.describe_active();
    let scope_str = scope.describe_active();
    if !scope_str.is_empty() {
        if !combined.is_empty() {
            combined.push(' ');
        }
        combined.push_str(&scope_str);
    }
    if thresholds.any_active() {
        if !combined.is_empty() {
            combined.push(' ');
        }
        combined.push_str(&thresholds.describe_active());
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

    #[test]
    fn thresholds_filter_args_describe_active_formats_each_combination() {
        let tokens_only = ThresholdsFilterArgs {
            min_tokens: Some(50_000),
            min_cost: None,
        };
        assert_eq!(tokens_only.describe_active(), "--min-tokens 50000");

        let cost_only = ThresholdsFilterArgs {
            min_tokens: None,
            min_cost: Some(0.50),
        };
        assert_eq!(cost_only.describe_active(), "--min-cost 0.5");

        let both = ThresholdsFilterArgs {
            min_tokens: Some(50_000),
            min_cost: Some(0.50),
        };
        assert_eq!(both.describe_active(), "--min-tokens 50000 --min-cost 0.5",);

        // Regression guard: the default `{}` formatter must round-trip
        // small values faithfully — `{:.2}` would truncate this to
        // `--min-cost 0.00`.
        let small = ThresholdsFilterArgs {
            min_tokens: None,
            min_cost: Some(0.0001),
        };
        assert!(
            small.describe_active().contains("--min-cost 0.0001"),
            "expected 0.0001 to round-trip; got: {}",
            small.describe_active(),
        );
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
    fn session_filter_args_describe_active_formats_each_combination() {
        let empty = SessionFilterArgs::default();
        assert_eq!(empty.describe_active(), "");

        let project_only = SessionFilterArgs {
            project: Some("alpha".to_string()),
            since: None,
            until: None,
        };
        assert_eq!(project_only.describe_active(), "--project alpha");

        let since_only = SessionFilterArgs {
            project: None,
            since: Some(ts("2026-04-15T00:00:00Z")),
            until: None,
        };
        // `to_rfc3339()` round-trips back to the input string.
        assert_eq!(
            since_only.describe_active(),
            "--since 2026-04-15T00:00:00+00:00"
        );

        let until_only = SessionFilterArgs {
            project: None,
            since: None,
            until: Some(ts("2026-04-20T00:00:00Z")),
        };
        assert_eq!(
            until_only.describe_active(),
            "--until 2026-04-20T00:00:00+00:00"
        );

        let all = SessionFilterArgs {
            project: Some("alpha".to_string()),
            since: Some(ts("2026-04-10T00:00:00Z")),
            until: Some(ts("2026-04-20T00:00:00Z")),
        };
        assert_eq!(
            all.describe_active(),
            "--project alpha --since 2026-04-10T00:00:00+00:00 --until 2026-04-20T00:00:00+00:00",
        );
    }

    #[test]
    fn inputs_args_describe_active_formats_session() {
        let empty = InputsArgs::default();
        assert_eq!(empty.describe_active(), "");
        assert!(!empty.any_active());

        let with_session = InputsArgs {
            session: Some("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".to_string()),
        };
        assert_eq!(
            with_session.describe_active(),
            "--session aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        );
        assert!(with_session.any_active());
    }
}
