use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cclens::aggregation::{
    Exchange, aggregate, exchange_filter_totals, group_into_exchanges, user_display_string,
};
use cclens::discovery::discover;
use cclens::domain::{CacheCreation, Role, Session, Turn, Usage};
use cclens::filter::Thresholds;
use cclens::parsing::parse_jsonl;
use cclens::pricing;
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand};
use comfy_table::presets::NOTHING;
use comfy_table::{CellAlignment, Table};
use serde_json::Value;

// ---- cli ----

#[derive(Parser)]
#[command(
    name = "cclens",
    about = "Browse Claude Code conversations (tokens + cost)"
)]
struct Cli {
    /// Directory to scan for project conversations.
    #[arg(long, default_value_os_t = default_projects_dir())]
    projects_dir: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
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
struct FilterArgs {
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
    /// the library crate and is what `render_session` (and any future
    /// library consumer of the threshold predicate) takes — keeping the
    /// library/CLI seam free of clap dependencies.
    fn thresholds(&self) -> Thresholds {
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
fn emit_empty_result_hint(filters: &FilterArgs) {
    if !filters.any_active() {
        return;
    }
    eprintln!("note: no rows matched {}", filters.describe_active());
}

#[derive(Subcommand, Clone, Copy)]
enum PricingAction {
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

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::List {
        filters: FilterArgs::default(),
    }) {
        Command::List { filters } => run_list(&cli.projects_dir, filters),
        Command::Show {
            session_id,
            filters,
        } => run_show(&cli.projects_dir, &session_id, filters),
        Command::Pricing { action } => run_pricing(action),
    }
}

fn run_list(projects_dir: &Path, filters: FilterArgs) -> anyhow::Result<()> {
    let catalog = pricing::load_catalog();
    let project_entries = discover(projects_dir)?;
    let mut sessions = Vec::new();
    for (project_dir, jsonl_paths) in project_entries {
        // Cross-file dedup state, scoped to one project. Resumed
        // sessions replay prior assistant turns verbatim; this set
        // ensures any `(message.id, requestId)` pair contributes cost
        // exactly once across the project's `.jsonl` files. `discover`
        // already returned `jsonl_paths` in mtime-ascending order, so
        // the earliest file containing a given key wins.
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for jsonl_path in jsonl_paths {
            // A single unreadable file should not abort the whole listing.
            let Ok(turns) = parse_jsonl(&jsonl_path) else {
                continue;
            };
            let turns = dedup_assistant_turns(turns, &mut seen);
            let session_id = jsonl_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            // The new threshold filter sits *after* `aggregate`'s
            // existing zero-billable pre-pass, so it composes
            // additively rather than altering the "session is
            // meaningful" contract.
            if let Some(session) = aggregate(&project_dir, session_id, turns, &catalog)
                && filters
                    .thresholds()
                    .matches(session.total_billable, session.total_cost)
            {
                sessions.push(session);
            }
        }
    }
    sessions.sort_by_key(|s| s.started_at);
    println!("{}", render_table(&sessions));
    if sessions.is_empty() {
        emit_empty_result_hint(&filters);
    }
    Ok(())
}

/// Drop assistant turns whose `(message_id, request_id)` pair has
/// already been seen earlier in this project's file walk.
///
/// Mirrors ccusage's `createUniqueHash`: a turn missing either part of
/// the pair passes through (matches their `null`-on-partial-key rule).
/// Non-assistant turns (user, attachment, system, other) are unaffected
/// — they don't carry billable usage and are required for title
/// extraction and exchange grouping. Sidechain markers (`isSidechain`)
/// aren't visible at this layer (the parser doesn't surface the flag),
/// so every assistant turn is keyed on `(message_id, request_id)`
/// regardless of sidechain status.
fn dedup_assistant_turns(turns: Vec<Turn>, seen: &mut HashSet<(String, String)>) -> Vec<Turn> {
    turns
        .into_iter()
        .filter(|turn| match &turn.role {
            Role::Assistant => match (turn.message_id.as_ref(), turn.request_id.as_ref()) {
                (Some(mid), Some(rid)) => seen.insert((mid.clone(), rid.clone())),
                // Missing key — pass through unchanged (matches
                // ccusage's createUniqueHash returning null on
                // partial keys).
                _ => true,
            },
            Role::User | Role::Attachment | Role::System | Role::Other(_) => true,
        })
        .collect()
}

fn run_pricing(action: PricingAction) -> anyhow::Result<()> {
    match action {
        PricingAction::Refresh => {
            let report = pricing::refresh_catalog()?;
            println!("Refreshed catalog at {}", report.path.display());
            println!(
                "  previous size: {} bytes → new size: {} bytes",
                report.previous_size, report.new_size,
            );
            println!("  Claude entries: {}", report.entry_count);
            Ok(())
        }
        PricingAction::Info => {
            let info = pricing::cache_info();
            match info.path {
                Some(path) => println!("Cache path: {}", path.display()),
                None => println!("Cache path: (no cache directory available)"),
            }
            println!("  exists: {}", info.exists);
            let mtime = info.last_modified.map_or_else(
                || "(never)".to_string(),
                |ts| {
                    let dt: chrono::DateTime<chrono::Local> = ts.into();
                    dt.format("%Y-%m-%d %H:%M:%S").to_string()
                },
            );
            println!("  last modified: {mtime}");
            println!("  size: {} bytes", info.size);
            let entries = info
                .entry_count
                .map_or_else(|| "(unreadable)".to_string(), |n| n.to_string());
            println!("  Claude entries: {entries}");
            Ok(())
        }
    }
}

fn run_show(projects_dir: &Path, session_id: &str, filters: FilterArgs) -> anyhow::Result<()> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        anyhow::bail!("session id must not be empty");
    }
    let project_entries = discover(projects_dir)?;

    // Locate the project that owns this session. A stem collision
    // *across* projects is unlikely (Claude Code uses fresh UUIDs)
    // but we keep the "multiple sessions match id …" error for
    // global ambiguity. Filenames within a single project directory
    // are unique, so per-project ambiguity isn't possible here.
    let mut matched_project: Option<(PathBuf, Vec<PathBuf>)> = None;
    for (project_dir, jsonl_paths) in project_entries {
        if jsonl_paths.iter().any(|p| stem_matches(p, session_id)) {
            if matched_project.is_some() {
                anyhow::bail!("multiple sessions match id {session_id}");
            }
            matched_project = Some((project_dir, jsonl_paths));
        }
    }
    let Some((_project_dir, jsonl_paths)) = matched_project else {
        anyhow::bail!("no session matches id {session_id}");
    };

    // Walk the project's files in mtime-ascending order, threading the
    // same per-project dedup state used by `run_list`. Stop as soon as
    // we've processed the target file: files later in mtime order
    // can't affect the target's filtered turns.
    let catalog = pricing::load_catalog();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut target_turns: Option<Vec<Turn>> = None;
    for jsonl_path in jsonl_paths {
        let Ok(turns) = parse_jsonl(&jsonl_path) else {
            continue;
        };
        let turns = dedup_assistant_turns(turns, &mut seen);
        if stem_matches(&jsonl_path, session_id) {
            target_turns = Some(turns);
            break;
        }
    }
    let turns =
        target_turns.ok_or_else(|| anyhow::anyhow!("no session matches id {session_id}"))?;
    let exchanges = group_into_exchanges(&turns);
    let (rendered, rows_shown) = render_session(&exchanges, &catalog, filters.thresholds());
    println!("{rendered}");
    if rows_shown == 0 {
        emit_empty_result_hint(&filters);
    }
    Ok(())
}

fn stem_matches(path: &Path, session_id: &str) -> bool {
    path.file_stem()
        .is_some_and(|s| s.to_string_lossy() == session_id)
}

// ---- rendering ----

const TITLE_MAX_CHARS: usize = 80;

// Indices of numeric columns in `render_table`'s header:
//   vec!["datetime", "project", "title", "tokens", "cost", "id"]
//                                         idx 3    idx 4
// Right-alignment is applied at these positions; reordering the header
// requires updating these constants in lockstep.
const TOKENS_COL_INDEX: usize = 3;
const COST_COL_INDEX: usize = 4;

// Indices of numeric columns in `render_session`'s header:
//   vec!["datetime", "role", "tokens", "cost", "cumulative", "cum_cost", "content"]
//                             idx 2    idx 3   idx 4         idx 5
const SHOW_TOKENS_COL_INDEX: usize = 2;
const SHOW_COST_COL_INDEX: usize = 3;
const SHOW_CUMULATIVE_COL_INDEX: usize = 4;
const SHOW_CUM_COST_COL_INDEX: usize = 5;

/// Format an optional cost as `$X.XXXX` or `—` for the unknown-model
/// case. Centralized so list and show share the exact same vocabulary.
fn format_cost_opt(c: Option<f64>) -> String {
    c.map_or_else(|| "—".to_string(), |n| format!("${n:.4}"))
}

fn truncate_title(s: &str, max: usize) -> String {
    // Collapse internal whitespace runs (including `\n`, `\t`) to a single
    // space and trim ends. Comfy-table respects embedded newlines and would
    // otherwise render a single cell across multiple visual rows — real
    // JSONL content (e.g. skill preambles) contains newlines that would
    // break the one-line-per-row invariant without this step.
    let normalized = s.split_whitespace().collect::<Vec<_>>().join(" ");

    if normalized.chars().count() <= max {
        return normalized;
    }
    let mut result: String = normalized.chars().take(max.saturating_sub(1)).collect();
    result.push('…');
    result
}

fn format_local(ts: DateTime<Utc>) -> String {
    ts.with_timezone(&chrono::Local)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

// Returns the empty string on `None` so `render_session` stays infallible
// without `.unwrap()` (banned by the `unwrap_used` lint). In practice the
// timestamp is always present for substantive user turns and assistant turns;
// this branch exists only to keep the code panic-free.
fn format_local_or_empty(ts: Option<DateTime<Utc>>) -> String {
    ts.map_or_else(String::new, format_local)
}

fn render_table(sessions: &[Session]) -> String {
    let mut table = Table::new();
    table.load_preset(NOTHING);
    table.set_header(vec!["datetime", "project", "title", "tokens", "cost", "id"]);
    for session in sessions {
        table.add_row(vec![
            format_local(session.started_at),
            session.project_short_name.clone(),
            truncate_title(&session.title, TITLE_MAX_CHARS),
            session.total_billable.to_string(),
            format_cost_opt(session.total_cost),
            session.id.clone(),
        ]);
    }
    // column_mut returns Option; the columns are guaranteed present because
    // the header above defines them at the indices.
    if let Some(col) = table.column_mut(TOKENS_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(COST_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    format!("{table}")
}

fn user_content_preview(turn: &Turn) -> String {
    // The grouper guarantees the turn is substantive, so `None` is unreachable
    // in practice; `unwrap_or_default` keeps the renderer infallible without
    // requiring `.unwrap()`.
    turn.content
        .as_ref()
        .and_then(user_display_string)
        .unwrap_or_default()
}

fn assistant_cluster_preview(assistants: &[&Turn]) -> String {
    // First non-empty text block across the cluster.
    for a in assistants {
        let Some(Value::Array(blocks)) = a.content.as_ref() else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            if let Some(text) = block.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                return text.to_string();
            }
        }
    }

    // Fallback: deduped tool-use names, in order of first appearance.
    let mut names: Vec<String> = Vec::new();
    for a in assistants {
        let Some(Value::Array(blocks)) = a.content.as_ref() else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            if let Some(name) = block.get("name").and_then(Value::as_str)
                && !names.iter().any(|n| n == name)
            {
                names.push(name.to_string());
            }
        }
    }
    names.join(", ")
}

/// Sum costs across an assistant cluster with strict `None`
/// propagation. The closure picks which token components count for
/// this row (e.g. `(input, 0, cache_creation, cache_read)` for the
/// user row, `(0, output, CacheCreation::default(), 0)` for the
/// assistant row). Any single turn the catalog can't price collapses
/// the whole sum to `None`, matching the session-level rule.
fn strict_fold_assistant_cost(
    assistants: &[&Turn],
    catalog: &pricing::PricingCatalog,
    pick: impl Fn(&Usage) -> (u64, u64, CacheCreation, u64),
) -> Option<f64> {
    let mut sum = 0.0;
    for turn in assistants {
        let Some(usage) = turn.usage.as_ref() else {
            // Assistant turn with no usage: zero contribution, does
            // not collapse the sum to `None`.
            continue;
        };
        let (input, output, cache_creation, cache_read) = pick(usage);
        let cost = catalog.cost_for_components(
            input,
            output,
            cache_creation,
            cache_read,
            turn.model.as_deref(),
        )?;
        sum += cost;
    }
    Some(sum)
}

fn count_tool_uses(assistants: &[&Turn]) -> u64 {
    let mut n: u64 = 0;
    for a in assistants {
        let Some(Value::Array(blocks)) = a.content.as_ref() else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                n += 1;
            }
        }
    }
    n
}

/// Strict-fold the running cost: both `Some` → sum; either `None` →
/// stays `None` for this row and every subsequent row. Callers retain
/// the previous accumulator value so a `None` "latches" through every
/// later row, matching the unknown-model propagation contract.
fn fold_cum_cost(prev: Option<f64>, delta: Option<f64>) -> Option<f64> {
    prev.zip(delta).map(|(a, b)| a + b)
}

/// Render the per-exchange table.
///
/// Returns `(rendered, rows_shown)`. `rows_shown` counts physical
/// rendered rows (1 for an orphan-only visible exchange, 2 for a
/// normal visible exchange). `run_show` uses `rows_shown == 0` to
/// decide whether to emit the empty-result stderr hint.
///
/// Filtering happens here rather than in a pre-pass so that
/// `cumulative` and `cum_cost` continue to fold over **every**
/// exchange — the running totals on visible rows must still match the
/// session-level `list` totals, which means the renderer needs both
/// the unfiltered slice (for accumulation) and the predicate (for
/// skipping `add_row`).
fn render_session(
    exchanges: &[Exchange<'_>],
    catalog: &pricing::PricingCatalog,
    thresholds: Thresholds,
) -> (String, usize) {
    let mut table = Table::new();
    table.load_preset(NOTHING);
    table.set_header(vec![
        "datetime",
        "role",
        "tokens",
        "cost",
        "cumulative",
        "cum_cost",
        "content",
    ]);
    let mut cumulative: u64 = 0;
    let mut cum_cost: Option<f64> = Some(0.0);
    let mut rows_shown: usize = 0;

    for exchange in exchanges {
        let (ex_tokens, ex_cost) = exchange_filter_totals(exchange, catalog);
        let visible = thresholds.matches(ex_tokens, ex_cost);

        let user_tokens_opt: Option<u64> = if exchange.assistants.is_empty() {
            None
        } else {
            let sum = exchange
                .assistants
                .iter()
                .filter_map(|t| t.usage.as_ref())
                .map(|u| u.input + u.cache_creation.total())
                .sum();
            Some(sum)
        };
        let output_tokens: u64 = exchange
            .assistants
            .iter()
            .filter_map(|t| t.usage.as_ref())
            .map(|u| u.output)
            .sum();

        // Per-row cost decomposes into a *displayed* cost (what the
        // cell shows) and an *accumulator delta* (what's added to the
        // running cum_cost). They diverge on the orphan-user case:
        // an empty assistant cluster displays `—` but contributes
        // `Some(0.0)` to the running total — same way `cumulative +=
        // user_tokens_opt.unwrap_or(0)` treats orphans as a no-op.
        // For non-empty clusters they're equal: any unknown-model turn
        // collapses both to `None`, latching cum_cost through the rest
        // of the session.
        let (user_cost_display, user_cost_delta) = if exchange.assistants.is_empty() {
            (None, Some(0.0))
        } else {
            let cost = strict_fold_assistant_cost(&exchange.assistants, catalog, |usage| {
                (usage.input, 0, usage.cache_creation, usage.cache_read)
            });
            (cost, cost)
        };

        cumulative += user_tokens_opt.unwrap_or(0);
        cum_cost = fold_cum_cost(cum_cost, user_cost_delta);
        if visible {
            table.add_row(vec![
                format_local_or_empty(exchange.user.timestamp),
                "user".to_string(),
                user_tokens_opt.map_or_else(|| "—".to_string(), |n| n.to_string()),
                format_cost_opt(user_cost_display),
                cumulative.to_string(),
                format_cost_opt(cum_cost),
                truncate_title(&user_content_preview(exchange.user), TITLE_MAX_CHARS),
            ]);
            rows_shown += 1;
        }

        if let Some(first_assistant) = exchange.assistants.first() {
            let assistant_cost =
                strict_fold_assistant_cost(&exchange.assistants, catalog, |usage| {
                    (0, usage.output, CacheCreation::default(), 0)
                });
            cumulative += output_tokens;
            cum_cost = fold_cum_cost(cum_cost, assistant_cost);

            if visible {
                let preview = assistant_cluster_preview(&exchange.assistants);
                let n_tools = count_tool_uses(&exchange.assistants);
                let content = if n_tools > 0 {
                    format!("{preview} +{n_tools} tool uses")
                } else {
                    preview
                };
                table.add_row(vec![
                    format_local_or_empty(first_assistant.timestamp),
                    "assistant".to_string(),
                    output_tokens.to_string(),
                    format_cost_opt(assistant_cost),
                    cumulative.to_string(),
                    format_cost_opt(cum_cost),
                    truncate_title(&content, TITLE_MAX_CHARS),
                ]);
                rows_shown += 1;
            }
        }
    }

    if let Some(col) = table.column_mut(SHOW_TOKENS_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(SHOW_COST_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(SHOW_CUMULATIVE_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(SHOW_CUM_COST_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    (format!("{table}"), rows_shown)
}

// ---- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    // --- test helpers ---

    fn user_string_turn(content: &str) -> Turn {
        Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::String(content.to_string())),
            cwd: None,
        }
    }

    fn user_array_turn(content: Value) -> Turn {
        Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(content),
            cwd: None,
        }
    }

    fn assistant_turn_with_usage(input: u64, output: u64, cache_creation: u64) -> Turn {
        // Helper takes a flat u64 for backwards-compatible test
        // ergonomics; treated as 5m, matching the legacy wire scalar.
        Turn {
            timestamp: Some("2026-04-01T10:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input,
                output,
                cache_creation: CacheCreation {
                    ephemeral_5m: cache_creation,
                    ephemeral_1h: 0,
                },
                cache_read: 0,
            }),
            content: None,
            cwd: None,
        }
    }

    // --- CLI parsing (Phase 1) ---

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

    // --- dedup ---

    #[test]
    fn dedup_drops_duplicate_assistant_turn_within_run() {
        // Build two assistants with the same (mid, rid); a third with
        // a different pair; a fourth with a missing key (must pass);
        // a user turn (must pass regardless).
        let mut first_dup = assistant_turn_with_usage(1, 1, 0);
        first_dup.message_id = Some("m1".into());
        first_dup.request_id = Some("r1".into());
        let mut second_dup = assistant_turn_with_usage(2, 2, 0);
        second_dup.message_id = Some("m1".into());
        second_dup.request_id = Some("r1".into());
        let mut other = assistant_turn_with_usage(3, 3, 0);
        other.message_id = Some("m2".into());
        other.request_id = Some("r2".into());
        let mut partial = assistant_turn_with_usage(4, 4, 0);
        // Partial key — must pass through.
        partial.message_id = Some("m3".into());
        partial.request_id = None;
        let user = user_string_turn("hi");

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let kept =
            dedup_assistant_turns(vec![first_dup, second_dup, other, partial, user], &mut seen);

        // first_dup (kept), second_dup (dropped), other (kept),
        // partial (kept, missing key), user (kept)
        assert_eq!(kept.len(), 4);
        // Verify the key cache observed the two complete pairs.
        assert!(seen.contains(&("m1".to_string(), "r1".to_string())));
        assert!(seen.contains(&("m2".to_string(), "r2".to_string())));
        // Surviving assistant turns retain their hash keys.
        assert_eq!(kept[0].message_id.as_deref(), Some("m1"));
        assert_eq!(kept[0].request_id.as_deref(), Some("r1"));
    }

    #[test]
    fn dedup_drops_duplicate_assistant_across_separate_calls() {
        // The orchestration layer reuses one HashSet across multiple
        // parse_jsonl results — confirm the second call picks up the
        // first call's state.
        let mut first_call_turn = assistant_turn_with_usage(1, 1, 0);
        first_call_turn.message_id = Some("m1".into());
        first_call_turn.request_id = Some("r1".into());
        let mut second_call_turn = assistant_turn_with_usage(2, 2, 0);
        second_call_turn.message_id = Some("m1".into());
        second_call_turn.request_id = Some("r1".into());

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let first = dedup_assistant_turns(vec![first_call_turn], &mut seen);
        let second = dedup_assistant_turns(vec![second_call_turn], &mut seen);
        assert_eq!(first.len(), 1);
        assert!(second.is_empty(), "duplicate from second call must drop");
    }

    #[test]
    fn dedup_passes_through_sidechain_assistant_with_unique_key() {
        // Sidechain turns aren't special-cased — they're deduped on
        // their own keys like any other assistant turn. This guards
        // against a regression that filtered them by role flag.
        let mut sidechain = assistant_turn_with_usage(50, 0, 0);
        sidechain.message_id = Some("msc".into());
        sidechain.request_id = Some("rsc".into());

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let kept = dedup_assistant_turns(vec![sidechain], &mut seen);
        assert_eq!(kept.len(), 1);
    }

    // --- rendering ---

    #[test]
    fn truncate_title_under_limit_returns_unchanged() {
        assert_eq!(truncate_title("hello", 80), "hello");
        assert_eq!(truncate_title("", 80), "");
        // Boundary: len == max is also considered "under limit" per the `<=`
        // check, so an exact-80-char title passes through unchanged.
        let exactly_80 = "a".repeat(80);
        assert_eq!(truncate_title(&exactly_80, 80), exactly_80);
    }

    #[test]
    fn truncate_title_over_limit_appends_ellipsis() {
        let long: String = "a".repeat(81);
        let truncated = truncate_title(&long, 80);
        assert_eq!(truncated.chars().count(), 80);
        assert!(truncated.ends_with('…'));
        // First 79 chars should be 'a's.
        assert_eq!(
            truncated.chars().take(79).collect::<String>(),
            "a".repeat(79)
        );
    }

    #[test]
    fn truncate_title_handles_multibyte_chars() {
        // "日本語" is 3 scalars but 9 UTF-8 bytes; truncating by scalar is
        // correct and must not panic on a byte boundary.
        let s = "日本語の説明".to_string() + &"あ".repeat(80);
        let truncated = truncate_title(&s, 10);
        assert_eq!(truncated.chars().count(), 10);
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn truncate_title_collapses_embedded_newlines() {
        // Real JSONL content (e.g. skill preambles) has embedded newlines;
        // unnormalized, comfy-table would render one cell across multiple
        // visual rows.
        assert_eq!(truncate_title("hello\nworld", 80), "hello world");
        assert_eq!(truncate_title("line1\n\nline2", 80), "line1 line2");
    }

    #[test]
    fn truncate_title_collapses_whitespace_runs() {
        assert_eq!(truncate_title("hello  \t  world", 80), "hello world");
        assert_eq!(
            truncate_title("  leading and trailing  ", 80),
            "leading and trailing"
        );
    }

    #[test]
    fn format_local_matches_explicit_chrono_composition() {
        let ts: DateTime<Utc> = "2026-04-01T10:30:00Z".parse().unwrap();
        let expected = ts
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M")
            .to_string();
        assert_eq!(format_local(ts), expected);
    }

    fn session_for_render(
        project: &str,
        title: &str,
        total_billable: u64,
        started_at: &str,
    ) -> Session {
        session_for_render_with_cost(project, title, total_billable, None, started_at)
    }

    fn session_for_render_with_cost(
        project: &str,
        title: &str,
        total_billable: u64,
        total_cost: Option<f64>,
        started_at: &str,
    ) -> Session {
        let ts: DateTime<Utc> = started_at.parse().unwrap();
        Session {
            id: "sid".to_string(),
            project_short_name: project.to_string(),
            started_at: ts,
            last_activity: ts,
            title: title.to_string(),
            turns: Vec::new(),
            total_billable,
            total_cost,
        }
    }

    #[test]
    fn render_table_includes_header_and_rows() {
        let sessions = vec![
            session_for_render("alpha", "first title", 100, "2026-04-01T10:00:00Z"),
            session_for_render("beta", "second title", 250, "2026-04-02T10:00:00Z"),
        ];
        let out = render_table(&sessions);
        assert!(out.contains("datetime"));
        assert!(out.contains("project"));
        assert!(out.contains("title"));
        assert!(out.contains("tokens"));
        assert!(out.contains("id"));
        assert!(out.contains("alpha"));
        assert!(out.contains("beta"));
        assert!(out.contains("first title"));
        assert!(out.contains("second title"));
        assert!(out.contains("100"));
        assert!(out.contains("250"));
        assert!(out.contains("sid"));
    }

    #[test]
    fn render_table_truncates_long_titles_with_ellipsis() {
        // Title of 100 chars — longer than TITLE_MAX_CHARS (80) — should be
        // truncated with a trailing `…` in the rendered output.
        let long_title = "x".repeat(100);
        let sessions = vec![session_for_render(
            "p",
            &long_title,
            1,
            "2026-04-01T10:00:00Z",
        )];
        let out = render_table(&sessions);
        assert!(
            out.contains('…'),
            "expected ellipsis in output, got:\n{out}"
        );
        // Full 100-x run must NOT appear verbatim.
        assert!(!out.contains(&"x".repeat(100)));
    }

    #[test]
    fn render_table_right_aligns_tokens_and_cost_columns() {
        // After Phase 3 the column order is:
        //   datetime | project | title | tokens | cost | id
        // Both rows below have unknown-model totals (None catalog), so
        // every cost cell renders `—`. Stripping `sid` then `—` exposes
        // the tokens column as the new right edge — same alignment
        // verification as before, just two strip steps instead of one.
        let sessions = vec![
            session_for_render("p1", "t1", 9, "2026-04-01T10:00:00Z"),
            session_for_render("p2", "t2", 123_456, "2026-04-02T10:00:00Z"),
        ];
        let out = render_table(&sessions);
        let data_lines: Vec<&str> = out
            .lines()
            .filter(|l| l.contains("p1") || l.contains("p2"))
            .collect();
        assert_eq!(data_lines.len(), 2);
        let strip_trailing = |l: &str| {
            let after_id = l
                .trim_end()
                .strip_suffix("sid")
                .expect("row should end with the hardcoded id column value")
                .trim_end();
            after_id
                .strip_suffix('—')
                .expect("cost cell should be em-dash for unknown-model session")
                .trim_end()
                .to_string()
        };
        let end_of_9 = strip_trailing(data_lines.iter().find(|l| l.contains("p1")).unwrap());
        let end_of_123456 = strip_trailing(data_lines.iter().find(|l| l.contains("p2")).unwrap());
        assert!(end_of_9.ends_with('9'), "got: {end_of_9}");
        assert!(end_of_123456.ends_with("123456"), "got: {end_of_123456}");
        // Right-alignment: shorter value has leading whitespace padding,
        // so the prefix-up-to-tokens has the same scalar count for both.
        assert_eq!(end_of_9.chars().count(), end_of_123456.chars().count());
    }

    // --- show: additional helpers ---

    fn assistant_turn_with_content(
        content: Value,
        input: u64,
        output: u64,
        cache_creation: u64,
    ) -> Turn {
        Turn {
            timestamp: Some("2026-04-22T09:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input,
                output,
                cache_creation: CacheCreation {
                    ephemeral_5m: cache_creation,
                    ephemeral_1h: 0,
                },
                cache_read: 0,
            }),
            content: Some(content),
            cwd: None,
        }
    }

    // --- user_content_preview ---

    #[test]
    fn user_content_preview_delegates_to_user_display_string() {
        let content = serde_json::json!([
            { "type": "tool_result", "content": "..." },
            { "type": "text", "text": "hello" },
        ]);
        let turn = user_array_turn(content);
        assert_eq!(user_content_preview(&turn), "hello");
    }

    #[test]
    fn user_content_preview_returns_empty_string_on_unreachable_none() {
        let turn = Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::Null),
            cwd: None,
        };
        assert_eq!(user_content_preview(&turn), "");
    }

    // --- assistant_cluster_preview ---

    #[test]
    fn assistant_cluster_preview_first_text_block() {
        // First assistant has only a tool_use; second opens with a text block.
        let tu = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "x", "input": {} },
        ]);
        let tx = serde_json::json!([
            { "type": "text", "text": "found it" },
        ]);
        let a1 = assistant_turn_with_content(tu, 0, 0, 0);
        let a2 = assistant_turn_with_content(tx, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a1, &a2];
        assert_eq!(assistant_cluster_preview(&cluster), "found it");
    }

    #[test]
    fn assistant_cluster_preview_falls_back_to_deduped_tool_names() {
        let tu_a = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "1", "input": {} },
            { "type": "tool_use", "name": "Bash", "id": "2", "input": {} },
        ]);
        let tu_b = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "3", "input": {} },
            { "type": "tool_use", "name": "Edit", "id": "4", "input": {} },
        ]);
        let a1 = assistant_turn_with_content(tu_a, 0, 0, 0);
        let a2 = assistant_turn_with_content(tu_b, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a1, &a2];
        assert_eq!(assistant_cluster_preview(&cluster), "Read, Bash, Edit");
    }

    #[test]
    fn assistant_cluster_preview_empty_cluster_returns_empty_string() {
        let cluster: Vec<&Turn> = Vec::new();
        assert_eq!(assistant_cluster_preview(&cluster), "");
    }

    // --- count_tool_uses ---

    #[test]
    fn count_tool_uses_counts_across_cluster() {
        let a_content = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "1", "input": {} },
        ]);
        let b_content = serde_json::json!([
            { "type": "text", "text": "..." },
            { "type": "tool_use", "name": "Bash", "id": "2", "input": {} },
            { "type": "tool_use", "name": "Edit", "id": "3", "input": {} },
        ]);
        let a1 = assistant_turn_with_content(a_content, 0, 0, 0);
        let a2 = assistant_turn_with_content(b_content, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a1, &a2];
        assert_eq!(count_tool_uses(&cluster), 3);
    }

    #[test]
    fn count_tool_uses_zero_for_text_only_cluster() {
        let tx = serde_json::json!([{ "type": "text", "text": "hi" }]);
        let a = assistant_turn_with_content(tx, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a];
        assert_eq!(count_tool_uses(&cluster), 0);
    }

    // --- render_session ---

    fn show_user_turn(content: &str, ts: &str) -> Turn {
        Turn {
            timestamp: Some(ts.parse().unwrap()),
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::String(content.to_string())),
            cwd: None,
        }
    }

    fn show_assistant_turn(
        content: Value,
        input: u64,
        output: u64,
        cache_creation: u64,
        ts: &str,
    ) -> Turn {
        Turn {
            timestamp: Some(ts.parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input,
                output,
                cache_creation: CacheCreation {
                    ephemeral_5m: cache_creation,
                    ephemeral_1h: 0,
                },
                cache_read: 0,
            }),
            content: Some(content),
            cwd: None,
        }
    }

    #[test]
    fn render_session_header_includes_all_seven_columns() {
        let (out, rows_shown) = render_session(
            &[],
            &pricing::PricingCatalog::empty(),
            Thresholds::default(),
        );
        assert_eq!(rows_shown, 0);
        assert!(out.contains("datetime"));
        assert!(out.contains("role"));
        assert!(out.contains("tokens"));
        assert!(out.contains("cost"));
        assert!(out.contains("cumulative"));
        assert!(out.contains("cum_cost"));
        assert!(out.contains("content"));
    }

    #[test]
    fn render_session_orphan_user_shows_em_dash_and_preserves_cumulative() {
        let u1 = show_user_turn("first", "2026-04-01T10:00:00Z");
        let a1 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "reply" }]),
            12000,
            345,
            0,
            "2026-04-01T10:01:00Z",
        );
        let u2 = show_user_turn("orphan", "2026-04-01T10:02:00Z");
        // Assistant row tokens column = 345 (output), cumulative after it =
        // 12000 + 345 = 12345.
        let exchanges = vec![
            Exchange {
                user: &u1,
                assistants: vec![&a1],
            },
            Exchange {
                user: &u2,
                assistants: Vec::new(),
            },
        ];
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            Thresholds::default(),
        );
        let lines: Vec<&str> = out.lines().collect();
        let assistant_line = lines
            .iter()
            .find(|l| l.contains("reply"))
            .expect("assistant row missing");
        let orphan_line = lines
            .iter()
            .find(|l| l.contains("orphan"))
            .expect("orphan row missing");

        // Orphan row's tokens column shows the em-dash and cumulative is
        // preserved from the prior row.
        assert!(orphan_line.contains('—'));
        assert!(orphan_line.contains("12345"));

        // Strip the trailing content column AND the cum_cost column so
        // the cumulative column sits at the new right edge. Both rows
        // here use an empty catalog, so every cum_cost cell renders `—`.
        // Pop content first (variable text), then pop the trailing `—`,
        // then we can pin on the numeric `cumulative` value.
        let strip_to_cumulative = |l: &str, content_marker: &str| {
            let trimmed = l.trim_end();
            let cut = trimmed
                .rfind(content_marker)
                .expect("content marker should be in the content column");
            let after_content = trimmed[..cut].trim_end();
            after_content
                .strip_suffix('—')
                .expect("cum_cost should be em-dash for unknown-model session")
                .trim_end()
                .to_string()
        };
        let a_cols = strip_to_cumulative(assistant_line, "reply");
        let o_cols = strip_to_cumulative(orphan_line, "orphan");
        assert!(a_cols.ends_with("12345"), "got: {a_cols}");
        assert!(o_cols.ends_with("12345"), "got: {o_cols}");
        // Right-aligned: the trailing cumulative columns end at the same
        // character position, so the preceding lines (through tokens + spaces +
        // cumulative) have the same scalar count. Compare by chars, not bytes,
        // because `—` is 1 scalar but 3 UTF-8 bytes — `.len()` would report a
        // spurious 2-byte difference even when columns are visually aligned.
        assert_eq!(a_cols.chars().count(), o_cols.chars().count());
    }

    #[test]
    fn render_session_cumulative_reaches_sum_of_billable() {
        let u1 = show_user_turn("q1", "2026-04-01T10:00:00Z");
        let a1 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "r1" }]),
            100,
            50,
            200,
            "2026-04-01T10:01:00Z",
        );
        let u2 = show_user_turn("q2", "2026-04-01T10:02:00Z");
        let a2 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "r2" }]),
            10,
            20,
            0,
            "2026-04-01T10:03:00Z",
        );
        let exchanges = vec![
            Exchange {
                user: &u1,
                assistants: vec![&a1],
            },
            Exchange {
                user: &u2,
                assistants: vec![&a2],
            },
        ];
        // Expected billable total = (100+50+200) + (10+20+0) = 380.
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            Thresholds::default(),
        );
        let last_line = out
            .lines()
            .rfind(|l| l.contains("r2"))
            .expect("final assistant row missing");
        assert!(
            last_line.contains(" 380"),
            "expected final cumulative 380 on last assistant row; got: {last_line}",
        );
    }

    #[test]
    fn render_session_right_aligns_numeric_columns() {
        let u1 = show_user_turn("small", "2026-04-01T10:00:00Z");
        let a1 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "s" }]),
            0,
            9,
            0,
            "2026-04-01T10:01:00Z",
        );
        let u2 = show_user_turn("big", "2026-04-01T10:02:00Z");
        let a2 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "b" }]),
            0,
            123_456,
            0,
            "2026-04-01T10:03:00Z",
        );
        let exchanges = vec![
            Exchange {
                user: &u1,
                assistants: vec![&a1],
            },
            Exchange {
                user: &u2,
                assistants: vec![&a2],
            },
        ];
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            Thresholds::default(),
        );
        // After Phase 3 the column order is:
        //   datetime | role | tokens | cost | cumulative | cum_cost | content
        // With an empty catalog, both `cost` and `cum_cost` render as
        // `—`. Strip content, then the trailing `—` (cum_cost) so the
        // cumulative column sits at the right edge.
        let a_small = out
            .lines()
            .find(|l| l.trim_end().ends_with(" s"))
            .expect("small assistant row missing");
        let a_big = out
            .lines()
            .find(|l| l.trim_end().ends_with(" b"))
            .expect("big assistant row missing");
        let strip_to_cumulative = |l: &str, tail: &str| {
            let after_content = l.trim_end().strip_suffix(tail).unwrap_or(l).trim_end();
            after_content
                .strip_suffix('—')
                .expect("cum_cost should be em-dash with empty catalog")
                .trim_end()
                .to_string()
        };
        let small_cols = strip_to_cumulative(a_small, "s");
        let big_cols = strip_to_cumulative(a_big, "b");
        // Cumulative column values: 9 for small row, 9 + 123456 = 123465 for
        // big row. Widths of tokens + cost + cumulative parts must
        // match (all three are right-aligned to the same boundaries).
        assert!(small_cols.ends_with('9'), "got: {small_cols}");
        assert!(big_cols.ends_with("123465"), "got: {big_cols}");
        assert_eq!(small_cols.chars().count(), big_cols.chars().count());
    }

    #[test]
    fn render_session_tool_use_suffix_appears_on_assistant_row() {
        let u = show_user_turn("q", "2026-04-01T10:00:00Z");
        let a = show_assistant_turn(
            serde_json::json!([
                { "type": "text", "text": "reading" },
                { "type": "tool_use", "name": "Read", "id": "1", "input": {} },
                { "type": "tool_use", "name": "Bash", "id": "2", "input": {} },
            ]),
            0,
            1,
            0,
            "2026-04-01T10:01:00Z",
        );
        let exchanges = vec![Exchange {
            user: &u,
            assistants: vec![&a],
        }];
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            Thresholds::default(),
        );
        assert!(
            out.contains("reading +2 tool uses"),
            "expected tool-use suffix; got:\n{out}",
        );
    }

    // --- FilterArgs ---

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
