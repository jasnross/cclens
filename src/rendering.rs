//! Comfy-table rendering for the `list`, `show`, and `inputs`
//! subcommands.
//!
//! Public API:
//! - `render_table(&[Session]) -> String` — the `list` view.
//! - `render_session(&[Exchange<'_>], &PricingCatalog, Thresholds) -> (String, usize)`
//!   — the `show` view; returns rendered table plus visible-row count
//!   for the empty-result hint decision.
//! - `render_inputs(&[AttributionRow], &CoverageStats) -> String` —
//!   the `inputs` view; returns the table plus a per-tier coverage
//!   line below it.
//!
//! All formatter helpers (`format_cost_opt`, `truncate_title`,
//! `format_local`, `format_local_or_empty`) and the cumulative-fold
//! helpers are module-private.

use chrono::{DateTime, Utc};
use comfy_table::presets::NOTHING;
use comfy_table::{CellAlignment, Table};
use serde_json::Value;

use crate::aggregation::{Exchange, exchange_filter_totals, user_display_string};
use crate::attribution::{AttributionRow, CoverageStats, TierCoverage};
use crate::domain::{CacheCreation, Session, Turn, Usage};
use crate::filter::Thresholds;
use crate::inventory::{CacheTier, ContextFileKind};
use crate::pricing::PricingCatalog;

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

// `inputs` view column indices. Header order:
//   file | kind | tier | tokens | events | billed | attributed_cost
// Note that the `tier` column shifts the numeric columns one slot
// right vs. the original (no-tier) header layout.
const INPUTS_TOKENS_COL_INDEX: usize = 3;
const INPUTS_EVENTS_COL_INDEX: usize = 4;
const INPUTS_BILLED_COL_INDEX: usize = 5;
const INPUTS_ATTRIBUTED_COST_COL_INDEX: usize = 6;

const INPUTS_PATH_MAX_CHARS: usize = 60;

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

#[must_use]
pub fn render_table(sessions: &[Session]) -> String {
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
    catalog: &PricingCatalog,
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
#[must_use]
pub fn render_session(
    exchanges: &[Exchange<'_>],
    catalog: &PricingCatalog,
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

// ---- inputs view ----

/// Render the `inputs` view: one table row per `AttributionRow` plus a
/// per-tier coverage line below.
///
/// `attributed_cost` (rather than `cost`) names the column to
/// distinguish it from the per-session / per-exchange `cost` columns
/// in `list` and `show` — those are billed totals, this is a
/// per-file estimate.
#[must_use]
pub fn render_inputs(rows: &[AttributionRow], coverage: &CoverageStats) -> String {
    let mut table = Table::new();
    table.load_preset(NOTHING);
    table.set_header(vec![
        "file",
        "kind",
        "tier",
        "tokens",
        "events",
        "billed",
        "attributed_cost",
    ]);
    for row in rows {
        table.add_row(vec![
            pretty_path(&row.file.path),
            kind_label(&row.file.kind),
            tier_label(row.file.kind.tier()),
            row.file.tokens.to_string(),
            row.events.to_string(),
            row.estimated_tokens_billed.to_string(),
            format_cost_opt(row.attributed_cost),
        ]);
    }
    for idx in [
        INPUTS_TOKENS_COL_INDEX,
        INPUTS_EVENTS_COL_INDEX,
        INPUTS_BILLED_COL_INDEX,
        INPUTS_ATTRIBUTED_COST_COL_INDEX,
    ] {
        if let Some(col) = table.column_mut(idx) {
            col.set_cell_alignment(CellAlignment::Right);
        }
    }
    let table_str = format!("{table}");
    format!("{table_str}\n{}", coverage_line(coverage))
}

/// Short label for a context-file kind. Enumerates every variant
/// (no wildcard) so adding a new variant forces a label decision at
/// compile time via `wildcard_enum_match_arm`.
fn kind_label(kind: &ContextFileKind) -> String {
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

fn tier_label(tier: CacheTier) -> String {
    match tier {
        CacheTier::Long1h => "1h".to_string(),
        CacheTier::Short5m => "5m".to_string(),
    }
}

/// Convert an absolute path to its `~/...` form when it lives under
/// the user's home directory, then truncate scalar-aware to
/// `INPUTS_PATH_MAX_CHARS` to keep the column readable.
fn pretty_path(path: &std::path::Path) -> String {
    let display: String = if let Some(home) = dirs::home_dir()
        && let Ok(rel) = path.strip_prefix(&home)
    {
        format!("~/{}", rel.display())
    } else {
        path.display().to_string()
    };
    truncate_title(&display, INPUTS_PATH_MAX_CHARS)
}

fn coverage_line(coverage: &CoverageStats) -> String {
    let one_h = coverage_half("1h", &coverage.long_1h);
    let five_m = coverage_half("5m", &coverage.short_5m);
    format!("coverage: {one_h} | {five_m}")
}

#[allow(clippy::cast_precision_loss)]
fn coverage_half(label: &str, tier: &TierCoverage) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    // --- test helpers ---

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

    fn session_for_render(
        project: &str,
        title: &str,
        total_billable: u64,
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
            total_cost: None,
        }
    }

    // --- truncate_title ---

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

    // --- format_local ---

    #[test]
    fn format_local_matches_explicit_chrono_composition() {
        let ts: DateTime<Utc> = "2026-04-01T10:30:00Z".parse().unwrap();
        let expected = ts
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M")
            .to_string();
        assert_eq!(format_local(ts), expected);
    }

    // --- render_table ---

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

    #[test]
    fn render_session_header_includes_all_seven_columns() {
        let (out, rows_shown) =
            render_session(&[], &PricingCatalog::empty(), Thresholds::default());
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
        let (out, _) = render_session(&exchanges, &PricingCatalog::empty(), Thresholds::default());
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
        let (out, _) = render_session(&exchanges, &PricingCatalog::empty(), Thresholds::default());
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
        let (out, _) = render_session(&exchanges, &PricingCatalog::empty(), Thresholds::default());
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
        let (out, _) = render_session(&exchanges, &PricingCatalog::empty(), Thresholds::default());
        assert!(
            out.contains("reading +2 tool uses"),
            "expected tool-use suffix; got:\n{out}",
        );
    }

    // --- render_inputs ---

    use std::path::PathBuf as StdPathBuf;

    use crate::attribution::TierCoverage;
    use crate::inventory::{ContextFile, ContextFileKind, Scope};

    fn inputs_row(
        path: &str,
        kind: ContextFileKind,
        tokens: u64,
        events: u64,
        billed: u64,
        cost: Option<f64>,
    ) -> AttributionRow {
        AttributionRow {
            file: ContextFile {
                path: StdPathBuf::from(path),
                kind,
                tokens,
                scope: Scope::Global,
            },
            events,
            estimated_tokens_billed: billed,
            attributed_cost: cost,
        }
    }

    fn cov_with(
        long_obs: u64,
        long_attr: u64,
        long_ratio: Option<f64>,
        short_obs: u64,
        short_attr: u64,
        short_ratio: Option<f64>,
    ) -> CoverageStats {
        CoverageStats {
            long_1h: TierCoverage {
                observed_tokens: long_obs,
                attributed_tokens: long_attr,
                ratio: long_ratio,
            },
            short_5m: TierCoverage {
                observed_tokens: short_obs,
                attributed_tokens: short_attr,
                ratio: short_ratio,
            },
        }
    }

    #[test]
    fn render_inputs_renders_header_and_rows() {
        let rows = vec![inputs_row(
            "/some/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            5,
            500,
            Some(0.003),
        )];
        let cov = cov_with(500, 500, Some(1.0), 0, 0, None);
        let out = render_inputs(&rows, &cov);
        for header in [
            "file",
            "kind",
            "tier",
            "tokens",
            "events",
            "billed",
            "attributed_cost",
        ] {
            assert!(out.contains(header), "header `{header}` missing in:\n{out}");
        }
        assert!(
            out.contains("global"),
            "expected `global` kind label; got:\n{out}",
        );
        assert!(out.contains("1h"));
        assert!(out.contains("$0.0030"));
    }

    #[test]
    fn render_inputs_right_aligns_numeric_columns() {
        // Two rows with very different scalar widths; the
        // events/billed/cost columns must right-align so each row's
        // prefix-up-to-events ends at the same character count.
        let rows = vec![
            inputs_row(
                "/a/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                9,
                9,
                81,
                Some(0.001),
            ),
            inputs_row(
                "/b/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                123_456,
                123_456,
                15_241_383_936,
                Some(123.4567),
            ),
        ];
        let cov = cov_with(0, 0, None, 0, 0, None);
        let out = render_inputs(&rows, &cov);
        let data_lines: Vec<&str> = out
            .lines()
            .filter(|l| l.contains("/a/CLAUDE.md") || l.contains("/b/CLAUDE.md"))
            .collect();
        assert_eq!(data_lines.len(), 2);
        // Strip from the right: cost cell, billed cell, events cell —
        // what remains should end at the events column's right edge.
        let strip = |l: &str| -> usize {
            let trimmed = l.trim_end();
            // Cost is the last cell; strip it off.
            let after_cost = trimmed.rsplit_once(' ').map(|(left, _)| left).unwrap();
            let after_billed = after_cost
                .trim_end()
                .rsplit_once(' ')
                .map(|(left, _)| left)
                .unwrap();
            after_billed.trim_end().chars().count()
        };
        let small_width = strip(data_lines.iter().find(|l| l.contains("/a/")).unwrap());
        let large_width = strip(data_lines.iter().find(|l| l.contains("/b/")).unwrap());
        assert_eq!(
            small_width, large_width,
            "right-aligned columns should produce equal scalar widths; \
             small={small_width}, large={large_width}\n\nfull output:\n{out}"
        );
    }

    #[test]
    fn render_inputs_renders_em_dash_for_unknown_cost() {
        let rows = vec![inputs_row(
            "/some/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            5,
            500,
            None,
        )];
        let cov = cov_with(0, 0, None, 0, 0, None);
        let out = render_inputs(&rows, &cov);
        assert!(out.contains('—'), "expected em-dash; got:\n{out}");
    }

    #[test]
    fn render_inputs_coverage_line_includes_both_tiers() {
        let cov = cov_with(5_000, 4_000, Some(0.8), 2_000, 1_500, Some(0.75));
        let out = render_inputs(&[], &cov);
        let coverage_line = out
            .lines()
            .find(|l| l.starts_with("coverage:"))
            .expect("coverage line present");
        assert!(coverage_line.contains("1h: 80.0%"), "got: {coverage_line}");
        assert!(
            coverage_line.contains("(4000 / 5000 1h-tokens)"),
            "got: {coverage_line}"
        );
        assert!(coverage_line.contains('|'), "got: {coverage_line}");
        assert!(coverage_line.contains("5m: 75.0%"), "got: {coverage_line}");
        assert!(
            coverage_line.contains("(1500 / 2000 5m-tokens)"),
            "got: {coverage_line}"
        );
    }

    #[test]
    fn render_inputs_coverage_line_renders_n_a_per_tier() {
        // Asymmetric: 1h has data, 5m doesn't.
        let cov = cov_with(100, 80, Some(0.8), 0, 0, None);
        let out = render_inputs(&[], &cov);
        let line = out.lines().find(|l| l.starts_with("coverage:")).unwrap();
        assert!(line.contains("1h: 80.0%"), "got: {line}");
        assert!(line.contains("5m: n/a"), "got: {line}");

        // Reversed shape.
        let cov = cov_with(0, 0, None, 100, 80, Some(0.8));
        let out = render_inputs(&[], &cov);
        let line = out.lines().find(|l| l.starts_with("coverage:")).unwrap();
        assert!(line.contains("1h: n/a"), "got: {line}");
        assert!(line.contains("5m: 80.0%"), "got: {line}");
    }

    #[test]
    fn kind_label_enumerates_every_variant() {
        // Smoke: every variant maps to a non-empty string. The
        // wildcard_enum_match_arm lint already prevents adding a new
        // variant without updating the function — this test pins the
        // *non-empty* property too.
        let kinds = [
            ContextFileKind::GlobalClaudeMd,
            ContextFileKind::UserRule,
            ContextFileKind::UserSkill,
            ContextFileKind::UserAgent,
            ContextFileKind::PluginSkill {
                plugin: "p".into(),
                marketplace: "m".into(),
            },
            ContextFileKind::PluginRule {
                plugin: "p".into(),
                marketplace: "m".into(),
            },
            ContextFileKind::PluginAgent {
                plugin: "p".into(),
                marketplace: "m".into(),
            },
            ContextFileKind::ProjectClaudeMd,
            ContextFileKind::ProjectLocalSkill,
            ContextFileKind::ProjectLocalCommand,
            ContextFileKind::ProjectLocalRule,
            ContextFileKind::ProjectLocalAgent,
        ];
        for k in &kinds {
            let label = kind_label(k);
            assert!(!label.is_empty(), "label for {k:?} should not be empty");
        }
    }

    #[test]
    fn pretty_path_replaces_home_with_tilde() {
        let Some(home) = dirs::home_dir() else {
            return; // Hermetic skip on platforms without a home dir.
        };
        let under_home = home.join("foo/bar");
        let displayed = pretty_path(&under_home);
        assert!(
            displayed.starts_with("~/"),
            "expected ~/ prefix, got: {displayed}",
        );
        let elsewhere = StdPathBuf::from("/var/tmp/elsewhere.md");
        let displayed = pretty_path(&elsewhere);
        assert!(
            displayed.starts_with('/'),
            "non-home path should render absolute, got: {displayed}",
        );
    }
}
