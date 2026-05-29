//! Comfy-table rendering for the `list`, `show`, `inputs`, and
//! `pricing list` subcommands.
//!
//! Public API:
//! - `render_table(&[Session]) -> String` — the `list` view; appends a
//!   totals row beneath the per-session rows when 2+ sessions are
//!   visible. The totals row's `title` cell carries a right-aligned
//!   `total` label flush against the tokens column; its numeric cells
//!   use the same formatters as data rows. Cost follows the same
//!   strict-`None` propagation as `Session.cost_breakdown` — any
//!   visible session with `cost_breakdown: None` collapses the totals
//!   cost cell to `—`.
//! - `render_session(&[PreparedExchange]) -> (String, usize)` — the
//!   `show` view; receives pre-computed exchanges and does pure layout.
//!   Each `PreparedExchange` expands its nested `PreparedRow`s into
//!   table rows. Domain computation (cost, cumulatives, filtering)
//!   lives in `aggregation::prepare_exchanges`.
//! - `render_inputs(&[AttributionRow], &CoverageStats) -> String` —
//!   the `inputs` view; returns the table plus a per-tier coverage
//!   line below it.
//! - `render_prices(&[(&str, &ClaudePricing)]) -> String` — the
//!   `pricing list` view; renders per-model rates in `$/MTok` with
//!   conditional sub-rows for models whose above-200k rates differ.
//!
//! Show-view content cells (every row, parent and subagent) are
//! truncated to `SHOW_CONTENT_MAX_CHARS` so a long subagent prefix
//! (or any other oversized content) cannot break the
//! one-line-per-row invariant.
//!
//! Format helpers (`format_cost_opt`, `format_local`, `format_tokens`)
//! live in `formatting` — shared with the `tui` module for visual
//! consistency. Content-preview and cumulative-fold helpers live in
//! `aggregation`.

use comfy_table::presets::NOTHING;
use comfy_table::{Cell, CellAlignment, Table};

use crate::aggregation::{PreparedExchange, PreparedRowRole, fold_cum_cost};
use crate::attribution::{AttributionRow, CoverageStats};
use crate::domain::{CostBreakdown, Session};
use crate::formatting::{
    coverage_line, display_path, format_cost_opt, format_local, format_local_or_empty,
    format_rate_mtok, format_tokens, kind_label, tiers_differ,
};
use crate::pricing::ClaudePricing;

const TITLE_MAX_CHARS: usize = 80;

/// Per-row content-cell width cap for `render_session`. Wider than
/// `TITLE_MAX_CHARS` because show rows already have less title-space
/// pressure (the columns to the left are narrower than `list`'s), but
/// still bounded so subagent prefixes (`(<agent_type> · "<desc>") …`)
/// don't break the one-line-per-row invariant by overflowing into
/// comfy-table's wrap.
const SHOW_CONTENT_MAX_CHARS: usize = 120;

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
//   file | kind | tier | tokens | loads | billed | attributed_cost
// Note that the `tier` column shifts the numeric columns one slot
// right vs. the original (no-tier) header layout.
const INPUTS_TOKENS_COL_INDEX: usize = 3;
const INPUTS_LOADS_COL_INDEX: usize = 4;
const INPUTS_BILLED_COL_INDEX: usize = 5;
const INPUTS_ATTRIBUTED_COST_COL_INDEX: usize = 6;

const INPUTS_PATH_MAX_CHARS: usize = 60;

// `pricing list` view column indices. Header order:
//   model | tier | input | output | cache_rd | cache_5m | cache_1h
const PRICES_INPUT_COL_INDEX: usize = 2;
const PRICES_OUTPUT_COL_INDEX: usize = 3;
const PRICES_CACHE_RD_COL_INDEX: usize = 4;
const PRICES_CACHE_5M_COL_INDEX: usize = 5;
const PRICES_CACHE_1H_COL_INDEX: usize = 6;

/// Format a decomposed cost breakdown showing non-zero components.
/// `None` renders as `—`; all-zero renders as `$0.0000`; otherwise
/// space-separated `label:$X.XXXX` for each non-zero component.
fn format_cost_breakdown(breakdown: Option<CostBreakdown>) -> String {
    let Some(b) = breakdown else {
        return "—".to_string();
    };
    let parts: Vec<String> = [
        ("in", b.input),
        ("out", b.output),
        ("c5m", b.cache_creation_5m),
        ("c1h", b.cache_creation_1h),
        ("cr", b.cache_read),
    ]
    .into_iter()
    .filter(|(_, v)| *v > 0.0)
    .map(|(label, v)| format!("{label}:${v:.4}"))
    .collect();
    if parts.is_empty() {
        "$0.0000".to_string()
    } else {
        parts.join(" ")
    }
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
            format_tokens(session.total_billable),
            format_cost_opt(session.cost_breakdown.map(|b| b.total())),
            session.id.clone(),
        ]);
    }
    add_totals_row(&mut table, sessions);
    // column_mut returns Option; the columns are guaranteed present because
    // the header above defines them at the indices. The totals row's
    // numeric cells inherit this column-level right alignment; its `total`
    // label cell carries a per-cell right-alignment override set inside
    // `add_totals_row` so it stays flush against the tokens column.
    if let Some(col) = table.column_mut(TOKENS_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(COST_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    format!("{table}")
}

/// Append a totals row when 2+ sessions are visible. The `title` cell
/// carries a right-aligned `total` label flush against the tokens
/// column; numeric cells use the same formatters as data rows and
/// inherit the column-level right alignment applied by `render_table`
/// after this call. Cost follows strict-`None` propagation via
/// `fold_cum_cost`: any visible session with `total_cost: None`
/// collapses the totals cost cell to `—`.
fn add_totals_row(table: &mut Table, sessions: &[Session]) {
    if sessions.len() < 2 {
        return;
    }
    let total_tokens: u64 = sessions.iter().map(|s| s.total_billable).sum();
    // `try_fold` short-circuits on the first `None`, matching the
    // strict-`None` contract that `fold_cum_cost` enforces in the
    // show view's `cum_cost` column. Reusing `fold_cum_cost` keeps
    // the propagation rule single-sourced.
    let total_cost: Option<f64> = sessions.iter().try_fold(0.0, |acc, s| {
        fold_cum_cost(Some(acc), s.cost_breakdown.map(|b| b.total()))
    });
    table.add_row(vec![
        Cell::new(""),
        Cell::new(""),
        Cell::new("total").set_alignment(CellAlignment::Right),
        Cell::new(format_tokens(total_tokens)),
        Cell::new(format_cost_opt(total_cost)),
        Cell::new(""),
    ]);
}

/// Render pre-computed exchanges into the `show` view table.
///
/// Returns `(rendered, rows_shown)`. Each `PreparedExchange` expands
/// its nested `PreparedRow`s into table rows — the renderer does pure
/// layout with no domain computation.
#[must_use]
pub fn render_session(prepared: &[PreparedExchange]) -> (String, usize) {
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
    let mut rows_shown: usize = 0;

    for exchange in prepared {
        for row in &exchange.rows {
            let role_str = match row.role {
                PreparedRowRole::User => "user",
                PreparedRowRole::Assistant => "assistant",
                PreparedRowRole::Subagent => "subagent",
            };
            let content = if row.tool_use_count > 0 {
                format!("{} +{} tool uses", row.content, row.tool_use_count)
            } else {
                row.content.clone()
            };
            table.add_row(vec![
                format_local_or_empty(row.timestamp),
                role_str.to_string(),
                row.tokens.map_or_else(|| "—".to_string(), format_tokens),
                format_cost_breakdown(row.cost),
                format_tokens(row.cumulative_tokens),
                format_cost_opt(row.cumulative_cost),
                truncate_title(&content, SHOW_CONTENT_MAX_CHARS),
            ]);
            rows_shown += 1;
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

// ---- pricing list view ----

#[must_use]
pub fn render_prices(entries: &[(&str, &ClaudePricing)]) -> String {
    let mut table = Table::new();
    table.load_preset(NOTHING);
    table.set_header(vec![
        "model", "tier", "input", "output", "cache_rd", "cache_5m", "cache_1h",
    ]);
    for &(model, pricing) in entries {
        if tiers_differ(pricing) {
            table.add_row(vec![
                model.to_string(),
                "\u{2264}200k".to_string(),
                format_rate_mtok(pricing.input.first_200k_rate),
                format_rate_mtok(pricing.output.first_200k_rate),
                format_rate_mtok(pricing.cache_read.first_200k_rate),
                format_rate_mtok(pricing.cache_creation_5m.first_200k_rate),
                format_rate_mtok(pricing.cache_creation_1h.first_200k_rate),
            ]);
            table.add_row(vec![
                String::new(),
                ">200k".to_string(),
                format_rate_mtok(pricing.input.above_200k_rate),
                format_rate_mtok(pricing.output.above_200k_rate),
                format_rate_mtok(pricing.cache_read.above_200k_rate),
                format_rate_mtok(pricing.cache_creation_5m.above_200k_rate),
                format_rate_mtok(pricing.cache_creation_1h.above_200k_rate),
            ]);
        } else {
            table.add_row(vec![
                model.to_string(),
                String::new(),
                format_rate_mtok(pricing.input.first_200k_rate),
                format_rate_mtok(pricing.output.first_200k_rate),
                format_rate_mtok(pricing.cache_read.first_200k_rate),
                format_rate_mtok(pricing.cache_creation_5m.first_200k_rate),
                format_rate_mtok(pricing.cache_creation_1h.first_200k_rate),
            ]);
        }
    }
    for idx in [
        PRICES_INPUT_COL_INDEX,
        PRICES_OUTPUT_COL_INDEX,
        PRICES_CACHE_RD_COL_INDEX,
        PRICES_CACHE_5M_COL_INDEX,
        PRICES_CACHE_1H_COL_INDEX,
    ] {
        if let Some(col) = table.column_mut(idx) {
            col.set_cell_alignment(CellAlignment::Right);
        }
    }
    let table_str = format!("{table}");
    format!("{table_str}\n\nRates in $/MTok (dollars per million tokens).")
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
        "loads",
        "billed",
        "attributed_cost",
    ]);
    for row in rows {
        table.add_row(vec![
            pretty_path(&row.file.path),
            kind_label(&row.file.kind),
            row.tier_label().to_string(),
            format_tokens(row.file.tokens),
            row.total_loads().to_string(),
            format_tokens(row.estimated_tokens_billed),
            format_cost_opt(row.attributed_cost),
        ]);
    }
    for idx in [
        INPUTS_TOKENS_COL_INDEX,
        INPUTS_LOADS_COL_INDEX,
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

fn pretty_path(path: &std::path::Path) -> String {
    truncate_title(&display_path(path), INPUTS_PATH_MAX_CHARS)
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};

    use super::*;
    use crate::aggregation::PreparedRow;
    use crate::domain::TurnOrigin;

    // --- test helpers ---

    fn test_row(
        role: PreparedRowRole,
        tokens: Option<u64>,
        cum_tokens: u64,
        cum_cost: Option<f64>,
        content: &str,
    ) -> PreparedRow {
        PreparedRow {
            timestamp: None,
            role,
            tokens,
            cost: None,
            cumulative_tokens: cum_tokens,
            cumulative_cost: cum_cost,
            content: content.to_string(),
            tool_use_count: 0,
        }
    }

    fn parent_exchange(rows: Vec<PreparedRow>) -> PreparedExchange {
        PreparedExchange {
            origin: TurnOrigin::Parent,
            rows,
        }
    }

    fn subagent_exchange(
        agent_type: &str,
        description: Option<&str>,
        rows: Vec<PreparedRow>,
    ) -> PreparedExchange {
        PreparedExchange {
            origin: TurnOrigin::Subagent {
                agent_type: agent_type.to_string(),
                description: description.map(str::to_string),
            },
            rows,
        }
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
            cost_breakdown: total_cost.map(|c| CostBreakdown {
                output: c,
                ..CostBreakdown::default()
            }),
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
        assert!(out.contains("0.10k"));
        assert!(out.contains("0.25k"));
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
        assert!(end_of_9.ends_with("0.01k"), "got: {end_of_9}");
        assert!(end_of_123456.ends_with("123.46k"), "got: {end_of_123456}");
        // Right-alignment: shorter value has leading whitespace padding,
        // so the prefix-up-to-tokens has the same scalar count for both.
        assert_eq!(end_of_9.chars().count(), end_of_123456.chars().count());
    }

    // --- render_table totals row ---

    // Returns the single totals row (if any) from rendered output. The
    // totals row is the only line whose `title`-column cell is the
    // literal `total`; data rows place project names and titles there
    // (the test fixtures use innocuous strings that don't contain
    // `total` as a substring).
    fn find_totals_row(rendered: &str) -> Option<String> {
        rendered
            .lines()
            .find(|l| l.contains("total"))
            .map(str::to_string)
    }

    #[test]
    fn render_table_appends_totals_row_when_two_or_more_sessions() {
        let sessions = vec![
            session_for_render_with_cost(
                "alpha",
                "first",
                1500,
                Some(0.01),
                "2026-04-01T10:00:00Z",
            ),
            session_for_render_with_cost(
                "beta",
                "second",
                2500,
                Some(0.02),
                "2026-04-02T10:00:00Z",
            ),
        ];
        let out = render_table(&sessions);
        let totals = find_totals_row(&out).expect("totals row missing");
        // 1500 + 2500 = 4000 → "4.00k"; 0.01 + 0.02 = 0.03 → "$0.0300".
        assert!(totals.contains("total"), "row missing label: {totals}");
        assert!(totals.contains("4.00k"), "row missing token sum: {totals}");
        assert!(totals.contains("$0.0300"), "row missing cost sum: {totals}");
        // Sanity: it really is the last non-empty line of output.
        let last_non_empty = out
            .lines()
            .rfind(|l| !l.trim().is_empty())
            .expect("output must have at least one line");
        assert!(
            last_non_empty.contains("total"),
            "totals row must be the final non-empty line; got: {last_non_empty}"
        );
    }

    #[test]
    fn render_table_totals_row_collapses_cost_to_em_dash_on_any_none() {
        let sessions = vec![
            session_for_render_with_cost(
                "alpha",
                "first",
                1000,
                Some(0.01),
                "2026-04-01T10:00:00Z",
            ),
            // Middle session is unknown-model: total_cost: None should
            // latch the totals cost cell to `—` even though the other
            // two are priced.
            session_for_render_with_cost("beta", "second", 2000, None, "2026-04-02T10:00:00Z"),
            session_for_render_with_cost(
                "gamma",
                "third",
                3000,
                Some(0.03),
                "2026-04-03T10:00:00Z",
            ),
        ];
        let out = render_table(&sessions);
        let totals = find_totals_row(&out).expect("totals row missing");
        // Tokens always sum (no None propagation on u64): 1000+2000+3000=6000.
        assert!(totals.contains("6.00k"), "row missing token sum: {totals}");
        // Cost collapses to em-dash; no `$` should appear in the totals
        // row, since the only cost-shaped value is the em-dash.
        assert!(
            totals.contains('—'),
            "totals cost cell should render as em-dash: {totals}"
        );
        assert!(
            !totals.contains('$'),
            "totals cost cell must not show a dollar amount when any session is None: {totals}"
        );
    }

    #[test]
    fn render_table_omits_totals_row_for_single_session() {
        let sessions = vec![session_for_render_with_cost(
            "alpha",
            "only one",
            1234,
            Some(0.0123),
            "2026-04-01T10:00:00Z",
        )];
        let out = render_table(&sessions);
        // The single data row does not contain "total" (title is "only
        // one", project "alpha"), and no totals row should be added.
        assert!(
            !out.contains("total"),
            "single-session output must not include a totals row; got:\n{out}"
        );
    }

    #[test]
    fn render_table_omits_totals_row_for_empty_input() {
        let out = render_table(&[]);
        // Header columns do not include the substring "total" (the
        // closest is `title`, which does not contain it).
        assert!(
            !out.contains("total"),
            "empty input must produce header-only output (no totals row); got:\n{out}"
        );
    }

    // --- render_session ---

    #[test]
    fn render_session_header_includes_all_seven_columns() {
        let (out, rows_shown) = render_session(&[]);
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
        let exchanges = vec![
            parent_exchange(vec![
                test_row(PreparedRowRole::User, Some(12000), 12000, None, "first"),
                test_row(PreparedRowRole::Assistant, Some(345), 12345, None, "reply"),
            ]),
            parent_exchange(vec![test_row(
                PreparedRowRole::User,
                None,
                12345,
                None,
                "orphan",
            )]),
        ];
        let (out, _) = render_session(&exchanges);
        let lines: Vec<&str> = out.lines().collect();
        let assistant_line = lines
            .iter()
            .find(|l| l.contains("reply"))
            .expect("assistant row missing");
        let orphan_line = lines
            .iter()
            .find(|l| l.contains("orphan"))
            .expect("orphan row missing");

        assert!(orphan_line.contains('—'));
        assert!(orphan_line.contains(&format_tokens(12345)));

        let strip_to_cumulative = |l: &str, content_marker: &str| {
            let trimmed = l.trim_end();
            let cut = trimmed
                .rfind(content_marker)
                .expect("content marker should be in the content column");
            let after_content = trimmed[..cut].trim_end();
            after_content
                .strip_suffix('—')
                .expect("cum_cost should be em-dash")
                .trim_end()
                .to_string()
        };
        let a_cols = strip_to_cumulative(assistant_line, "reply");
        let o_cols = strip_to_cumulative(orphan_line, "orphan");
        assert!(a_cols.ends_with(&format_tokens(12345)), "got: {a_cols}");
        assert!(o_cols.ends_with(&format_tokens(12345)), "got: {o_cols}");
        assert_eq!(a_cols.chars().count(), o_cols.chars().count());
    }

    #[test]
    fn render_session_cumulative_reaches_sum_of_billable() {
        let exchanges = vec![
            parent_exchange(vec![
                test_row(PreparedRowRole::User, Some(300), 300, None, "q1"),
                test_row(PreparedRowRole::Assistant, Some(50), 350, None, "r1"),
            ]),
            parent_exchange(vec![
                test_row(PreparedRowRole::User, Some(10), 360, None, "q2"),
                test_row(PreparedRowRole::Assistant, Some(20), 380, None, "r2"),
            ]),
        ];
        let (out, _) = render_session(&exchanges);
        let last_line = out
            .lines()
            .rfind(|l| l.contains("r2"))
            .expect("final assistant row missing");
        assert!(
            last_line.contains(" 0.38k"),
            "expected final cumulative 380 on last assistant row; got: {last_line}",
        );
    }

    #[test]
    fn render_session_right_aligns_numeric_columns() {
        let exchanges = vec![
            parent_exchange(vec![
                test_row(PreparedRowRole::User, Some(0), 0, None, "small"),
                test_row(PreparedRowRole::Assistant, Some(9), 9, None, "s"),
            ]),
            parent_exchange(vec![
                test_row(PreparedRowRole::User, Some(0), 9, None, "big"),
                test_row(
                    PreparedRowRole::Assistant,
                    Some(123_456),
                    123_465,
                    None,
                    "b",
                ),
            ]),
        ];
        let (out, _) = render_session(&exchanges);
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
                .expect("cum_cost should be em-dash")
                .trim_end()
                .to_string()
        };
        let small_cols = strip_to_cumulative(a_small, "s");
        let big_cols = strip_to_cumulative(a_big, "b");
        assert!(small_cols.ends_with(&format_tokens(9)), "got: {small_cols}");
        assert!(
            big_cols.ends_with(&format_tokens(123_465)),
            "got: {big_cols}"
        );
        assert_eq!(small_cols.chars().count(), big_cols.chars().count());
    }

    #[test]
    fn render_session_tool_use_suffix_appears_on_assistant_row() {
        let mut row = test_row(PreparedRowRole::Assistant, Some(1), 1, None, "reading");
        row.tool_use_count = 2;
        let exchanges = vec![parent_exchange(vec![
            test_row(PreparedRowRole::User, Some(0), 0, None, "q"),
            row,
        ])];
        let (out, _) = render_session(&exchanges);
        assert!(
            out.contains("reading +2 tool uses"),
            "expected tool-use suffix; got:\n{out}",
        );
    }

    // --- subagent rendering ---

    #[test]
    fn render_session_renders_subagent_exchange_as_single_row_with_description() {
        let exchanges = vec![subagent_exchange(
            "tw-code-reviewer",
            Some("Review auth changes"),
            vec![test_row(
                PreparedRowRole::Subagent,
                Some(30),
                30,
                None,
                "(tw-code-reviewer · \"Review auth changes\") found 2 issues",
            )],
        )];
        let (out, rows) = render_session(&exchanges);
        assert_eq!(rows, 1, "subagent exchange must render exactly one row");
        assert!(out.contains("subagent"));
        assert!(
            out.contains("(tw-code-reviewer · \"Review auth changes\") found 2 issues"),
            "got:\n{out}",
        );
    }

    #[test]
    fn render_session_renders_subagent_without_description() {
        let exchanges = vec![subagent_exchange(
            "tw-code-reviewer",
            None,
            vec![test_row(
                PreparedRowRole::Subagent,
                Some(30),
                30,
                None,
                "(tw-code-reviewer) all good",
            )],
        )];
        let (out, _) = render_session(&exchanges);
        assert!(out.contains("(tw-code-reviewer) all good"), "got:\n{out}");
        assert!(!out.contains('·'));
    }

    #[test]
    fn render_session_subagent_row_contributes_to_cumulative() {
        let exchanges = vec![
            parent_exchange(vec![
                test_row(PreparedRowRole::User, Some(200), 200, None, "ask"),
                test_row(PreparedRowRole::Assistant, Some(50), 200, None, "thinking"),
            ]),
            subagent_exchange(
                "agent",
                None,
                vec![test_row(
                    PreparedRowRole::Subagent,
                    Some(90),
                    290,
                    None,
                    "(agent) done",
                )],
            ),
        ];
        let (out, _) = render_session(&exchanges);
        let last_data_line = out
            .lines()
            .rfind(|l| l.contains("done"))
            .expect("subagent row missing");
        assert!(
            last_data_line.contains(" 0.29k "),
            "cumulative-at-bottom should be 0.29k; got: {last_data_line}",
        );
    }

    #[test]
    fn render_session_empty_subagent_cluster_renders_no_response_marker() {
        let exchanges = vec![subagent_exchange(
            "tw-code-reviewer",
            Some("Empty case"),
            vec![test_row(
                PreparedRowRole::Subagent,
                None,
                0,
                Some(0.0),
                "(tw-code-reviewer · \"Empty case\") — (no response)",
            )],
        )];
        let (out, rows) = render_session(&exchanges);
        assert_eq!(rows, 1);
        assert!(out.contains("— (no response)"), "got:\n{out}");
        let row = out
            .lines()
            .find(|l| l.contains("subagent"))
            .expect("subagent row missing");
        let dash_count = row.matches('—').count();
        assert!(
            dash_count >= 3,
            "expected `—` in tokens + cost + content; got: {row}",
        );
    }

    #[test]
    fn render_session_truncates_show_content_at_max_chars() {
        let long_text = "x".repeat(SHOW_CONTENT_MAX_CHARS + 50);
        let exchanges = vec![parent_exchange(vec![
            test_row(PreparedRowRole::User, Some(10), 10, None, &long_text),
            test_row(PreparedRowRole::Assistant, Some(5), 15, None, "ok"),
        ])];
        let (out, _) = render_session(&exchanges);
        assert!(
            out.contains('…'),
            "truncated content should end with ellipsis; got:\n{out}",
        );
        assert!(!out.contains(&long_text));
    }

    // --- render_inputs ---

    use std::path::PathBuf as StdPathBuf;

    use crate::attribution::TierCoverage;
    use crate::inventory::{ContextFile, ContextFileKind, Scope};

    #[allow(clippy::similar_names)]
    fn inputs_row(
        path: &str,
        kind: ContextFileKind,
        tokens: u64,
        loads_1h: u64,
        loads_5m: u64,
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
            loads_1h,
            loads_5m,
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
        // Single 1h-tier load → loads_1h=5, loads_5m=0; tier_label() = "1h".
        let rows = vec![inputs_row(
            "/some/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            5,
            0,
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
            "loads",
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
        assert!(out.contains("$0.0030"), "got:\n{out}");
    }

    #[test]
    fn render_inputs_right_aligns_numeric_columns() {
        // Two rows with very different scalar widths; the
        // loads/billed/cost columns must right-align so each row's
        // prefix-up-to-loads ends at the same character count.
        let rows = vec![
            inputs_row(
                "/a/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                9,
                9,
                0,
                81,
                Some(0.001),
            ),
            inputs_row(
                "/b/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                123_456,
                123_456,
                0,
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
        // Strip from the right: cost cell, billed cell, loads cell —
        // what remains should end at the loads column's right edge.
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
            0,
            500,
            None,
        )];
        let cov = cov_with(0, 0, None, 0, 0, None);
        let out = render_inputs(&rows, &cov);
        assert!(out.contains('—'), "expected em-dash; got:\n{out}");
    }

    #[test]
    fn render_inputs_renders_mixed_tier_label() {
        // Loaded at both 1h (parent) and 5m (subagent) — tier column
        // should render `1h+5m` rather than collapsing to one tier.
        let rows = vec![inputs_row(
            "/some/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            1,
            1,
            200,
            Some(0.0012),
        )];
        let cov = cov_with(0, 0, None, 0, 0, None);
        let out = render_inputs(&rows, &cov);
        assert!(
            out.contains("1h+5m"),
            "expected mixed tier label `1h+5m`; got:\n{out}",
        );
    }

    #[test]
    fn render_inputs_renders_em_dash_tier_for_unloaded_row() {
        // In-scope file that no session loaded — both load counts
        // are 0 and the tier column collapses to `—`.
        let rows = vec![inputs_row(
            "/some/agents/never-invoked.md",
            ContextFileKind::UserAgent,
            42,
            0,
            0,
            0,
            Some(0.0),
        )];
        let cov = cov_with(0, 0, None, 0, 0, None);
        let out = render_inputs(&rows, &cov);
        let agent_line = out
            .lines()
            .find(|l| l.contains("never-invoked.md"))
            .expect("agent row present");
        assert!(
            agent_line.contains('—'),
            "expected em-dash tier for unloaded row; got: {agent_line}",
        );
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

    // --- pricing list view ---

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
    fn render_prices_uniform_tier_produces_one_row_per_model() {
        let p = uniform_pricing(3e-6);
        let entries = vec![("claude-haiku-4-5", &p)];
        let out = render_prices(&entries);
        assert!(out.contains("claude-haiku-4-5"));
        assert!(out.contains("$3.00"));
        assert!(out.contains("$/MTok"));
        let data_lines: Vec<&str> = out.lines().filter(|l| l.contains("claude-haiku")).collect();
        assert_eq!(
            data_lines.len(),
            1,
            "uniform-tier model should have exactly one row"
        );
    }

    #[test]
    fn render_prices_split_tier_produces_two_rows() {
        let p = split_pricing();
        let entries = vec![("claude-sonnet-4-5", &p)];
        let out = render_prices(&entries);
        assert!(out.contains("claude-sonnet-4-5"));
        assert!(out.contains("\u{2264}200k"));
        assert!(out.contains(">200k"));
        assert!(out.contains("$3.00"));
        assert!(out.contains("$6.00"));
        let tier_rows: Vec<&str> = out.lines().filter(|l| l.contains("200k")).collect();
        assert_eq!(
            tier_rows.len(),
            2,
            "split-tier model should produce two sub-rows (\u{2264}200k and >200k)"
        );
    }

    #[test]
    fn render_prices_mixed_entries_sort_and_display() {
        let uniform = uniform_pricing(15e-6);
        let split = split_pricing();
        let entries = vec![("claude-opus-4-7", &uniform), ("claude-sonnet-4-5", &split)];
        let out = render_prices(&entries);
        let opus_pos = out.find("claude-opus-4-7").expect("opus present");
        let sonnet_pos = out.find("claude-sonnet-4-5").expect("sonnet present");
        assert!(
            opus_pos < sonnet_pos,
            "entries should appear in order given"
        );
        assert!(out.contains("$15.00"));
        assert!(out.contains("$3.00"));
    }

    #[test]
    fn render_prices_includes_all_header_columns() {
        let out = render_prices(&[]);
        for col in [
            "model", "tier", "input", "output", "cache_rd", "cache_5m", "cache_1h",
        ] {
            assert!(out.contains(col), "header column `{col}` missing");
        }
    }

    // --- pretty_path ---

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
