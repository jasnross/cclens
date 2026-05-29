//! Shared view builders consumed by both `rendering` (comfy-table
//! plain-text path) and `tui` (ratatui interactive path).
//!
//! Each builder maps domain types to display-ready cell values. The
//! two rendering consumers import these builders and adapt the cell
//! data to their respective table frameworks — `rendering` applies
//! truncation and comfy-table alignment, `tui` wraps cells in ratatui
//! `Row`/`Line` widgets with styling and constraints.
//!
//! Public API:
//! - `SessionCells` / `session_cells(&Session)` — list-view row cells.
//! - `ShowRowCells` / `show_row_cells(&PreparedRow)` — show-view row
//!   cells. Returns raw `Option<CostBreakdown>` so each consumer can
//!   format cost differently (TUI decomposes into 5 columns, plain
//!   calls `format_cost_breakdown`).
//! - `InputsCells` / `inputs_cells(&AttributionRow)` — inputs-view
//!   row cells.
//! - `pricing_view_rows(&str, &ClaudePricing) -> Vec<Vec<String>>` —
//!   pricing-view rows (cells identical between paths).
//! - `SessionTotals` / `session_totals(&[Session])` — aggregate
//!   token + cost totals for 2+ sessions.

use crate::aggregation::{PreparedRow, PreparedRowRole, fold_cum_cost};
use crate::attribution::AttributionRow;
use crate::domain::{CostBreakdown, Session};
use crate::formatting::{
    display_path, format_cost_opt, format_local, format_local_or_empty, format_rate_mtok,
    format_tokens, kind_label, tiers_differ,
};
use crate::pricing::ClaudePricing;

pub struct SessionCells {
    pub datetime: String,
    pub project: String,
    pub title: String,
    pub tokens: String,
    pub cost: String,
}

pub struct ShowRowCells {
    pub datetime: String,
    pub role: String,
    pub tokens: String,
    pub cost: Option<CostBreakdown>,
    pub cumulative_tokens: String,
    pub cumulative_cost: String,
    pub content: String,
}

pub struct InputsCells {
    pub file_path: String,
    pub kind: String,
    pub tier: String,
    pub tokens: String,
    pub loads: String,
    pub billed: String,
    pub cost: String,
}

pub struct SessionTotals {
    pub total_tokens: u64,
    pub total_cost: Option<f64>,
}

#[must_use]
pub fn session_cells(s: &Session) -> SessionCells {
    SessionCells {
        datetime: format_local(s.started_at),
        project: s.project_short_name.clone(),
        title: s.title.clone(),
        tokens: format_tokens(s.total_billable),
        cost: format_cost_opt(s.cost_breakdown.map(|b| b.total())),
    }
}

#[must_use]
pub fn show_row_cells(row: &PreparedRow) -> ShowRowCells {
    let role = match row.role {
        PreparedRowRole::User => "user",
        PreparedRowRole::Assistant => "assistant",
        PreparedRowRole::Subagent => "subagent",
    };
    let content = if row.tool_use_count > 0 {
        format!("{} +{} tool uses", row.content, row.tool_use_count)
    } else {
        row.content.clone()
    };
    ShowRowCells {
        datetime: format_local_or_empty(row.timestamp),
        role: role.to_string(),
        tokens: row
            .tokens
            .map_or_else(|| "\u{2014}".to_string(), format_tokens),
        cost: row.cost,
        cumulative_tokens: format_tokens(row.cumulative_tokens),
        cumulative_cost: format_cost_opt(row.cumulative_cost),
        content,
    }
}

#[must_use]
pub fn inputs_cells(row: &AttributionRow) -> InputsCells {
    InputsCells {
        file_path: display_path(&row.file.path),
        kind: kind_label(&row.file.kind),
        tier: row.tier_label().to_string(),
        tokens: format_tokens(row.file.tokens),
        loads: row.total_loads().to_string(),
        billed: format_tokens(row.estimated_tokens_billed),
        cost: format_cost_opt(row.attributed_cost),
    }
}

#[must_use]
pub fn pricing_view_rows(model: &str, pricing: &ClaudePricing) -> Vec<Vec<String>> {
    if tiers_differ(pricing) {
        vec![
            vec![
                model.to_string(),
                "\u{2264}200k".to_string(),
                format_rate_mtok(pricing.input.first_200k_rate),
                format_rate_mtok(pricing.output.first_200k_rate),
                format_rate_mtok(pricing.cache_read.first_200k_rate),
                format_rate_mtok(pricing.cache_creation_5m.first_200k_rate),
                format_rate_mtok(pricing.cache_creation_1h.first_200k_rate),
            ],
            vec![
                String::new(),
                ">200k".to_string(),
                format_rate_mtok(pricing.input.above_200k_rate),
                format_rate_mtok(pricing.output.above_200k_rate),
                format_rate_mtok(pricing.cache_read.above_200k_rate),
                format_rate_mtok(pricing.cache_creation_5m.above_200k_rate),
                format_rate_mtok(pricing.cache_creation_1h.above_200k_rate),
            ],
        ]
    } else {
        vec![vec![
            model.to_string(),
            String::new(),
            format_rate_mtok(pricing.input.first_200k_rate),
            format_rate_mtok(pricing.output.first_200k_rate),
            format_rate_mtok(pricing.cache_read.first_200k_rate),
            format_rate_mtok(pricing.cache_creation_5m.first_200k_rate),
            format_rate_mtok(pricing.cache_creation_1h.first_200k_rate),
        ]]
    }
}

#[must_use]
pub fn session_totals(sessions: &[Session]) -> Option<SessionTotals> {
    if sessions.len() < 2 {
        return None;
    }
    let total_tokens: u64 = sessions.iter().map(|s| s.total_billable).sum();
    let total_cost: Option<f64> = sessions.iter().try_fold(0.0, |acc, s| {
        fold_cum_cost(Some(acc), s.cost_breakdown.map(|b| b.total()))
    });
    Some(SessionTotals {
        total_tokens,
        total_cost,
    })
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};

    use super::*;
    use crate::aggregation::PreparedRow;
    use crate::domain::CostBreakdown;

    // --- test helpers ---

    fn fixture_session(
        project: &str,
        title: &str,
        total_billable: u64,
        cost: Option<f64>,
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
            cost_breakdown: cost.map(|c| CostBreakdown {
                output: c,
                ..CostBreakdown::default()
            }),
        }
    }

    fn fixture_row(
        role: PreparedRowRole,
        tokens: Option<u64>,
        cost: Option<CostBreakdown>,
        cum_tokens: u64,
        cum_cost: Option<f64>,
        content: &str,
        tool_use_count: usize,
    ) -> PreparedRow {
        PreparedRow {
            timestamp: Some("2026-04-01T10:00:00Z".parse::<DateTime<Utc>>().unwrap()),
            role,
            tokens,
            cost,
            cumulative_tokens: cum_tokens,
            cumulative_cost: cum_cost,
            content: content.to_string(),
            tool_use_count,
        }
    }

    // --- session_cells ---

    #[test]
    fn session_cells_formats_all_fields() {
        let s = fixture_session(
            "proj",
            "my title",
            12345,
            Some(0.0042),
            "2026-04-01T10:30:00Z",
        );
        let cells = session_cells(&s);

        let expected_dt = chrono::DateTime::parse_from_rfc3339("2026-04-01T10:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M")
            .to_string();
        assert_eq!(cells.datetime, expected_dt);
        assert_eq!(cells.project, "proj");
        assert_eq!(cells.title, "my title");
        assert_eq!(cells.tokens, "12.35k");
        assert_eq!(cells.cost, "$0.0042");
    }

    #[test]
    fn session_cells_renders_em_dash_for_none_cost() {
        let s = fixture_session("proj", "title", 100, None, "2026-04-01T10:00:00Z");
        let cells = session_cells(&s);
        assert_eq!(cells.cost, "\u{2014}");
    }

    // --- show_row_cells ---

    #[test]
    fn show_row_cells_maps_user_role() {
        let row = fixture_row(PreparedRowRole::User, Some(100), None, 100, None, "q", 0);
        let cells = show_row_cells(&row);
        assert_eq!(cells.role, "user");
    }

    #[test]
    fn show_row_cells_maps_assistant_role() {
        let row = fixture_row(
            PreparedRowRole::Assistant,
            Some(50),
            None,
            150,
            None,
            "r",
            0,
        );
        let cells = show_row_cells(&row);
        assert_eq!(cells.role, "assistant");
    }

    #[test]
    fn show_row_cells_maps_subagent_role() {
        let row = fixture_row(PreparedRowRole::Subagent, Some(30), None, 180, None, "s", 0);
        let cells = show_row_cells(&row);
        assert_eq!(cells.role, "subagent");
    }

    #[test]
    fn show_row_cells_appends_tool_use_suffix() {
        let row = fixture_row(
            PreparedRowRole::Assistant,
            Some(50),
            None,
            50,
            None,
            "reading files",
            3,
        );
        let cells = show_row_cells(&row);
        assert_eq!(cells.content, "reading files +3 tool uses");
    }

    #[test]
    fn show_row_cells_omits_tool_use_suffix_when_zero() {
        let row = fixture_row(
            PreparedRowRole::Assistant,
            Some(50),
            None,
            50,
            None,
            "done",
            0,
        );
        let cells = show_row_cells(&row);
        assert_eq!(cells.content, "done");
    }

    #[test]
    fn show_row_cells_returns_raw_cost_breakdown() {
        let breakdown = CostBreakdown {
            input: 0.003,
            output: 0.005,
            cache_creation_5m: 0.001,
            cache_creation_1h: 0.0,
            cache_read: 0.0002,
        };
        let row = fixture_row(
            PreparedRowRole::User,
            Some(100),
            Some(breakdown),
            100,
            Some(0.0092),
            "q",
            0,
        );
        let cells = show_row_cells(&row);
        let cost = cells.cost.expect("should be Some");
        assert!((cost.input - 0.003).abs() < 1e-12);
        assert!((cost.output - 0.005).abs() < 1e-12);
    }

    #[test]
    fn show_row_cells_renders_em_dash_for_none_tokens() {
        let row = fixture_row(PreparedRowRole::User, None, None, 0, None, "orphan", 0);
        let cells = show_row_cells(&row);
        assert_eq!(cells.tokens, "\u{2014}");
    }

    // --- inputs_cells ---

    use std::path::PathBuf;

    use crate::attribution::AttributionRow;
    use crate::inventory::{ContextFile, ContextFileKind, Scope};

    #[test]
    fn inputs_cells_formats_all_fields() {
        let row = AttributionRow {
            file: ContextFile {
                path: PathBuf::from("/some/path/CLAUDE.md"),
                kind: ContextFileKind::GlobalClaudeMd,
                tokens: 500,
                scope: Scope::Global,
            },
            loads_1h: 3,
            loads_5m: 0,
            estimated_tokens_billed: 1500,
            attributed_cost: Some(0.0045),
        };
        let cells = inputs_cells(&row);
        assert_eq!(cells.file_path, "/some/path/CLAUDE.md");
        assert_eq!(cells.kind, "global");
        assert_eq!(cells.tier, "1h");
        assert_eq!(cells.tokens, "0.50k");
        assert_eq!(cells.loads, "3");
        assert_eq!(cells.billed, "1.50k");
        assert_eq!(cells.cost, "$0.0045");
    }

    #[test]
    fn inputs_cells_renders_em_dash_for_none_cost() {
        let row = AttributionRow {
            file: ContextFile {
                path: PathBuf::from("/x.md"),
                kind: ContextFileKind::UserRule,
                tokens: 10,
                scope: Scope::Global,
            },
            loads_1h: 0,
            loads_5m: 0,
            estimated_tokens_billed: 0,
            attributed_cost: None,
        };
        let cells = inputs_cells(&row);
        assert_eq!(cells.cost, "\u{2014}");
    }

    // --- pricing_view_rows ---

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
    fn pricing_view_rows_uniform_returns_one_row() {
        let p = uniform_pricing(3e-6);
        let rows = pricing_view_rows("claude-haiku-4-5", &p);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], "claude-haiku-4-5");
        assert_eq!(rows[0][1], "");
        assert_eq!(rows[0][2], "$3.00");
    }

    #[test]
    fn pricing_view_rows_split_returns_two_rows() {
        let p = split_pricing();
        let rows = pricing_view_rows("claude-sonnet-4-5", &p);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], "claude-sonnet-4-5");
        assert_eq!(rows[0][1], "\u{2264}200k");
        assert_eq!(rows[1][0], "");
        assert_eq!(rows[1][1], ">200k");
    }

    // --- session_totals ---

    #[test]
    fn session_totals_returns_none_for_zero_sessions() {
        assert!(session_totals(&[]).is_none());
    }

    #[test]
    fn session_totals_returns_none_for_one_session() {
        let sessions = vec![fixture_session(
            "p",
            "t",
            100,
            Some(0.01),
            "2026-04-01T10:00:00Z",
        )];
        assert!(session_totals(&sessions).is_none());
    }

    #[test]
    fn session_totals_sums_for_two_sessions() {
        let sessions = vec![
            fixture_session("a", "t1", 1000, Some(0.01), "2026-04-01T10:00:00Z"),
            fixture_session("b", "t2", 2000, Some(0.02), "2026-04-02T10:00:00Z"),
        ];
        let totals = session_totals(&sessions).expect("should be Some for 2 sessions");
        assert_eq!(totals.total_tokens, 3000);
        assert!((totals.total_cost.unwrap() - 0.03).abs() < 1e-10);
    }

    #[test]
    fn session_totals_propagates_none_cost() {
        let sessions = vec![
            fixture_session("a", "t1", 1000, Some(0.01), "2026-04-01T10:00:00Z"),
            fixture_session("b", "t2", 2000, None, "2026-04-02T10:00:00Z"),
        ];
        let totals = session_totals(&sessions).expect("should be Some for 2 sessions");
        assert_eq!(totals.total_tokens, 3000);
        assert!(totals.total_cost.is_none());
    }
}
