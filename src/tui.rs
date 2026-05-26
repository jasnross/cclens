//! Interactive TUI rendering for `cclens list` and `cclens inputs`
//! with tabbed navigation and session drill-down.
//!
//! Public API:
//! - `Tab` — `Sessions` | `Inputs` — the active tab.
//! - `run_tui<F, G>(Vec<Session>, F, G, Tab) -> anyhow::Result<()>`
//!   where `F: Fn(&str) -> Result<Vec<PreparedExchange>>` and
//!   `G: Fn() -> Result<(Vec<AttributionRow>, CoverageStats)>` —
//!   fullscreen TUI with tab switching (1/2 keys), scrollable tables,
//!   session drill-down (Enter/Esc within Sessions tab), and
//!   attribution table with coverage footer (Inputs tab).
//!
//! The plain-text path (`rendering::render_table` / `rendering::render_session`
//! / `rendering::render_inputs`) remains the fallback for piped output,
//! `--plain`, and standalone `cclens show <id>`.

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Style, Stylize};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Row, Table, TableState};
use ratatui::{DefaultTerminal, Frame};

use crate::aggregation::{PreparedExchange, PreparedRow, PreparedRowRole, fold_cum_cost};
use crate::attribution::{AttributionRow, CoverageStats};
use crate::domain::Session;
use crate::formatting::{
    coverage_line, display_path, format_cost_opt, format_local, format_local_or_empty,
    format_tokens, kind_label,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Sessions,
    Inputs,
}

struct InputsState {
    rows: Vec<AttributionRow>,
    coverage: CoverageStats,
    table_state: TableState,
}

impl InputsState {
    fn new(rows: Vec<AttributionRow>, coverage: CoverageStats) -> Self {
        let mut table_state = TableState::default();
        if !rows.is_empty() {
            table_state.select_first();
        }
        Self {
            rows,
            coverage,
            table_state,
        }
    }
}

enum View {
    List,
    Show {
        header_label: String,
        prepared: Vec<PreparedExchange>,
        table_state: TableState,
    },
}

struct App {
    tab: Tab,
    sessions: Vec<Session>,
    list_state: TableState,
    total_tokens: u64,
    total_cost: Option<f64>,
    view: View,
    inputs: Option<InputsState>,
}

impl App {
    fn new(sessions: Vec<Session>, default_tab: Tab) -> Self {
        let total_tokens = sessions.iter().map(|s| s.total_billable).sum();
        let total_cost = sessions.iter().try_fold(0.0, |acc, s| {
            fold_cum_cost(Some(acc), s.cost_breakdown.map(|b| b.total()))
        });
        let mut list_state = TableState::default();
        if !sessions.is_empty() {
            list_state.select_first();
        }
        Self {
            tab: default_tab,
            sessions,
            list_state,
            total_tokens,
            total_cost,
            view: View::List,
            inputs: None,
        }
    }
}

#[allow(clippy::missing_errors_doc)]
pub fn run_tui<F, G>(
    sessions: Vec<Session>,
    load_show: F,
    load_inputs: G,
    default_tab: Tab,
) -> anyhow::Result<()>
where
    F: Fn(&str) -> anyhow::Result<Vec<PreparedExchange>>,
    G: Fn() -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)>,
{
    let mut terminal = ratatui::try_init()?;
    let result = run_event_loop(
        &mut terminal,
        sessions,
        &load_show,
        &load_inputs,
        default_tab,
    );
    ratatui::restore();
    result
}

fn run_event_loop<F, G>(
    terminal: &mut DefaultTerminal,
    sessions: Vec<Session>,
    load_show: &F,
    load_inputs: &G,
    default_tab: Tab,
) -> anyhow::Result<()>
where
    F: Fn(&str) -> anyhow::Result<Vec<PreparedExchange>>,
    G: Fn() -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)>,
{
    let mut app = App::new(sessions, default_tab);

    if default_tab == Tab::Inputs
        && let Ok((rows, coverage)) = load_inputs()
    {
        app.inputs = Some(InputsState::new(rows, coverage));
    }

    loop {
        terminal.draw(|frame| render(&mut app, frame))?;
        if let Event::Key(key) = event::read()?
            && handle_key_event(&mut app, key, load_show, load_inputs)
        {
            break;
        }
    }
    Ok(())
}

fn handle_key_event<F, G>(app: &mut App, key: KeyEvent, load_show: &F, load_inputs: &G) -> bool
where
    F: Fn(&str) -> anyhow::Result<Vec<PreparedExchange>>,
    G: Fn() -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)>,
{
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return true;
    }
    match key.code {
        KeyCode::Char('1') => {
            app.tab = Tab::Sessions;
            return false;
        }
        KeyCode::Char('2') => {
            try_switch_to_inputs(app, load_inputs);
            return false;
        }
        KeyCode::Backspace
        | KeyCode::Enter
        | KeyCode::Left
        | KeyCode::Right
        | KeyCode::Up
        | KeyCode::Down
        | KeyCode::Home
        | KeyCode::End
        | KeyCode::PageUp
        | KeyCode::PageDown
        | KeyCode::Tab
        | KeyCode::BackTab
        | KeyCode::Delete
        | KeyCode::Insert
        | KeyCode::F(_)
        | KeyCode::Char(_)
        | KeyCode::Null
        | KeyCode::Esc
        | KeyCode::CapsLock
        | KeyCode::ScrollLock
        | KeyCode::NumLock
        | KeyCode::PrintScreen
        | KeyCode::Pause
        | KeyCode::Menu
        | KeyCode::KeypadBegin
        | KeyCode::Media(_)
        | KeyCode::Modifier(_) => {}
    }
    match app.tab {
        Tab::Sessions => {
            if matches!(app.view, View::Show { .. }) {
                handle_show_key(app, key)
            } else {
                handle_list_key(app, key, load_show)
            }
        }
        Tab::Inputs => handle_inputs_key(app, key),
    }
}

fn handle_list_key<F>(app: &mut App, key: KeyEvent, load_show: &F) -> bool
where
    F: Fn(&str) -> anyhow::Result<Vec<PreparedExchange>>,
{
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Enter => {
            try_open_show(app, load_show);
            false
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.list_state.select_next();
            false
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.list_state.select_previous();
            false
        }
        KeyCode::Home | KeyCode::Char('g') => {
            app.list_state.select_first();
            false
        }
        KeyCode::End | KeyCode::Char('G') => {
            app.list_state.select_last();
            false
        }
        KeyCode::Backspace
        | KeyCode::Left
        | KeyCode::Right
        | KeyCode::PageUp
        | KeyCode::PageDown
        | KeyCode::Tab
        | KeyCode::BackTab
        | KeyCode::Delete
        | KeyCode::Insert
        | KeyCode::F(_)
        | KeyCode::Char(_)
        | KeyCode::Null
        | KeyCode::CapsLock
        | KeyCode::ScrollLock
        | KeyCode::NumLock
        | KeyCode::PrintScreen
        | KeyCode::Pause
        | KeyCode::Menu
        | KeyCode::KeypadBegin
        | KeyCode::Media(_)
        | KeyCode::Modifier(_) => false,
    }
}

fn try_open_show<F>(app: &mut App, load_show: &F)
where
    F: Fn(&str) -> anyhow::Result<Vec<PreparedExchange>>,
{
    let Some(idx) = app.list_state.selected() else {
        return;
    };
    let session_id = app.sessions[idx].id.clone();
    let header_label = format!(
        "\"{}\" ({})",
        app.sessions[idx].title, app.sessions[idx].project_short_name,
    );
    let Ok(prepared) = load_show(&session_id) else {
        return;
    };
    let mut table_state = TableState::default();
    if !prepared.is_empty() {
        table_state.select_first();
    }
    app.view = View::Show {
        header_label,
        prepared,
        table_state,
    };
}

fn handle_show_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') => true,
        KeyCode::Esc | KeyCode::Backspace => {
            app.view = View::List;
            false
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if let View::Show { table_state, .. } = &mut app.view {
                table_state.select_next();
            }
            false
        }
        KeyCode::Up | KeyCode::Char('k') => {
            if let View::Show { table_state, .. } = &mut app.view {
                table_state.select_previous();
            }
            false
        }
        KeyCode::Home | KeyCode::Char('g') => {
            if let View::Show { table_state, .. } = &mut app.view {
                table_state.select_first();
            }
            false
        }
        KeyCode::End | KeyCode::Char('G') => {
            if let View::Show { table_state, .. } = &mut app.view {
                table_state.select_last();
            }
            false
        }
        KeyCode::Enter
        | KeyCode::Left
        | KeyCode::Right
        | KeyCode::PageUp
        | KeyCode::PageDown
        | KeyCode::Tab
        | KeyCode::BackTab
        | KeyCode::Delete
        | KeyCode::Insert
        | KeyCode::F(_)
        | KeyCode::Char(_)
        | KeyCode::Null
        | KeyCode::CapsLock
        | KeyCode::ScrollLock
        | KeyCode::NumLock
        | KeyCode::PrintScreen
        | KeyCode::Pause
        | KeyCode::Menu
        | KeyCode::KeypadBegin
        | KeyCode::Media(_)
        | KeyCode::Modifier(_) => false,
    }
}

fn try_switch_to_inputs<G>(app: &mut App, load_inputs: &G)
where
    G: Fn() -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)>,
{
    if app.inputs.is_none()
        && let Ok((rows, coverage)) = load_inputs()
    {
        app.inputs = Some(InputsState::new(rows, coverage));
    }
    if app.inputs.is_some() {
        app.tab = Tab::Inputs;
    }
}

fn handle_inputs_key(app: &mut App, key: KeyEvent) -> bool {
    let Some(inputs) = &mut app.inputs else {
        return false;
    };
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Down | KeyCode::Char('j') => {
            inputs.table_state.select_next();
            false
        }
        KeyCode::Up | KeyCode::Char('k') => {
            inputs.table_state.select_previous();
            false
        }
        KeyCode::Home | KeyCode::Char('g') => {
            inputs.table_state.select_first();
            false
        }
        KeyCode::End | KeyCode::Char('G') => {
            inputs.table_state.select_last();
            false
        }
        KeyCode::Enter
        | KeyCode::Backspace
        | KeyCode::Left
        | KeyCode::Right
        | KeyCode::PageUp
        | KeyCode::PageDown
        | KeyCode::Tab
        | KeyCode::BackTab
        | KeyCode::Delete
        | KeyCode::Insert
        | KeyCode::F(_)
        | KeyCode::Char(_)
        | KeyCode::Null
        | KeyCode::CapsLock
        | KeyCode::ScrollLock
        | KeyCode::NumLock
        | KeyCode::PrintScreen
        | KeyCode::Pause
        | KeyCode::Menu
        | KeyCode::KeypadBegin
        | KeyCode::Media(_)
        | KeyCode::Modifier(_) => false,
    }
}

// ---- rendering ----

fn render(app: &mut App, frame: &mut Frame) {
    let [header_area, content_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(frame.area());

    render_tab_header(app, frame, header_area);

    match app.tab {
        Tab::Sessions => {
            if matches!(app.view, View::Show { .. }) {
                render_show_content(app, frame, content_area);
            } else {
                render_list_content(app, frame, content_area);
            }
        }
        Tab::Inputs => render_inputs_content(app, frame, content_area),
    }
}

fn render_tab_header(app: &App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let sessions_label = if app.tab == Tab::Sessions {
        "[Sessions]"
    } else {
        " Sessions "
    };
    let inputs_label = if app.tab == Tab::Inputs {
        "[Inputs]"
    } else {
        " Inputs "
    };

    let context = match app.tab {
        Tab::Sessions => {
            if let View::Show { header_label, .. } = &app.view {
                header_label.clone()
            } else {
                let count = app.sessions.len();
                let label = if count == 1 { "session" } else { "sessions" };
                format!("{count} {label}")
            }
        }
        Tab::Inputs => {
            let count = app.inputs.as_ref().map_or(0, |i| i.rows.len());
            let label = if count == 1 { "file" } else { "files" };
            format!("{count} {label}")
        }
    };

    let [left_area, right_area] =
        Layout::horizontal([Constraint::Fill(1), Constraint::Fill(1)]).areas(area);

    let left = Line::from(format!(" {sessions_label}  {inputs_label}")).bold();
    let right = Line::from(format!("cclens — {context} "))
        .bold()
        .right_aligned();
    frame.render_widget(Paragraph::new(left), left_area);
    frame.render_widget(Paragraph::new(right), right_area);
}

fn render_list_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let [table_area, totals_area, footer_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(area);

    render_sessions_table(app, frame, table_area);
    render_totals(app, frame, totals_area);
    render_list_footer(frame, footer_area);
}

fn render_sessions_table(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let rows: Vec<Row> = app
        .sessions
        .iter()
        .map(|s| {
            Row::new(vec![
                format_local(s.started_at),
                s.project_short_name.clone(),
                s.title.clone(),
                format_tokens(s.total_billable),
                format_cost_opt(s.cost_breakdown.map(|b| b.total())),
            ])
        })
        .collect();

    let header =
        Row::new(["datetime", "project", "title", "tokens", "cost"]).style(Style::new().bold());

    let table = Table::new(rows, session_table_widths())
        .header(header)
        .row_highlight_style(Style::new().reversed());

    frame.render_stateful_widget(table, area, &mut app.list_state);
}

fn session_table_widths() -> [Constraint; 5] {
    [
        Constraint::Length(16), // datetime
        Constraint::Length(16), // project
        Constraint::Fill(1),    // title (responsive)
        Constraint::Length(8),  // tokens
        Constraint::Length(9),  // cost
    ]
}

fn render_totals(app: &App, frame: &mut Frame, area: ratatui::layout::Rect) {
    if app.sessions.len() < 2 {
        return;
    }
    let row = Row::new(vec![
        Line::raw(""),
        Line::raw(""),
        Line::raw("total").right_aligned(),
        Line::raw(format_tokens(app.total_tokens)).right_aligned(),
        Line::raw(format_cost_opt(app.total_cost)).right_aligned(),
    ]);

    let table = Table::new(vec![row], session_table_widths());
    frame.render_widget(table, area);
}

fn render_list_footer(frame: &mut Frame, area: ratatui::layout::Rect) {
    let text = Line::from(" 1/2 tabs  ↑↓ navigate  Enter open  q quit").dim();
    frame.render_widget(Paragraph::new(text), area);
}

// ---- show view ----

fn render_show_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let View::Show {
        prepared,
        table_state,
        ..
    } = &mut app.view
    else {
        return;
    };

    let [table_area, footer_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(area);

    render_show_table(prepared, table_state, frame, table_area);

    let footer = Line::from(" 1/2 tabs  ↑↓ navigate  Esc back  q quit").dim();
    frame.render_widget(Paragraph::new(footer), footer_area);
}

fn show_table_widths() -> [Constraint; 11] {
    [
        Constraint::Length(16), // datetime
        Constraint::Length(10), // role
        Constraint::Length(8),  // tokens
        Constraint::Length(8),  // in
        Constraint::Length(8),  // out
        Constraint::Length(8),  // c5m
        Constraint::Length(8),  // c1h
        Constraint::Length(8),  // cr
        Constraint::Length(8),  // cum
        Constraint::Length(9),  // cum_cost
        Constraint::Fill(1),    // content
    ]
}

fn render_show_table(
    prepared: &[PreparedExchange],
    table_state: &mut TableState,
    frame: &mut Frame,
    area: ratatui::layout::Rect,
) {
    let rows: Vec<Row> = prepared
        .iter()
        .enumerate()
        .flat_map(|(ex_idx, ex)| {
            let dim = ex_idx % 2 == 1;
            ex.rows.iter().map(move |row| show_row(row, dim))
        })
        .collect();

    let header = Row::new([
        "datetime", "role", "tokens", "in", "out", "c5m", "c1h", "cr", "cum", "cum_cost", "content",
    ])
    .style(Style::new().bold());

    let table = Table::new(rows, show_table_widths())
        .header(header)
        .row_highlight_style(Style::new().reversed());

    frame.render_stateful_widget(table, area, table_state);
}

fn show_row(row: &PreparedRow, dim: bool) -> Row<'static> {
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
    let r = Row::new(vec![
        Line::raw(format_local_or_empty(row.timestamp)),
        Line::raw(role_str.to_string()),
        Line::raw(row.tokens.map_or_else(|| "—".to_string(), format_tokens)).right_aligned(),
        Line::raw(format_cost_opt(row.cost.map(|b| b.input))).right_aligned(),
        Line::raw(format_cost_opt(row.cost.map(|b| b.output))).right_aligned(),
        Line::raw(format_cost_opt(row.cost.map(|b| b.cache_creation_5m))).right_aligned(),
        Line::raw(format_cost_opt(row.cost.map(|b| b.cache_creation_1h))).right_aligned(),
        Line::raw(format_cost_opt(row.cost.map(|b| b.cache_read))).right_aligned(),
        Line::raw(format_tokens(row.cumulative_tokens)).right_aligned(),
        Line::raw(format_cost_opt(row.cumulative_cost)).right_aligned(),
        Line::raw(content),
    ]);
    if dim { r.dim() } else { r }
}

// ---- inputs view ----

fn render_inputs_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let Some(inputs) = &mut app.inputs else {
        return;
    };

    let [table_area, coverage_area, footer_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(area);

    render_inputs_table(inputs, frame, table_area);
    render_inputs_coverage(inputs, frame, coverage_area);
    render_inputs_footer(frame, footer_area);
}

fn inputs_table_widths() -> [Constraint; 7] {
    [
        Constraint::Fill(2),   // file (responsive, 2/3 of flexible space)
        Constraint::Fill(1),   // kind (responsive, 1/3 — handles long plugin names)
        Constraint::Length(6), // tier
        Constraint::Length(8), // tokens
        Constraint::Length(6), // loads
        Constraint::Length(8), // billed
        Constraint::Length(9), // cost
    ]
}

fn render_inputs_table(inputs: &mut InputsState, frame: &mut Frame, area: ratatui::layout::Rect) {
    let rows: Vec<Row> = inputs
        .rows
        .iter()
        .map(|row| {
            Row::new(vec![
                Line::raw(display_path(&row.file.path)),
                Line::raw(kind_label(&row.file.kind)),
                Line::raw(row.tier_label().to_string()),
                Line::raw(format_tokens(row.file.tokens)).right_aligned(),
                Line::raw(row.total_loads().to_string()).right_aligned(),
                Line::raw(format_tokens(row.estimated_tokens_billed)).right_aligned(),
                Line::raw(format_cost_opt(row.attributed_cost)).right_aligned(),
            ])
        })
        .collect();

    let header = Row::new(["file", "kind", "tier", "tokens", "loads", "billed", "cost"])
        .style(Style::new().bold());

    let table = Table::new(rows, inputs_table_widths())
        .header(header)
        .row_highlight_style(Style::new().reversed());

    frame.render_stateful_widget(table, area, &mut inputs.table_state);
}

fn render_inputs_coverage(inputs: &InputsState, frame: &mut Frame, area: ratatui::layout::Rect) {
    let text = Line::from(format!(" {}", coverage_line(&inputs.coverage)));
    frame.render_widget(Paragraph::new(text), area);
}

fn render_inputs_footer(frame: &mut Frame, area: ratatui::layout::Rect) {
    let text = Line::from(" 1/2 tabs  ↑↓ navigate  q quit").dim();
    frame.render_widget(Paragraph::new(text), area);
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use chrono::{DateTime, Utc};
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::aggregation::PreparedExchange;
    use crate::attribution::TierCoverage;
    use crate::domain::{CostBreakdown, TurnOrigin};
    use crate::inventory::{ContextFile, ContextFileKind, Scope};

    fn fixture_sessions() -> Vec<Session> {
        vec![
            Session {
                id: "aaa".to_string(),
                project_short_name: "alpha".to_string(),
                started_at: "2026-04-01T10:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                last_activity: "2026-04-01T11:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                title: "First session".to_string(),
                turns: Vec::new(),
                total_billable: 1500,
                cost_breakdown: Some(CostBreakdown {
                    output: 0.01,
                    ..CostBreakdown::default()
                }),
            },
            Session {
                id: "bbb".to_string(),
                project_short_name: "beta".to_string(),
                started_at: "2026-04-05T10:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                last_activity: "2026-04-05T11:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                title: "Second session".to_string(),
                turns: Vec::new(),
                total_billable: 2500,
                cost_breakdown: Some(CostBreakdown {
                    output: 0.02,
                    ..CostBreakdown::default()
                }),
            },
            Session {
                id: "ccc".to_string(),
                project_short_name: "gamma".to_string(),
                started_at: "2026-04-10T10:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                last_activity: "2026-04-10T11:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                title: "Third session".to_string(),
                turns: Vec::new(),
                total_billable: 3000,
                cost_breakdown: Some(CostBreakdown {
                    output: 0.03,
                    ..CostBreakdown::default()
                }),
            },
        ]
    }

    fn fixture_prepared_exchanges() -> Vec<PreparedExchange> {
        vec![
            PreparedExchange {
                origin: TurnOrigin::Parent,
                rows: vec![
                    PreparedRow {
                        timestamp: Some("2026-04-01T10:00:00Z".parse::<DateTime<Utc>>().unwrap()),
                        role: PreparedRowRole::User,
                        tokens: Some(500),
                        cost: Some(CostBreakdown {
                            input: 0.0066,
                            output: 0.0,
                            cache_creation_5m: 0.0019,
                            cache_creation_1h: 0.0,
                            cache_read: 0.0001,
                        }),
                        cumulative_tokens: 500,
                        cumulative_cost: Some(0.0086),
                        content: "What is this?".to_string(),
                        tool_use_count: 0,
                    },
                    PreparedRow {
                        timestamp: Some("2026-04-01T10:01:00Z".parse::<DateTime<Utc>>().unwrap()),
                        role: PreparedRowRole::Assistant,
                        tokens: Some(200),
                        cost: Some(CostBreakdown {
                            input: 0.0,
                            output: 0.0042,
                            cache_creation_5m: 0.0,
                            cache_creation_1h: 0.0,
                            cache_read: 0.0,
                        }),
                        cumulative_tokens: 700,
                        cumulative_cost: Some(0.0128),
                        content: "This is a CLI tool".to_string(),
                        tool_use_count: 3,
                    },
                ],
            },
            PreparedExchange {
                origin: TurnOrigin::Subagent {
                    agent_type: "tw-code-reviewer".to_string(),
                    description: Some("Review changes".to_string()),
                },
                rows: vec![PreparedRow {
                    timestamp: Some("2026-04-01T10:02:00Z".parse::<DateTime<Utc>>().unwrap()),
                    role: PreparedRowRole::Subagent,
                    tokens: Some(300),
                    cost: None,
                    cumulative_tokens: 1000,
                    cumulative_cost: None,
                    content: "(tw-code-reviewer · \"Review changes\") all good".to_string(),
                    tool_use_count: 0,
                }],
            },
        ]
    }

    fn noop_loader() -> impl Fn(&str) -> anyhow::Result<Vec<PreparedExchange>> {
        |_| Ok(vec![])
    }

    fn fixture_loader() -> impl Fn(&str) -> anyhow::Result<Vec<PreparedExchange>> {
        |_| Ok(fixture_prepared_exchanges())
    }

    fn noop_inputs_loader() -> impl Fn() -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)> {
        || anyhow::bail!("no loader")
    }

    fn key_event(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    // --- App construction tests ---

    #[test]
    fn app_new_selects_first_when_non_empty() {
        let app = App::new(fixture_sessions(), Tab::Sessions);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn app_new_no_selection_when_empty() {
        let app = App::new(vec![], Tab::Sessions);
        assert_eq!(app.list_state.selected(), None);
    }

    #[test]
    fn app_new_computes_totals() {
        let app = App::new(fixture_sessions(), Tab::Sessions);
        assert_eq!(app.total_tokens, 1500 + 2500 + 3000);
        let expected_cost = 0.01 + 0.02 + 0.03;
        assert!(
            (app.total_cost.unwrap() - expected_cost).abs() < 1e-10,
            "expected {expected_cost}, got {:?}",
            app.total_cost,
        );
    }

    // --- View state tests ---

    #[test]
    fn app_starts_in_list_view() {
        let app = App::new(fixture_sessions(), Tab::Sessions);
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn enter_transitions_to_show_view() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::Show { .. }));
    }

    #[test]
    fn enter_with_failed_closure_stays_on_list() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader =
            |_: &str| -> anyhow::Result<Vec<PreparedExchange>> { anyhow::bail!("load failed") };
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn enter_with_no_selection_is_noop() {
        let mut app = App::new(vec![], Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn esc_in_show_returns_to_list() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::Show { .. }));
        handle_key_event(&mut app, key_event(KeyCode::Esc), &loader, &il);
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn backspace_in_show_returns_to_list() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::Show { .. }));
        handle_key_event(&mut app, key_event(KeyCode::Backspace), &loader, &il);
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn q_in_show_quits() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::Show { .. }));
        assert!(handle_key_event(
            &mut app,
            key_event(KeyCode::Char('q')),
            &loader,
            &il
        ));
    }

    #[test]
    fn list_selection_preserved_after_roundtrip() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        app.list_state.select(Some(2));
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::Show { .. }));
        handle_key_event(&mut app, key_event(KeyCode::Esc), &loader, &il);
        assert_eq!(app.list_state.selected(), Some(2));
    }

    // --- List key handling tests ---

    #[test]
    fn handle_key_quit_on_q() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert!(handle_key_event(
            &mut app,
            key_event(KeyCode::Char('q')),
            &loader,
            &il
        ));
    }

    #[test]
    fn handle_key_quit_on_esc() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert!(handle_key_event(
            &mut app,
            key_event(KeyCode::Esc),
            &loader,
            &il
        ));
    }

    #[test]
    fn handle_key_quit_on_ctrl_c() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(handle_key_event(&mut app, key, &loader, &il));
    }

    #[test]
    fn handle_key_down_advances_selection() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert_eq!(app.list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Down), &loader, &il);
        assert_eq!(app.list_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_up_retreats_selection() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        app.list_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Up), &loader, &il);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_j_advances_like_down() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert_eq!(app.list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Char('j')), &loader, &il);
        assert_eq!(app.list_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_k_retreats_like_up() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        app.list_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Char('k')), &loader, &il);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_home_selects_first() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        app.list_state.select(Some(2));
        handle_key_event(&mut app, key_event(KeyCode::Home), &loader, &il);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_end_selects_last() {
        let sessions = fixture_sessions();
        let last = sessions.len() - 1;
        let mut app = App::new(sessions, Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert_eq!(app.list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::End), &loader, &il);
        render_app(&mut app, 80, 10);
        assert_eq!(app.list_state.selected(), Some(last));
    }

    #[test]
    fn handle_key_unknown_is_noop() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert_eq!(app.list_state.selected(), Some(0));
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('x')), &loader, &il);
        assert!(!quit);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    // --- Show view key handling tests ---

    #[test]
    fn show_down_advances_selection() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(0));
        }
        handle_key_event(&mut app, key_event(KeyCode::Down), &loader, &il);
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(1));
        }
    }

    #[test]
    fn show_up_retreats_selection() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::Down), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::Up), &loader, &il);
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(0));
        }
    }

    #[test]
    fn show_home_selects_first() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::Down), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::Down), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::Home), &loader, &il);
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(0));
        }
    }

    #[test]
    fn show_end_selects_last() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::End), &loader, &il);
        render_app(&mut app, 120, 20);
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(2));
        }
    }

    // --- Rendering tests (TestBackend) ---

    fn render_app(app: &mut App, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(app, frame)).unwrap();
        format!("{}", terminal.backend())
    }

    #[test]
    fn list_tui_renders_header_with_session_count() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("cclens — 3 sessions"),
            "header missing session count; got: {first_line}",
        );
    }

    #[test]
    fn list_tui_renders_table_header_columns() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let second_line = output.lines().nth(1).unwrap();
        for col in ["datetime", "project", "title", "tokens", "cost"] {
            assert!(
                second_line.contains(col),
                "header column `{col}` missing; got: {second_line}",
            );
        }
    }

    #[test]
    fn list_tui_renders_footer_key_hints() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("navigate") && last_line.contains("q quit"),
            "footer key hints missing; got: {last_line}",
        );
    }

    #[test]
    fn list_tui_renders_totals_when_multiple_sessions() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let lines: Vec<&str> = output.lines().collect();
        let totals_line = lines[lines.len() - 2];
        assert!(
            totals_line.contains("total"),
            "totals row missing; got: {totals_line}",
        );
        assert!(
            totals_line.contains("7.00k"),
            "totals row missing token sum 7.00k; got: {totals_line}",
        );
    }

    #[test]
    fn list_tui_omits_totals_for_single_session() {
        let mut app = App::new(vec![fixture_sessions().remove(0)], Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let lines: Vec<&str> = output.lines().collect();
        let totals_line = lines[lines.len() - 2];
        assert!(
            !totals_line.contains("total"),
            "single session should not show totals; got: {totals_line}",
        );
    }

    #[test]
    fn list_tui_renders_session_data_in_rows() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let data_line = output.lines().nth(2).unwrap();
        assert!(
            data_line.contains("alpha"),
            "first data row should contain project name 'alpha'; got: {data_line}",
        );
        assert!(
            data_line.contains("1.50k"),
            "first data row should contain formatted token count; got: {data_line}",
        );
    }

    #[test]
    fn list_footer_shows_enter_hint() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("Enter open"),
            "footer should contain 'Enter open'; got: {last_line}",
        );
    }

    // --- Show view rendering tests ---

    fn enter_show_view(app: &mut App) {
        let loader = fixture_loader();
        let il = noop_inputs_loader();
        handle_key_event(app, key_event(KeyCode::Enter), &loader, &il);
    }

    #[test]
    fn show_tui_renders_header_with_session_info() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let output = render_app(&mut app, 120, 20);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("cclens") && first_line.contains("First session"),
            "header should contain session title; got: {first_line}",
        );
        assert!(
            first_line.contains("alpha"),
            "header should contain project name; got: {first_line}",
        );
    }

    #[test]
    fn show_tui_renders_column_headers() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let output = render_app(&mut app, 140, 20);
        let second_line = output.lines().nth(1).unwrap();
        for col in [
            "datetime", "role", "tokens", "in", "out", "c5m", "c1h", "cr", "cum", "cum_cost",
            "content",
        ] {
            assert!(
                second_line.contains(col),
                "show header column `{col}` missing; got: {second_line}",
            );
        }
    }

    #[test]
    fn show_tui_renders_footer_with_back_hint() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let output = render_app(&mut app, 120, 20);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("Esc back") && last_line.contains("q quit"),
            "footer should contain back and quit hints; got: {last_line}",
        );
    }

    #[test]
    fn show_tui_renders_exchange_rows() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let output = render_app(&mut app, 140, 20);
        assert!(
            output.contains("user"),
            "output should contain 'user' role; got:\n{output}",
        );
        assert!(
            output.contains("assistant"),
            "output should contain 'assistant' role; got:\n{output}",
        );
        assert!(
            output.contains("subagent"),
            "output should contain 'subagent' role; got:\n{output}",
        );
    }

    #[test]
    fn show_tui_renders_per_component_costs() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let output = render_app(&mut app, 140, 20);
        assert!(
            output.contains("$0.0066"),
            "should contain input cost $0.0066; got:\n{output}",
        );
        assert!(
            output.contains("$0.0042"),
            "should contain output cost $0.0042; got:\n{output}",
        );
    }

    #[test]
    fn show_tui_renders_dash_for_none_cost() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let output = render_app(&mut app, 140, 20);
        let subagent_line = output.lines().find(|l| l.contains("subagent")).unwrap();
        assert!(
            subagent_line.contains('—'),
            "subagent row (None cost) should contain em-dash; got: {subagent_line}",
        );
    }

    #[test]
    fn show_tui_renders_tool_use_count_in_content() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let output = render_app(&mut app, 160, 20);
        assert!(
            output.contains("+3 tool uses"),
            "should contain tool use count; got:\n{output}",
        );
    }

    #[test]
    fn show_tui_dims_odd_exchange_rows() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        enter_show_view(&mut app);
        let backend = TestBackend::new(140, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(&mut app, frame)).unwrap();
        let buf = terminal.backend().buffer();

        let user_row_y = (2..buf.area.height).find(|&y| {
            let line: String = (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol().chars().next().unwrap_or(' '))
                .collect();
            line.contains("user") && line.contains("What is this")
        });
        let subagent_row_y = (2..buf.area.height).find(|&y| {
            let line: String = (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol().chars().next().unwrap_or(' '))
                .collect();
            line.contains("subagent")
        });

        if let Some(user_y) = user_row_y {
            let user_cell = &buf[(20, user_y)];
            assert!(
                !user_cell.modifier.contains(ratatui::style::Modifier::DIM),
                "first exchange (idx 0) should not be dimmed",
            );
        }
        if let Some(sub_y) = subagent_row_y {
            let sub_cell = &buf[(20, sub_y)];
            assert!(
                sub_cell.modifier.contains(ratatui::style::Modifier::DIM),
                "second exchange (idx 1) should be dimmed",
            );
        }
    }

    // --- Inputs fixtures ---

    fn fixture_attribution_rows() -> Vec<AttributionRow> {
        vec![
            AttributionRow {
                file: ContextFile {
                    path: PathBuf::from("/home/user/.claude/CLAUDE.md"),
                    kind: ContextFileKind::GlobalClaudeMd,
                    tokens: 500,
                    scope: Scope::Global,
                },
                loads_1h: 5,
                loads_5m: 0,
                estimated_tokens_billed: 2500,
                attributed_cost: Some(0.003),
            },
            AttributionRow {
                file: ContextFile {
                    path: PathBuf::from("/home/user/.claude/skills/foo/SKILL.md"),
                    kind: ContextFileKind::UserSkill,
                    tokens: 1200,
                    scope: Scope::Global,
                },
                loads_1h: 0,
                loads_5m: 3,
                estimated_tokens_billed: 3600,
                attributed_cost: Some(0.005),
            },
            AttributionRow {
                file: ContextFile {
                    path: PathBuf::from("/home/user/project/CLAUDE.md"),
                    kind: ContextFileKind::ProjectClaudeMd,
                    tokens: 800,
                    scope: Scope::Global,
                },
                loads_1h: 2,
                loads_5m: 1,
                estimated_tokens_billed: 2400,
                attributed_cost: None,
            },
        ]
    }

    fn fixture_coverage_stats() -> CoverageStats {
        CoverageStats {
            long_1h: TierCoverage {
                observed_tokens: 5000,
                attributed_tokens: 4000,
                ratio: Some(0.8),
            },
            short_5m: TierCoverage {
                observed_tokens: 2000,
                attributed_tokens: 1500,
                ratio: Some(0.75),
            },
        }
    }

    fn fixture_inputs_loader() -> impl Fn() -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)>
    {
        || Ok((fixture_attribution_rows(), fixture_coverage_stats()))
    }

    // --- Tab switching tests ---

    #[test]
    fn app_starts_on_specified_tab() {
        let app = App::new(fixture_sessions(), Tab::Inputs);
        assert_eq!(app.tab, Tab::Inputs);
    }

    #[test]
    fn key_1_switches_to_sessions() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        let loader = noop_loader();
        let il = fixture_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Char('1')), &loader, &il);
        assert_eq!(app.tab, Tab::Sessions);
    }

    #[test]
    fn key_2_switches_to_inputs_loading_data() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = fixture_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Char('2')), &loader, &il);
        assert_eq!(app.tab, Tab::Inputs);
        assert!(app.inputs.is_some());
    }

    #[test]
    fn key_2_with_failed_loader_stays_on_sessions() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Char('2')), &loader, &il);
        assert_eq!(app.tab, Tab::Sessions);
    }

    #[test]
    fn key_2_reuses_cached_inputs() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = noop_loader();
        let il = fixture_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Char('2')), &loader, &il);
        assert!(app.inputs.is_some());
        handle_key_event(&mut app, key_event(KeyCode::Char('1')), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::Char('2')), &loader, &il);
        assert!(app.inputs.is_some());
    }

    #[test]
    fn tab_switch_from_show_view() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let loader = fixture_loader();
        let il = fixture_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Enter), &loader, &il);
        assert!(matches!(app.view, View::Show { .. }));
        handle_key_event(&mut app, key_event(KeyCode::Char('2')), &loader, &il);
        assert_eq!(app.tab, Tab::Inputs);
        handle_key_event(&mut app, key_event(KeyCode::Char('1')), &loader, &il);
        assert_eq!(app.tab, Tab::Sessions);
        assert!(matches!(app.view, View::Show { .. }));
    }

    // --- InputsState construction tests ---

    #[test]
    fn inputs_state_selects_first_when_non_empty() {
        let state = InputsState::new(fixture_attribution_rows(), fixture_coverage_stats());
        assert_eq!(state.table_state.selected(), Some(0));
    }

    #[test]
    fn inputs_state_no_selection_when_empty() {
        let state = InputsState::new(vec![], fixture_coverage_stats());
        assert_eq!(state.table_state.selected(), None);
    }

    // --- Inputs key handling tests ---

    #[test]
    fn inputs_key_quit_on_q() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert!(handle_key_event(
            &mut app,
            key_event(KeyCode::Char('q')),
            &loader,
            &il
        ));
    }

    #[test]
    fn inputs_key_quit_on_esc() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert!(handle_key_event(
            &mut app,
            key_event(KeyCode::Esc),
            &loader,
            &il
        ));
    }

    #[test]
    fn inputs_key_down_advances_selection() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        let loader = noop_loader();
        let il = noop_inputs_loader();
        assert_eq!(app.inputs.as_ref().unwrap().table_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Down), &loader, &il);
        assert_eq!(app.inputs.as_ref().unwrap().table_state.selected(), Some(1));
    }

    #[test]
    fn inputs_key_up_retreats_selection() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        let loader = noop_loader();
        let il = noop_inputs_loader();
        handle_key_event(&mut app, key_event(KeyCode::Down), &loader, &il);
        handle_key_event(&mut app, key_event(KeyCode::Up), &loader, &il);
        assert_eq!(app.inputs.as_ref().unwrap().table_state.selected(), Some(0));
    }

    #[test]
    fn inputs_key_unknown_is_noop() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        let loader = noop_loader();
        let il = noop_inputs_loader();
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('x')), &loader, &il);
        assert!(!quit);
        assert_eq!(app.inputs.as_ref().unwrap().table_state.selected(), Some(0));
    }

    // --- Tab rendering tests ---

    #[test]
    fn tab_header_shows_sessions_active() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("[Sessions]"),
            "header should show active Sessions tab; got: {first_line}",
        );
        assert!(
            first_line.contains("Inputs") && !first_line.contains("[Inputs]"),
            "Inputs should appear without brackets; got: {first_line}",
        );
    }

    #[test]
    fn tab_header_shows_inputs_active() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        app.tab = Tab::Inputs;
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("[Inputs]"),
            "header should show active Inputs tab; got: {first_line}",
        );
    }

    #[test]
    fn tab_header_shows_context_count() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("3 sessions"),
            "header should show session count; got: {first_line}",
        );
    }

    #[test]
    fn inputs_tab_renders_table_columns() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        app.tab = Tab::Inputs;
        let output = render_app(&mut app, 120, 10);
        for col in ["file", "kind", "tier", "tokens", "loads", "billed", "cost"] {
            assert!(
                output.contains(col),
                "inputs column `{col}` missing; got:\n{output}",
            );
        }
    }

    #[test]
    fn inputs_tab_renders_coverage_line() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        app.tab = Tab::Inputs;
        let output = render_app(&mut app, 120, 10);
        assert!(
            output.contains("coverage:"),
            "should contain coverage line; got:\n{output}",
        );
        assert!(
            output.contains("1h:") && output.contains("5m:"),
            "should contain both tier labels; got:\n{output}",
        );
    }

    #[test]
    fn inputs_tab_renders_footer_with_tab_hint() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        app.tab = Tab::Inputs;
        let output = render_app(&mut app, 120, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("1/2 tabs") && last_line.contains("q quit"),
            "footer should contain tab hint and quit; got: {last_line}",
        );
    }

    #[test]
    fn inputs_tab_renders_kind_labels() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        app.tab = Tab::Inputs;
        let output = render_app(&mut app, 120, 10);
        assert!(
            output.contains("global"),
            "should contain 'global' kind label; got:\n{output}",
        );
        assert!(
            output.contains("skill"),
            "should contain 'skill' kind label; got:\n{output}",
        );
    }

    #[test]
    fn inputs_tab_renders_dash_for_unknown_cost() {
        let mut app = App::new(fixture_sessions(), Tab::Inputs);
        app.inputs = Some(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        ));
        app.tab = Tab::Inputs;
        let output = render_app(&mut app, 120, 10);
        let project_line = output.lines().find(|l| l.contains("project"));
        assert!(
            project_line.is_some_and(|l| l.contains('—')),
            "row with None cost should contain em-dash; got:\n{output}",
        );
    }

    #[test]
    fn list_footer_shows_tab_hint() {
        let mut app = App::new(fixture_sessions(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("1/2 tabs"),
            "list footer should contain '1/2 tabs'; got: {last_line}",
        );
    }
}
