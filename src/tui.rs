//! Interactive TUI rendering for the `list` subcommand.
//!
//! Public API:
//! - `run_list_tui(Vec<Session>) -> anyhow::Result<()>` — fullscreen
//!   alternate-screen TUI with scrollable session table, pinned totals
//!   footer, and keyboard navigation. Initializes the terminal on entry
//!   and restores it on exit (including on error).
//!
//! The plain-text path (`rendering::render_table`) remains the fallback
//! for piped output and `--plain`; this module is only entered when
//! stdout is a TTY and `--plain` is not set.

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Style, Stylize};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Row, Table, TableState};
use ratatui::{DefaultTerminal, Frame};

use crate::aggregation::fold_cum_cost;
use crate::domain::Session;
use crate::formatting::{format_cost_opt, format_local, format_tokens};

struct App {
    sessions: Vec<Session>,
    table_state: TableState,
    total_tokens: u64,
    total_cost: Option<f64>,
}

impl App {
    fn new(sessions: Vec<Session>) -> Self {
        let total_tokens = sessions.iter().map(|s| s.total_billable).sum();
        let total_cost = sessions.iter().try_fold(0.0, |acc, s| {
            fold_cum_cost(Some(acc), s.cost_breakdown.map(|b| b.total()))
        });
        let mut table_state = TableState::default();
        if !sessions.is_empty() {
            table_state.select_first();
        }
        Self {
            sessions,
            table_state,
            total_tokens,
            total_cost,
        }
    }
}

#[allow(clippy::missing_errors_doc)]
pub fn run_list_tui(sessions: Vec<Session>) -> anyhow::Result<()> {
    let mut terminal = ratatui::try_init()?;
    let result = run_event_loop(&mut terminal, sessions);
    ratatui::restore();
    result
}

fn run_event_loop(terminal: &mut DefaultTerminal, sessions: Vec<Session>) -> anyhow::Result<()> {
    let mut app = App::new(sessions);
    loop {
        terminal.draw(|frame| render_list(&mut app, frame))?;
        if let Event::Key(key) = event::read()?
            && handle_key_event(&mut app, key)
        {
            break;
        }
    }
    Ok(())
}

fn handle_key_event(app: &mut App, key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return true;
    }
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Down | KeyCode::Char('j') => {
            app.table_state.select_next();
            false
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.table_state.select_previous();
            false
        }
        KeyCode::Home | KeyCode::Char('g') => {
            app.table_state.select_first();
            false
        }
        KeyCode::End | KeyCode::Char('G') => {
            app.table_state.select_last();
            false
        }
        KeyCode::Backspace
        | KeyCode::Enter
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

fn render_list(app: &mut App, frame: &mut Frame) {
    let [header_area, table_area, totals_area, footer_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    render_header(app, frame, header_area);
    render_sessions_table(app, frame, table_area);
    render_totals(app, frame, totals_area);
    render_footer(frame, footer_area);
}

fn render_header(app: &App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let count = app.sessions.len();
    let label = if count == 1 { "session" } else { "sessions" };
    let text = Line::from(format!(" cclens — {count} {label}")).bold();
    frame.render_widget(Paragraph::new(text), area);
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

    frame.render_stateful_widget(table, area, &mut app.table_state);
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

fn render_footer(frame: &mut Frame, area: ratatui::layout::Rect) {
    let text = Line::from(" ↑↓ navigate  q quit").dim();
    frame.render_widget(Paragraph::new(text), area);
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::domain::CostBreakdown;

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

    fn key_event(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    // --- App construction tests ---

    #[test]
    fn app_new_selects_first_when_non_empty() {
        let app = App::new(fixture_sessions());
        assert_eq!(app.table_state.selected(), Some(0));
    }

    #[test]
    fn app_new_no_selection_when_empty() {
        let app = App::new(vec![]);
        assert_eq!(app.table_state.selected(), None);
    }

    #[test]
    fn app_new_computes_totals() {
        let app = App::new(fixture_sessions());
        assert_eq!(app.total_tokens, 1500 + 2500 + 3000);
        let expected_cost = 0.01 + 0.02 + 0.03;
        assert!(
            (app.total_cost.unwrap() - expected_cost).abs() < 1e-10,
            "expected {expected_cost}, got {:?}",
            app.total_cost,
        );
    }

    // --- Key handling tests ---

    #[test]
    fn handle_key_quit_on_q() {
        let mut app = App::new(fixture_sessions());
        assert!(handle_key_event(&mut app, key_event(KeyCode::Char('q'))));
    }

    #[test]
    fn handle_key_quit_on_esc() {
        let mut app = App::new(fixture_sessions());
        assert!(handle_key_event(&mut app, key_event(KeyCode::Esc)));
    }

    #[test]
    fn handle_key_quit_on_ctrl_c() {
        let mut app = App::new(fixture_sessions());
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(handle_key_event(&mut app, key));
    }

    #[test]
    fn handle_key_down_advances_selection() {
        let mut app = App::new(fixture_sessions());
        assert_eq!(app.table_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Down));
        assert_eq!(app.table_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_up_retreats_selection() {
        let mut app = App::new(fixture_sessions());
        app.table_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Up));
        assert_eq!(app.table_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_j_advances_like_down() {
        let mut app = App::new(fixture_sessions());
        assert_eq!(app.table_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Char('j')));
        assert_eq!(app.table_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_k_retreats_like_up() {
        let mut app = App::new(fixture_sessions());
        app.table_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Char('k')));
        assert_eq!(app.table_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_home_selects_first() {
        let mut app = App::new(fixture_sessions());
        app.table_state.select(Some(2));
        handle_key_event(&mut app, key_event(KeyCode::Home));
        assert_eq!(app.table_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_end_selects_last() {
        let sessions = fixture_sessions();
        let last = sessions.len() - 1;
        let mut app = App::new(sessions);
        assert_eq!(app.table_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::End));
        render_app(&mut app, 80, 10);
        assert_eq!(app.table_state.selected(), Some(last));
    }

    #[test]
    fn handle_key_unknown_is_noop() {
        let mut app = App::new(fixture_sessions());
        assert_eq!(app.table_state.selected(), Some(0));
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('x')));
        assert!(!quit);
        assert_eq!(app.table_state.selected(), Some(0));
    }

    // --- Rendering tests (TestBackend) ---

    fn render_app(app: &mut App, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| render_list(app, frame)).unwrap();
        format!("{}", terminal.backend())
    }

    #[test]
    fn list_tui_renders_header_with_session_count() {
        let mut app = App::new(fixture_sessions());
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("cclens — 3 sessions"),
            "header missing session count; got: {first_line}",
        );
    }

    #[test]
    fn list_tui_renders_table_header_columns() {
        let mut app = App::new(fixture_sessions());
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
        let mut app = App::new(fixture_sessions());
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("navigate") && last_line.contains("q quit"),
            "footer key hints missing; got: {last_line}",
        );
    }

    #[test]
    fn list_tui_renders_totals_when_multiple_sessions() {
        let mut app = App::new(fixture_sessions());
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
        let mut app = App::new(vec![fixture_sessions().remove(0)]);
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
        let mut app = App::new(fixture_sessions());
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
}
