//! Interactive TUI rendering for `cclens list` and `cclens inputs`
//! with tabbed navigation, session drill-down, and pricing overlay.
//!
//! Cell data comes from `views` (shared with `rendering`); this
//! module handles ratatui widget construction, styling, layout
//! constraints, and the event loop. Data loading is `loading`'s
//! responsibility — this module consumes `loading::DataContext` and
//! its load functions directly, with no dependency-inversion layer.
//!
//! Public API:
//! - `Tab` — `Sessions` | `Inputs` — the active tab.
//! - `run_tui(DataContext, Vec<Session>, RefreshFingerprint,
//!   PricingData, Tab) -> anyhow::Result<()>` — fullscreen TUI with
//!   tab switching (1/2 keys), scrollable tables, session drill-down
//!   (Enter/Esc within Sessions tab), attribution table with coverage
//!   footer (Inputs tab), pricing overlay (`p` toggles, `r` refreshes,
//!   Esc/q closes), and timer-driven auto-refresh (3s interval,
//!   fingerprint-gated).

use std::sync::Arc;

use crossterm::event::EventStream;
use futures_util::StreamExt;
use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Style, Stylize};
use ratatui::text::Line;
use ratatui::widgets::{Block, Clear, Paragraph, Row, Table, TableState};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc;
use tokio::time::{self, MissedTickBehavior};

use crate::aggregation::{PreparedExchange, PreparedRow};
use crate::attribution::{AttributionRow, CoverageStats};
use crate::domain::Session;
use crate::formatting::{coverage_line, format_cost_opt, format_tokens, tiers_differ};
use crate::loading::{self, DataContext, PricingData, RefreshFingerprint};
use crate::pricing::{CacheInfo, PricingCatalog};
use crate::views::{
    inputs_cells, pricing_view_rows, session_cells, session_totals, show_row_cells,
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

enum InputsData {
    Loading,
    Loaded(InputsState),
    Error(String),
}

enum Overlay {
    Pricing,
}

enum View {
    List,
    ShowLoading {
        session_id: String,
        header_label: String,
    },
    Show {
        session_id: String,
        header_label: String,
        prepared: Vec<PreparedExchange>,
        table_state: TableState,
    },
    ShowError {
        session_id: String,
        header_label: String,
        message: String,
    },
}

/// Which view's data a forced or guarded refresh targets.
enum RefreshScope {
    Sessions,
    Show { session_id: String },
    Inputs,
}

enum LoadRequest {
    ShowDetail {
        session_id: String,
        header_label: String,
    },
    InputsRefresh,
    /// `guard: Some(fp)` is a timer-driven refresh — skipped if the
    /// filesystem is unchanged. `guard: None` is a forced reload
    /// (filter/catalog change) — always runs.
    Refresh {
        scope: RefreshScope,
        guard: Option<RefreshFingerprint>,
    },
    RefreshPricing,
}

enum LoadResult {
    ShowDetail {
        session_id: String,
        header_label: String,
        result: anyhow::Result<Vec<PreparedExchange>>,
        refresh_fingerprint: Option<RefreshFingerprint>,
        is_refresh: bool,
    },
    InputsData {
        result: anyhow::Result<(Vec<AttributionRow>, CoverageStats)>,
        refresh_fingerprint: Option<RefreshFingerprint>,
    },
    SessionsData {
        result: anyhow::Result<Vec<Session>>,
        fingerprint: Option<RefreshFingerprint>,
    },
    Pricing {
        result: anyhow::Result<(Arc<PricingCatalog>, PricingData)>,
    },
    /// A guarded fingerprint build failed (the watched path is
    /// unreadable or gone). Distinct from `NoChange` so a future
    /// status footer can surface it; today it only clears
    /// backpressure, like `NoChange`.
    RefreshFailed {
        message: String,
    },
    NoChange,
}

/// Generation wrapper carried on the result channel. Every dispatched
/// load is stamped with the `ctx_generation` it ran against;
/// `handle_load_result` discards a result whose stamp is behind the
/// app's current generation before applying it, so a load computed
/// against a superseded catalog/query can never overwrite fresher
/// data.
struct StampedResult {
    generation: u64,
    result: LoadResult,
}

/// The status footer's only content today is a failure — every
/// producer in this phase is an error path (`apply_show_detail`,
/// `apply_inputs_data`, `apply_pricing`, `apply_refresh_failed`), so
/// `render_status_footer` styles every message the same way. A
/// severity field (e.g. for a future non-error status like an
/// active-filter indicator) is a one-line addition to make when a
/// second kind of message actually exists — see CLAUDE.md's Project
/// Status on not pre-widening pre-1.0 types speculatively.
struct StatusMessage {
    text: String,
    /// Set only by `apply_refresh_failed`. A later `NoChange` proves
    /// the *guarded-refresh pipeline* recovered — that is meaningful
    /// evidence only against a status this same mechanism raised.
    /// Without this distinction, `NoChange`'s auto-clear (added so a
    /// pricing-refresh error isn't wiped by an unrelated background
    /// tick — see `handle_load_result`) would also permanently trap a
    /// resolved `RefreshFailed` warning: a directory that's fixed by
    /// restoring its exact prior state (e.g. `mv away && mv back`,
    /// which preserves mtime) never produces a *changed* fingerprint,
    /// so `NoChange` — not a fresh successful reload — is the only
    /// signal recovery ever produces for that case.
    from_refresh_failure: bool,
}

struct App {
    /// Everything the next dispatched load needs. Cloned into each
    /// `spawn_blocking` task at dispatch time — the single-threaded
    /// event loop is the only mutator, so no lock is needed.
    ctx: DataContext,
    /// Bumped on every `ctx` mutation (today: catalog swap). Loads
    /// carry the generation they were dispatched with.
    ctx_generation: u64,
    tab: Tab,
    sessions: Vec<Session>,
    list_state: TableState,
    total_tokens: u64,
    total_cost: Option<f64>,
    view: View,
    inputs: Option<InputsData>,
    overlay: Option<Overlay>,
    pricing: PricingData,
    pending_loads: Vec<LoadRequest>,
    refresh_fingerprint: RefreshFingerprint,
    refresh_in_flight: bool,
    pricing_refresh_in_flight: bool,
    /// Replaces the footer's keybinding hint when set. Cleared on any
    /// successful load, alongside `consecutive_refresh_failures`.
    status: Option<StatusMessage>,
    /// Consecutive silent (timer-driven) refresh failures. Reset to 0
    /// on any successful load. At `CONSECUTIVE_REFRESH_FAILURE_THRESHOLD`,
    /// `apply_refresh_failed` breaks silence and sets `status`.
    consecutive_refresh_failures: u8,
}

/// Number of consecutive silent `RefreshFailed` results before the
/// status footer speaks up. A single transient failure stays silent
/// (see the prior design's Error Handling section — the user already
/// has valid data displayed and the next tick retries); this many in
/// a row means the failure probably isn't transient.
const CONSECUTIVE_REFRESH_FAILURE_THRESHOLD: u8 = 3;

impl App {
    fn new(
        ctx: DataContext,
        sessions: Vec<Session>,
        pricing: PricingData,
        default_tab: Tab,
    ) -> Self {
        let (total_tokens, total_cost) =
            session_totals(&sessions).map_or((0, None), |t| (t.total_tokens, t.total_cost));
        let mut list_state = TableState::default();
        if !sessions.is_empty() {
            list_state.select_first();
        }
        Self {
            ctx,
            ctx_generation: 0,
            tab: default_tab,
            sessions,
            list_state,
            total_tokens,
            total_cost,
            view: View::List,
            inputs: None,
            overlay: None,
            pricing,
            pending_loads: Vec::new(),
            refresh_fingerprint: RefreshFingerprint::default(),
            refresh_in_flight: false,
            pricing_refresh_in_flight: false,
            status: None,
            consecutive_refresh_failures: 0,
        }
    }

    /// Clear the status footer and reset the failure counter. Called
    /// by every applier on a successful load.
    fn clear_status(&mut self) {
        self.status = None;
        self.consecutive_refresh_failures = 0;
    }

    /// Reset only the failure counter, leaving `status` untouched.
    /// Used by `NoChange`: an unchanged fingerprint proves the guarded
    /// refresh pipeline is healthy (evidence against a *persistent*
    /// failure, resetting the threshold), but it is not itself
    /// evidence against whatever specific status is currently
    /// displayed — a user-triggered error (e.g. a failed pricing
    /// refresh) must survive an unrelated background tick finding
    /// nothing changed, or it would vanish within one ~3s cycle.
    fn reset_refresh_failures(&mut self) {
        self.consecutive_refresh_failures = 0;
    }

    fn selected_session_id(&self) -> Option<&str> {
        let idx = self.list_state.selected()?;
        self.sessions.get(idx).map(|s| s.id.as_str())
    }

    fn apply_sessions_refresh(
        &mut self,
        new_sessions: Vec<Session>,
        new_fingerprint: RefreshFingerprint,
    ) {
        let prev_id = self.selected_session_id().map(str::to_owned);
        let prev_idx = self.list_state.selected();

        let (total_tokens, total_cost) =
            session_totals(&new_sessions).map_or((0, None), |t| (t.total_tokens, t.total_cost));
        self.sessions = new_sessions;
        self.total_tokens = total_tokens;
        self.total_cost = total_cost;
        self.refresh_fingerprint = new_fingerprint;

        if let Some(prev_id) = prev_id {
            if let Some(new_idx) = self.sessions.iter().position(|s| s.id == prev_id) {
                self.list_state.select(Some(new_idx));
            } else if self.sessions.is_empty() {
                self.list_state.select(None);
            } else {
                let fallback = prev_idx.unwrap_or(0).min(self.sessions.len() - 1);
                self.list_state.select(Some(fallback));
            }
        } else if !self.sessions.is_empty() {
            self.list_state.select_first();
        }
    }
}

/// Maps the current tab/view to the `RefreshScope` a refresh should
/// target, or `None` when nothing on screen has loaded data to
/// refresh. Shared by `build_refresh_request` (guarded, timer-driven)
/// and `apply_pricing` (forced, after a catalog swap) so both refresh
/// triggers agree on "what's on screen right now".
fn current_scope(app: &App) -> Option<RefreshScope> {
    match (&app.tab, &app.view, &app.inputs) {
        (Tab::Sessions, View::List, _) => Some(RefreshScope::Sessions),
        (Tab::Sessions, View::Show { session_id, .. }, _) => Some(RefreshScope::Show {
            session_id: session_id.clone(),
        }),
        (Tab::Inputs, _, Some(InputsData::Loaded(_))) => Some(RefreshScope::Inputs),
        _ => None,
    }
}

fn build_refresh_request(app: &App) -> Option<LoadRequest> {
    let scope = current_scope(app)?;
    Some(LoadRequest::Refresh {
        scope,
        guard: Some(app.refresh_fingerprint.clone()),
    })
}

/// # Errors
///
/// Propagates a terminal init/restore failure or an unrecoverable
/// event-loop error.
pub async fn run_tui(
    ctx: DataContext,
    sessions: Vec<Session>,
    initial_fingerprint: RefreshFingerprint,
    pricing: PricingData,
    default_tab: Tab,
) -> anyhow::Result<()> {
    let mut terminal = ratatui::try_init()?;
    let result = run_event_loop(
        &mut terminal,
        ctx,
        sessions,
        initial_fingerprint,
        pricing,
        default_tab,
    )
    .await;
    ratatui::restore();
    result
}

async fn run_event_loop(
    terminal: &mut DefaultTerminal,
    ctx: DataContext,
    sessions: Vec<Session>,
    initial_fingerprint: RefreshFingerprint,
    pricing: PricingData,
    default_tab: Tab,
) -> anyhow::Result<()> {
    let mut app = App::new(ctx, sessions, pricing, default_tab);
    app.refresh_fingerprint = initial_fingerprint;
    let mut event_stream = EventStream::new();

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<StampedResult>();

    let mut refresh_interval = time::interval(std::time::Duration::from_secs(3));
    refresh_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    refresh_interval.tick().await;

    if default_tab == Tab::Inputs {
        app.inputs = Some(InputsData::Loading);
        app.pending_loads.push(LoadRequest::InputsRefresh);
    }

    loop {
        drain_pending_loads(&mut app, &result_tx);
        terminal.draw(|frame| render(&mut app, frame))?;

        tokio::select! {
            biased;
            event = event_stream.next() => {
                match event {
                    Some(Ok(Event::Key(key))) => {
                        if handle_key_event(&mut app, key) {
                            break;
                        }
                    }
                    Some(Err(_)) | None => break,
                    _ => {}
                }
            }
            Some(stamped) = result_rx.recv() => {
                handle_load_result(&mut app, stamped);
            }
            _ = refresh_interval.tick() => {
                if !app.refresh_in_flight
                    && let Some(request) = build_refresh_request(&app)
                {
                    app.refresh_in_flight = true;
                    app.pending_loads.push(request);
                }
            }
        }
    }
    Ok(())
}

/// Dispatch one load onto a blocking task and send its stamped
/// outcome back over `tx`. Owns the shape shared by every dispatch
/// site: clone `ctx`, `spawn_blocking`, `catch_unwind` (panic → an
/// `Err`/`RefreshFailed` outcome depending on `guard`), gate on
/// `guard` when present, run `load`, map the outcome, stamp the
/// dispatch generation.
///
/// `guard: Some(old)` is a timer-driven refresh: a fresh fingerprint
/// is built first. A build failure sends `LoadResult::RefreshFailed`;
/// an unchanged fingerprint sends `LoadResult::NoChange`; otherwise
/// `load` runs and `map(result, Some(new_fp))` is sent. `guard: None`
/// is a forced or user-triggered load: `load` runs unconditionally
/// and `map` is called with `None`.
fn spawn_load<T, L, M>(
    ctx: &DataContext,
    generation: u64,
    guard: Option<RefreshFingerprint>,
    tx: &mpsc::UnboundedSender<StampedResult>,
    load: L,
    map: M,
) where
    T: Send + 'static,
    L: FnOnce(&DataContext) -> anyhow::Result<T> + Send + 'static,
    M: FnOnce(anyhow::Result<T>, Option<RefreshFingerprint>) -> LoadResult + Send + 'static,
{
    let ctx = ctx.clone();
    let tx = tx.clone();
    tokio::task::spawn_blocking(move || {
        let result = match guard {
            None => {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load(&ctx)))
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("internal error: loader panicked")));
                map(outcome, None)
            }
            Some(old) => {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> anyhow::Result<Option<(anyhow::Result<T>, RefreshFingerprint)>> {
                        let new_fp = loading::build_fingerprint(&ctx.projects_dir)?;
                        if new_fp == old {
                            return Ok(None);
                        }
                        Ok(Some((load(&ctx), new_fp)))
                    },
                ));
                match outcome {
                    Ok(Ok(None)) => LoadResult::NoChange,
                    Ok(Ok(Some((result, new_fp)))) => map(result, Some(new_fp)),
                    Ok(Err(e)) => LoadResult::RefreshFailed {
                        message: format!("{e}"),
                    },
                    Err(_) => LoadResult::RefreshFailed {
                        message: "internal error: loader panicked".to_string(),
                    },
                }
            }
        };
        let _ = tx.send(StampedResult { generation, result });
    });
}

fn drain_pending_loads(app: &mut App, tx: &mpsc::UnboundedSender<StampedResult>) {
    for request in app.pending_loads.drain(..) {
        let ctx = &app.ctx;
        let generation = app.ctx_generation;
        match request {
            LoadRequest::ShowDetail {
                session_id,
                header_label,
            } => {
                let sid = session_id.clone();
                spawn_load(
                    ctx,
                    generation,
                    None,
                    tx,
                    move |c| loading::load_show(c, &session_id),
                    move |result, _fp| LoadResult::ShowDetail {
                        session_id: sid,
                        header_label,
                        result,
                        refresh_fingerprint: None,
                        is_refresh: false,
                    },
                );
            }
            LoadRequest::InputsRefresh => {
                spawn_load(
                    ctx,
                    generation,
                    None,
                    tx,
                    loading::load_inputs,
                    |result, _fp| LoadResult::InputsData {
                        result,
                        refresh_fingerprint: None,
                    },
                );
            }
            LoadRequest::Refresh { scope, guard } => match scope {
                RefreshScope::Sessions => {
                    spawn_load(
                        ctx,
                        generation,
                        guard,
                        tx,
                        loading::load_sessions,
                        |result, fp| LoadResult::SessionsData {
                            result,
                            fingerprint: fp,
                        },
                    );
                }
                RefreshScope::Show { session_id } => {
                    let sid = session_id.clone();
                    spawn_load(
                        ctx,
                        generation,
                        guard,
                        tx,
                        move |c| loading::load_show(c, &session_id),
                        move |result, fp| LoadResult::ShowDetail {
                            session_id: sid,
                            header_label: String::new(),
                            result,
                            refresh_fingerprint: fp,
                            is_refresh: true,
                        },
                    );
                }
                RefreshScope::Inputs => {
                    spawn_load(
                        ctx,
                        generation,
                        guard,
                        tx,
                        loading::load_inputs,
                        |result, fp| LoadResult::InputsData {
                            result,
                            refresh_fingerprint: fp,
                        },
                    );
                }
            },
            LoadRequest::RefreshPricing => {
                spawn_load(
                    ctx,
                    generation,
                    None,
                    tx,
                    |_ctx| loading::refresh_pricing(),
                    |result, _fp| LoadResult::Pricing { result },
                );
            }
        }
    }
}

fn handle_load_result(app: &mut App, stamped: StampedResult) {
    if stamped.generation < app.ctx_generation {
        // Superseded context: the payload is computed against stale
        // state. Still clear backpressure, or the timer never
        // refreshes again. Both flags are cleared defensively — only
        // `refresh_in_flight` is reachable here today (`ctx_generation`
        // bumps only inside `apply_pricing`, and `pricing_refresh_in_flight`
        // already enforces one in-flight pricing load) — but a dropped
        // `Pricing` result must not leave `r` permanently dead if a
        // second generation source is ever added.
        app.refresh_in_flight = false;
        app.pricing_refresh_in_flight = false;
        return;
    }
    match stamped.result {
        LoadResult::ShowDetail {
            session_id,
            header_label,
            result,
            refresh_fingerprint,
            is_refresh,
        } => apply_show_detail(
            app,
            session_id,
            header_label,
            result,
            refresh_fingerprint,
            is_refresh,
        ),
        LoadResult::InputsData {
            result,
            refresh_fingerprint,
        } => apply_inputs_data(app, result, refresh_fingerprint),
        LoadResult::SessionsData {
            result,
            fingerprint,
        } => apply_sessions_data(app, result, fingerprint),
        LoadResult::Pricing { result } => apply_pricing(app, result),
        LoadResult::RefreshFailed { message } => apply_refresh_failed(app, &message),
        LoadResult::NoChange => {
            app.refresh_in_flight = false;
            // An unchanged fingerprint means the guarded fingerprint
            // build itself succeeded — always evidence the failure
            // counter should reset. Whether it also clears `status`
            // depends on who set it: a warning `apply_refresh_failed`
            // raised is provably resolved by this same signal (e.g.
            // recovery via `mv away && mv back`, which restores an
            // identical fingerprint and so can *only* ever be
            // observed as `NoChange`, never a changed-data reload).
            // A status from anywhere else (a failed pricing refresh)
            // is unrelated — clearing it here would let it vanish
            // within one ~3s idle tick before anyone reads it.
            match &app.status {
                Some(status) if status.from_refresh_failure => app.clear_status(),
                _ => app.reset_refresh_failures(),
            }
        }
    }
}

fn apply_show_detail(
    app: &mut App,
    session_id: String,
    header_label: String,
    result: anyhow::Result<Vec<PreparedExchange>>,
    refresh_fingerprint: Option<RefreshFingerprint>,
    is_refresh: bool,
) {
    let is_user_triggered = !is_refresh
        && matches!(
            &app.view,
            View::ShowLoading { session_id: sid, .. } if *sid == session_id
        );
    let is_refresh_triggered = is_refresh
        && matches!(
            &app.view,
            View::Show { session_id: sid, .. } if *sid == session_id
        );

    if is_user_triggered {
        match result {
            Ok(prepared) => {
                let mut table_state = TableState::default();
                if !prepared.is_empty() {
                    table_state.select_first();
                }
                app.view = View::Show {
                    session_id,
                    header_label,
                    prepared,
                    table_state,
                };
                app.clear_status();
            }
            Err(e) => {
                let message = format!("{e}");
                // User-triggered (Enter / retry) — the user is owed
                // an answer immediately, no threshold.
                app.status = Some(StatusMessage {
                    text: message.clone(),
                    from_refresh_failure: false,
                });
                app.view = View::ShowError {
                    session_id,
                    header_label,
                    message,
                };
            }
        }
    } else if is_refresh_triggered
        && let Ok(new_prepared) = result
        && let View::Show {
            prepared,
            table_state,
            ..
        } = &mut app.view
    {
        let row_count: usize = prepared.iter().map(|e| e.rows.len()).sum();
        let was_at_end = row_count > 0 && table_state.selected() == Some(row_count - 1);
        *prepared = new_prepared;
        if was_at_end {
            let new_row_count: usize = prepared.iter().map(|e| e.rows.len()).sum();
            if new_row_count > 0 {
                table_state.select(Some(new_row_count - 1));
            }
        }
        app.clear_status();
    }
    // Clearing is gated on `refresh_fingerprint.is_some()`, not on
    // `is_refresh` — `is_refresh` is also `true` for a *forced*
    // (`guard: None`) Show reload dispatched after a catalog swap,
    // which never set `refresh_in_flight` in the first place. Clearing
    // it there would let the next timer tick dispatch a second guarded
    // load while a real one is still in flight. `NoChange` and
    // `RefreshFailed` — the only other outcomes a guarded dispatch can
    // produce — clear it in their own `handle_load_result` arms.
    if let Some(fp) = refresh_fingerprint {
        app.refresh_fingerprint = fp;
        app.refresh_in_flight = false;
    }
}

fn apply_inputs_data(
    app: &mut App,
    result: anyhow::Result<(Vec<AttributionRow>, CoverageStats)>,
    refresh_fingerprint: Option<RefreshFingerprint>,
) {
    let is_user_triggered =
        refresh_fingerprint.is_none() && matches!(&app.inputs, Some(InputsData::Loading));
    // Not gated on `refresh_fingerprint.is_some()`: a forced reload
    // (catalog swap) dispatches with `guard: None`, so a fingerprint-
    // free result must still be treated as a refresh when the inputs
    // table is already loaded — otherwise a pricing-catalog swap
    // would never recompute costs on a visible Inputs tab.
    let is_refresh_triggered = matches!(&app.inputs, Some(InputsData::Loaded(_)));

    if is_user_triggered {
        match result {
            Ok((rows, coverage)) => {
                app.inputs = Some(InputsData::Loaded(InputsState::new(rows, coverage)));
                app.clear_status();
            }
            Err(e) => {
                let message = format!("{e}");
                app.status = Some(StatusMessage {
                    text: message.clone(),
                    from_refresh_failure: false,
                });
                app.inputs = Some(InputsData::Error(message));
            }
        }
    } else if is_refresh_triggered
        && let Ok((new_rows, new_coverage)) = result
        && let Some(InputsData::Loaded(state)) = &mut app.inputs
    {
        let prev_path = state
            .table_state
            .selected()
            .and_then(|idx| state.rows.get(idx))
            .map(|r| r.file.path.clone());
        let prev_idx = state.table_state.selected();

        state.rows = new_rows;
        state.coverage = new_coverage;

        if let Some(prev_path) = prev_path {
            if let Some(new_idx) = state.rows.iter().position(|r| r.file.path == prev_path) {
                state.table_state.select(Some(new_idx));
            } else if state.rows.is_empty() {
                state.table_state.select(None);
            } else {
                let fallback = prev_idx.unwrap_or(0).min(state.rows.len() - 1);
                state.table_state.select(Some(fallback));
            }
        } else if !state.rows.is_empty() {
            state.table_state.select_first();
        }
        app.clear_status();
    }
    if let Some(fp) = refresh_fingerprint {
        app.refresh_fingerprint = fp;
        app.refresh_in_flight = false;
    }
}

fn apply_sessions_data(
    app: &mut App,
    result: anyhow::Result<Vec<Session>>,
    fingerprint: Option<RefreshFingerprint>,
) {
    // Only a guarded (timer-driven) dispatch ever set `refresh_in_flight`
    // — a forced (`guard: None`) reload after a catalog swap carries no
    // fingerprint and never acquired the flag, so it must not clear it
    // out from under a real guarded load still in flight.
    let was_guarded = fingerprint.is_some();
    if app.tab != Tab::Sessions || !matches!(app.view, View::List) {
        if was_guarded {
            app.refresh_in_flight = false;
        }
        return;
    }
    if let Ok(sessions) = result {
        // A forced reload (catalog swap) carries no fresh fingerprint
        // — the filesystem didn't change, so the existing one stands.
        let fp = fingerprint.unwrap_or_else(|| app.refresh_fingerprint.clone());
        app.apply_sessions_refresh(sessions, fp);
        app.clear_status();
    }
    if was_guarded {
        app.refresh_in_flight = false;
    }
}

fn apply_pricing(app: &mut App, result: anyhow::Result<(Arc<PricingCatalog>, PricingData)>) {
    match result {
        Ok((catalog, data)) => {
            app.pricing = data;
            app.ctx.catalog = catalog;
            app.ctx_generation += 1;
            if let Some(scope) = current_scope(app) {
                app.pending_loads
                    .push(LoadRequest::Refresh { scope, guard: None });
            }
            app.clear_status();
        }
        Err(e) => {
            // User pressed `r` in the overlay and is owed an answer
            // immediately — no threshold, unlike `apply_refresh_failed`.
            app.status = Some(StatusMessage {
                text: format!("{e}"),
                from_refresh_failure: false,
            });
        }
    }
    app.pricing_refresh_in_flight = false;
}

fn apply_refresh_failed(app: &mut App, message: &str) {
    app.refresh_in_flight = false;
    // Single transient failures stay silent — the prior design's
    // rationale holds: valid data is still displayed and the next
    // tick retries. Only a persistent, repeated failure breaks
    // silence, since by then "wait for the next tick" has stopped
    // being a credible remedy.
    app.consecutive_refresh_failures = app.consecutive_refresh_failures.saturating_add(1);
    if app.consecutive_refresh_failures >= CONSECUTIVE_REFRESH_FAILURE_THRESHOLD {
        app.status = Some(StatusMessage {
            // Diagnostic first: the footer is one row, so on a narrow
            // terminal a long `message` truncates — put the part
            // worth reading before the boilerplate, not after it.
            text: format!("{message} — refresh failing, data may be stale"),
            from_refresh_failure: true,
        });
    }
}

fn handle_key_event(app: &mut App, key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return true;
    }
    if app.overlay.is_some() {
        match key.code {
            KeyCode::Esc | KeyCode::Char('p' | 'q') => {
                app.overlay = None;
            }
            KeyCode::Char('r') => {
                if !app.pricing_refresh_in_flight {
                    app.pricing_refresh_in_flight = true;
                    app.pending_loads.push(LoadRequest::RefreshPricing);
                }
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
        return false;
    }
    match key.code {
        KeyCode::Char('1') => {
            app.tab = Tab::Sessions;
            return false;
        }
        KeyCode::Char('2') => {
            try_switch_to_inputs(app);
            return false;
        }
        KeyCode::Char('p') => {
            app.overlay = Some(Overlay::Pricing);
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
        Tab::Sessions => match app.view {
            View::Show { .. } => handle_show_key(app, key),
            View::ShowLoading { .. } => handle_show_loading_key(app, key),
            View::ShowError { .. } => handle_show_error_key(app, key),
            View::List => handle_list_key(app, key),
        },
        Tab::Inputs => handle_inputs_key(app, key),
    }
}

fn handle_list_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Enter => {
            try_open_show(app);
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

fn try_open_show(app: &mut App) {
    let Some(idx) = app.list_state.selected() else {
        return;
    };
    let session_id = app.sessions[idx].id.clone();
    let header_label = format!(
        "\"{}\" ({})",
        app.sessions[idx].title, app.sessions[idx].project_short_name,
    );
    app.pending_loads.push(LoadRequest::ShowDetail {
        session_id: session_id.clone(),
        header_label: header_label.clone(),
    });
    app.view = View::ShowLoading {
        session_id,
        header_label,
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

fn handle_show_loading_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') => true,
        KeyCode::Esc | KeyCode::Backspace => {
            app.view = View::List;
            false
        }
        KeyCode::Enter
        | KeyCode::Down
        | KeyCode::Up
        | KeyCode::Left
        | KeyCode::Right
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

fn handle_show_error_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') => true,
        KeyCode::Esc | KeyCode::Backspace => {
            app.view = View::List;
            false
        }
        KeyCode::Enter | KeyCode::Char('r') => {
            let View::ShowError {
                session_id,
                header_label,
                ..
            } = &app.view
            else {
                return false;
            };
            let session_id = session_id.clone();
            let header_label = header_label.clone();
            app.pending_loads.push(LoadRequest::ShowDetail {
                session_id: session_id.clone(),
                header_label: header_label.clone(),
            });
            app.view = View::ShowLoading {
                session_id,
                header_label,
            };
            false
        }
        KeyCode::Down
        | KeyCode::Up
        | KeyCode::Left
        | KeyCode::Right
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

fn try_switch_to_inputs(app: &mut App) {
    if matches!(app.inputs, None | Some(InputsData::Error(_))) {
        app.inputs = Some(InputsData::Loading);
        app.pending_loads.push(LoadRequest::InputsRefresh);
    }
    app.tab = Tab::Inputs;
}

fn handle_inputs_loading_key(key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Enter
        | KeyCode::Backspace
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

fn handle_inputs_key(app: &mut App, key: KeyEvent) -> bool {
    if matches!(&app.inputs, Some(InputsData::Loading)) {
        return handle_inputs_loading_key(key);
    }
    if matches!(&app.inputs, Some(InputsData::Error(_))) {
        return match key.code {
            KeyCode::Char('q') | KeyCode::Esc => true,
            KeyCode::Enter | KeyCode::Char('r') => {
                try_switch_to_inputs(app);
                false
            }
            KeyCode::Backspace
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
            | KeyCode::CapsLock
            | KeyCode::ScrollLock
            | KeyCode::NumLock
            | KeyCode::PrintScreen
            | KeyCode::Pause
            | KeyCode::Menu
            | KeyCode::KeypadBegin
            | KeyCode::Media(_)
            | KeyCode::Modifier(_) => false,
        };
    }
    let Some(InputsData::Loaded(inputs)) = &mut app.inputs else {
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
        Tab::Sessions => match &app.view {
            View::ShowLoading { .. } => {
                render_feedback_content(
                    app,
                    frame,
                    content_area,
                    vec![Line::raw(""), Line::from(" Loading session...")],
                    " 1/2 tabs  Esc back  q quit",
                );
            }
            View::ShowError { message, .. } => {
                render_feedback_content(
                    app,
                    frame,
                    content_area,
                    vec![
                        Line::raw(""),
                        Line::from(format!(" Error: {message}")),
                        Line::raw(""),
                        Line::from(" Press Enter or r to retry.").dim(),
                    ],
                    " 1/2 tabs  Enter retry  Esc back  q quit",
                );
            }
            View::Show { .. } => {
                render_show_content(app, frame, content_area);
            }
            View::List => {
                render_list_content(app, frame, content_area);
            }
        },
        Tab::Inputs => render_inputs_content(app, frame, content_area),
    }

    if app.overlay.is_some() {
        render_pricing_overlay(app, frame);
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
        Tab::Sessions => match &app.view {
            View::Show { header_label, .. }
            | View::ShowLoading { header_label, .. }
            | View::ShowError { header_label, .. } => header_label.clone(),
            View::List => {
                let count = app.sessions.len();
                let label = if count == 1 { "session" } else { "sessions" };
                format!("{count} {label}")
            }
        },
        Tab::Inputs => match &app.inputs {
            Some(InputsData::Loaded(inputs)) => {
                let count = inputs.rows.len();
                let label = if count == 1 { "file" } else { "files" };
                format!("{count} {label}")
            }
            Some(InputsData::Loading) => "...".to_string(),
            Some(InputsData::Error(_)) => "!".to_string(),
            None => "0 files".to_string(),
        },
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
    render_status_footer(
        app,
        frame,
        footer_area,
        " 1/2 tabs  ↑↓ navigate  Enter open  p pricing  q quit",
    );
}

fn render_sessions_table(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let rows: Vec<Row> = app
        .sessions
        .iter()
        .map(|s| {
            let cells = session_cells(s);
            Row::new(vec![
                cells.datetime,
                cells.project,
                cells.title,
                cells.tokens,
                cells.cost,
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

/// Renders `app.status` (styled red — see `StatusMessage`'s doc
/// comment on why there's no severity field yet) in place of `hint`
/// when set. Shared by every footer row so error/refresh-failure
/// feedback surfaces uniformly across List, Show, Inputs, and the
/// loading/error feedback views — one status slot, four render sites.
fn render_status_footer(app: &App, frame: &mut Frame, area: ratatui::layout::Rect, hint: &str) {
    let line = match &app.status {
        Some(status) => Line::from(format!(" {}", status.text)).red(),
        None => Line::from(hint).dim(),
    };
    frame.render_widget(Paragraph::new(line), area);
}

// ---- show view ----

fn render_show_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let [table_area, footer_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(area);

    let View::Show {
        prepared,
        table_state,
        ..
    } = &mut app.view
    else {
        return;
    };
    render_show_table(prepared, table_state, frame, table_area);

    render_status_footer(
        app,
        frame,
        footer_area,
        " 1/2 tabs  ↑↓ navigate  Esc back  p pricing  q quit",
    );
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
    let cells = show_row_cells(row);
    let r = Row::new(vec![
        Line::raw(cells.datetime),
        Line::raw(cells.role),
        Line::raw(cells.tokens).right_aligned(),
        Line::raw(format_cost_opt(cells.cost.map(|b| b.input))).right_aligned(),
        Line::raw(format_cost_opt(cells.cost.map(|b| b.output))).right_aligned(),
        Line::raw(format_cost_opt(cells.cost.map(|b| b.cache_creation_5m))).right_aligned(),
        Line::raw(format_cost_opt(cells.cost.map(|b| b.cache_creation_1h))).right_aligned(),
        Line::raw(format_cost_opt(cells.cost.map(|b| b.cache_read))).right_aligned(),
        Line::raw(cells.cumulative_tokens).right_aligned(),
        Line::raw(cells.cumulative_cost).right_aligned(),
        Line::raw(cells.content),
    ]);
    if dim { r.dim() } else { r }
}

// ---- shared feedback rendering ----

fn render_feedback_content(
    app: &App,
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    message_lines: Vec<Line<'static>>,
    footer_text: &str,
) {
    let [content_area, footer_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(area);
    frame.render_widget(Paragraph::new(message_lines), content_area);
    render_status_footer(app, frame, footer_area, footer_text);
}

// ---- inputs view ----

fn render_inputs_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    match &mut app.inputs {
        Some(InputsData::Loaded(inputs)) => {
            let [table_area, coverage_area, footer_area] = Layout::vertical([
                Constraint::Fill(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .areas(area);

            render_inputs_table(inputs, frame, table_area);
            render_inputs_coverage(inputs, frame, coverage_area);
            render_status_footer(
                app,
                frame,
                footer_area,
                " 1/2 tabs  ↑↓ navigate  p pricing  q quit",
            );
        }
        Some(InputsData::Loading) => {
            render_feedback_content(
                app,
                frame,
                area,
                vec![Line::raw(""), Line::from(" Loading inputs...")],
                " 1/2 tabs  q quit",
            );
        }
        Some(InputsData::Error(msg)) => {
            // Built before the call, not inline as an argument: `msg`
            // borrows `app.inputs` mutably (this match's scrutinee),
            // and that borrow must end before `app` is reborrowed for
            // the status footer — argument evaluation is left-to-right,
            // so `app` would otherwise be reborrowed while `msg` is
            // still live for a later argument.
            let lines = vec![
                Line::raw(""),
                Line::from(format!(" Error: {msg}")),
                Line::raw(""),
                Line::from(" Press Enter or r to retry.").dim(),
            ];
            render_feedback_content(app, frame, area, lines, " 1/2 tabs  Enter retry  q quit");
        }
        None => {}
    }
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
            let cells = inputs_cells(row);
            Row::new(vec![
                Line::raw(cells.file_path),
                Line::raw(cells.kind),
                Line::raw(cells.tier),
                Line::raw(cells.tokens).right_aligned(),
                Line::raw(cells.loads).right_aligned(),
                Line::raw(cells.billed).right_aligned(),
                Line::raw(cells.cost).right_aligned(),
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

// ---- pricing overlay ----

fn render_pricing_overlay(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let popup_width = 80.min(area.width);
    #[allow(clippy::cast_possible_truncation)]
    let row_count: u16 = app
        .pricing
        .entries
        .iter()
        .map(|(_, p)| if tiers_differ(p) { 2u16 } else { 1u16 })
        .sum();
    let popup_height = (row_count + 4).min(area.height);

    let popup_area = centered_rect(popup_width, popup_height, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::bordered().title(" Pricing ($/MTok) ");
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let [table_area, footer_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

    let mut rows = Vec::new();
    for (model, pricing) in &app.pricing.entries {
        for cells in pricing_view_rows(model, pricing) {
            rows.push(Row::new(cells));
        }
    }

    let header = Row::new([
        "model", "tier", "input", "output", "cache_rd", "cache_5m", "cache_1h",
    ])
    .style(Style::new().bold());
    let widths = [
        Constraint::Fill(1),   // model
        Constraint::Length(6), // tier
        Constraint::Length(8), // input
        Constraint::Length(8), // output
        Constraint::Length(8), // cache_rd
        Constraint::Length(8), // cache_5m
        Constraint::Length(8), // cache_1h
    ];
    let table = Table::new(rows, widths).header(header);
    frame.render_widget(table, table_area);

    let footer = if app.pricing_refresh_in_flight {
        Line::from(" refreshing\u{2026}  Esc close").dim()
    } else {
        let staleness = format_cache_staleness(&app.pricing.cache_info);
        Line::from(format!(" {staleness}  r refresh  Esc close")).dim()
    };
    frame.render_widget(Paragraph::new(footer), footer_area);
}

fn centered_rect(width: u16, height: u16, area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    ratatui::layout::Rect::new(x, y, width.min(area.width), height.min(area.height))
}

fn format_cache_staleness(info: &CacheInfo) -> String {
    match &info.last_modified {
        Some(mtime) => {
            let elapsed = mtime.elapsed().unwrap_or_default();
            let hours = elapsed.as_secs() / 3600;
            let days = hours / 24;
            if days > 0 {
                format!("catalog: {days}d ago")
            } else if hours > 0 {
                format!("catalog: {hours}h ago")
            } else {
                "catalog: <1h ago".to_string()
            }
        }
        None => "catalog: unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::SystemTime;

    use chrono::{DateTime, Utc};
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::aggregation::{PreparedExchange, PreparedRowRole};
    use crate::attribution::TierCoverage;
    use crate::domain::{CostBreakdown, TurnOrigin};
    use crate::inventory::{ContextFile, ContextFileKind, Scope};
    use crate::pricing::{ClaudePricing, TieredRate};

    /// A `DataContext` pointed at a nonexistent directory with an
    /// empty pricing catalog — every test in this module drives `App`
    /// state directly rather than through a real load, so the context
    /// itself is never dereferenced.
    fn fixture_ctx() -> DataContext {
        DataContext {
            projects_dir: PathBuf::from("/nonexistent"),
            catalog: Arc::new(PricingCatalog::default()),
            query: crate::loading::Query::default(),
        }
    }

    fn new_app(sessions: Vec<Session>, pricing: PricingData, default_tab: Tab) -> App {
        App::new(fixture_ctx(), sessions, pricing, default_tab)
    }

    /// Wraps `handle_load_result` for call sites that don't care about
    /// generation staleness — stamps with the app's current
    /// generation, which is never considered stale.
    fn hlr(app: &mut App, result: LoadResult) {
        let generation = app.ctx_generation;
        handle_load_result(app, StampedResult { generation, result });
    }

    fn fixture_pricing_data() -> PricingData {
        let uniform = |rate: f64| {
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
        };
        let split = ClaudePricing {
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
        };
        PricingData {
            entries: vec![
                ("claude-haiku-4-5".to_string(), uniform(0.8e-6)),
                ("claude-sonnet-4-5".to_string(), split),
            ],
            cache_info: CacheInfo {
                path: None,
                exists: true,
                last_modified: Some(SystemTime::now()),
                size: 1024,
                entry_count: Some(2),
            },
        }
    }

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

    fn key_event(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn inputs_table_selected(app: &App) -> Option<usize> {
        match &app.inputs {
            Some(InputsData::Loaded(inputs)) => inputs.table_state.selected(),
            Some(InputsData::Loading | InputsData::Error(_)) | None => None,
        }
    }

    fn set_show_view(app: &mut App) {
        let mut table_state = TableState::default();
        table_state.select_first();
        app.view = View::Show {
            session_id: "aaa".to_string(),
            header_label: "\"First session\" (alpha)".to_string(),
            prepared: fixture_prepared_exchanges(),
            table_state,
        };
    }

    // --- App construction tests ---

    #[test]
    fn app_new_selects_first_when_non_empty() {
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn app_new_no_selection_when_empty() {
        let app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        assert_eq!(app.list_state.selected(), None);
    }

    #[test]
    fn app_new_computes_totals() {
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
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
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn enter_transitions_to_show_loading() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(matches!(app.view, View::ShowLoading { .. }));
        assert_eq!(app.pending_loads.len(), 1);
        assert!(matches!(
            app.pending_loads[0],
            LoadRequest::ShowDetail { .. }
        ));
    }

    #[test]
    fn enter_with_no_selection_is_noop() {
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn esc_in_show_returns_to_list() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn backspace_in_show_returns_to_list() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Backspace));
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn q_in_show_quits() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        assert!(handle_key_event(&mut app, key_event(KeyCode::Char('q'))));
    }

    #[test]
    fn list_selection_preserved_after_roundtrip() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.list_state.select(Some(2));
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert_eq!(app.list_state.selected(), Some(2));
    }

    // --- List key handling tests ---

    #[test]
    fn handle_key_quit_on_q() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(handle_key_event(&mut app, key_event(KeyCode::Char('q'))));
    }

    #[test]
    fn handle_key_quit_on_esc() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(handle_key_event(&mut app, key_event(KeyCode::Esc)));
    }

    #[test]
    fn handle_key_quit_on_ctrl_c() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(handle_key_event(&mut app, key));
    }

    #[test]
    fn handle_key_down_advances_selection() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert_eq!(app.list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Down));
        assert_eq!(app.list_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_up_retreats_selection() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.list_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Up));
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_j_advances_like_down() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert_eq!(app.list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Char('j')));
        assert_eq!(app.list_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_k_retreats_like_up() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.list_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Char('k')));
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_home_selects_first() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.list_state.select(Some(2));
        handle_key_event(&mut app, key_event(KeyCode::Home));
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_end_selects_last() {
        let sessions = fixture_sessions();
        let last = sessions.len() - 1;
        let mut app = new_app(sessions, fixture_pricing_data(), Tab::Sessions);
        assert_eq!(app.list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::End));
        render_app(&mut app, 80, 10);
        assert_eq!(app.list_state.selected(), Some(last));
    }

    #[test]
    fn handle_key_unknown_is_noop() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert_eq!(app.list_state.selected(), Some(0));
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('x')));
        assert!(!quit);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    // --- Show view key handling tests ---

    #[test]
    fn show_down_advances_selection() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(0));
        }
        handle_key_event(&mut app, key_event(KeyCode::Down));
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(1));
        }
    }

    #[test]
    fn show_up_retreats_selection() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Down));
        handle_key_event(&mut app, key_event(KeyCode::Up));
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(0));
        }
    }

    #[test]
    fn show_home_selects_first() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Down));
        handle_key_event(&mut app, key_event(KeyCode::Down));
        handle_key_event(&mut app, key_event(KeyCode::Home));
        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(table_state.selected(), Some(0));
        }
    }

    #[test]
    fn show_end_selects_last() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::End));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("cclens — 3 sessions"),
            "header missing session count; got: {first_line}",
        );
    }

    #[test]
    fn list_tui_renders_table_header_columns() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("navigate") && last_line.contains("q quit"),
            "footer key hints missing; got: {last_line}",
        );
    }

    #[test]
    fn list_tui_renders_totals_when_multiple_sessions() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
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
        let mut app = new_app(
            vec![fixture_sessions().remove(0)],
            fixture_pricing_data(),
            Tab::Sessions,
        );
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("Enter open"),
            "footer should contain 'Enter open'; got: {last_line}",
        );
    }

    // --- Show view rendering tests ---

    #[test]
    fn show_tui_renders_header_with_session_info() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        let output = render_app(&mut app, 120, 20);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("Esc back") && last_line.contains("q quit"),
            "footer should contain back and quit hints; got: {last_line}",
        );
    }

    #[test]
    fn show_tui_renders_exchange_rows() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        let output = render_app(&mut app, 140, 20);
        let subagent_line = output.lines().find(|l| l.contains("subagent")).unwrap();
        assert!(
            subagent_line.contains('—'),
            "subagent row (None cost) should contain em-dash; got: {subagent_line}",
        );
    }

    #[test]
    fn show_tui_renders_tool_use_count_in_content() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        let output = render_app(&mut app, 160, 20);
        assert!(
            output.contains("+3 tool uses"),
            "should contain tool use count; got:\n{output}",
        );
    }

    #[test]
    fn show_tui_dims_odd_exchange_rows() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
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

    // --- Tab switching tests ---

    #[test]
    fn app_starts_on_specified_tab() {
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        assert_eq!(app.tab, Tab::Inputs);
    }

    #[test]
    fn key_1_switches_to_sessions() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
        assert_eq!(app.tab, Tab::Sessions);
    }

    #[test]
    fn key_2_switches_to_inputs_loading() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('2')));
        assert_eq!(app.tab, Tab::Inputs);
        assert!(matches!(app.inputs, Some(InputsData::Loading)));
        assert_eq!(app.pending_loads.len(), 1);
        assert!(matches!(app.pending_loads[0], LoadRequest::InputsRefresh));
    }

    #[test]
    fn key_2_reuses_cached_inputs() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('2')));
        assert!(matches!(app.inputs, Some(InputsData::Loading)));
        hlr(
            &mut app,
            LoadResult::InputsData {
                result: Ok((fixture_attribution_rows(), fixture_coverage_stats())),
                refresh_fingerprint: None,
            },
        );
        assert!(matches!(app.inputs, Some(InputsData::Loaded(_))));
        app.pending_loads.clear();
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
        handle_key_event(&mut app, key_event(KeyCode::Char('2')));
        assert!(matches!(app.inputs, Some(InputsData::Loaded(_))));
        assert!(app.pending_loads.is_empty());
    }

    #[test]
    fn tab_switch_from_show_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Char('2')));
        assert_eq!(app.tab, Tab::Inputs);
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert!(handle_key_event(&mut app, key_event(KeyCode::Char('q'))));
    }

    #[test]
    fn inputs_key_quit_on_esc() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert!(handle_key_event(&mut app, key_event(KeyCode::Esc)));
    }

    #[test]
    fn inputs_key_down_advances_selection() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert_eq!(inputs_table_selected(&app), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Down));
        assert_eq!(inputs_table_selected(&app), Some(1));
    }

    #[test]
    fn inputs_key_up_retreats_selection() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        handle_key_event(&mut app, key_event(KeyCode::Down));
        handle_key_event(&mut app, key_event(KeyCode::Up));
        assert_eq!(inputs_table_selected(&app), Some(0));
    }

    #[test]
    fn inputs_key_unknown_is_noop() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('x')));
        assert!(!quit);
        assert_eq!(inputs_table_selected(&app), Some(0));
    }

    // --- Tab rendering tests ---

    #[test]
    fn tab_header_shows_sessions_active() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("3 sessions"),
            "header should show session count; got: {first_line}",
        );
    }

    #[test]
    fn inputs_tab_renders_table_columns() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
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
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("1/2 tabs"),
            "list footer should contain '1/2 tabs'; got: {last_line}",
        );
    }

    // --- Pricing overlay key routing tests ---

    #[test]
    fn p_opens_pricing_overlay() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(matches!(app.overlay, Some(Overlay::Pricing)));
    }

    #[test]
    fn p_opens_overlay_from_show_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(matches!(app.overlay, Some(Overlay::Pricing)));
    }

    #[test]
    fn p_opens_overlay_from_inputs_tab() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(matches!(app.overlay, Some(Overlay::Pricing)));
    }

    #[test]
    fn esc_closes_pricing_overlay() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(app.overlay.is_some());
        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert!(app.overlay.is_none());
    }

    #[test]
    fn p_toggles_pricing_overlay_closed() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(app.overlay.is_some());
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(app.overlay.is_none());
    }

    #[test]
    fn q_closes_pricing_overlay() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(app.overlay.is_some());
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('q')));
        assert!(app.overlay.is_none());
        assert!(!quit);
    }

    #[test]
    fn overlay_swallows_unhandled_keys() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        let original_tab = app.tab;
        let original_selected = app.list_state.selected();
        for code in [
            KeyCode::Down,
            KeyCode::Char('1'),
            KeyCode::Char('2'),
            KeyCode::Enter,
        ] {
            handle_key_event(&mut app, key_event(code));
        }
        assert!(app.overlay.is_some());
        assert_eq!(app.tab, original_tab);
        assert_eq!(app.list_state.selected(), original_selected);
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn ctrl_c_quits_even_with_overlay_open() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(app.overlay.is_some());
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(handle_key_event(&mut app, key));
    }

    #[test]
    fn overlay_close_preserves_tab_and_view_state() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        assert!(app.overlay.is_some());
        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert!(app.overlay.is_none());
        assert!(matches!(app.view, View::Show { .. }));
        assert_eq!(app.tab, Tab::Sessions);
    }

    // --- Pricing overlay rendering tests ---

    #[test]
    fn pricing_overlay_renders_model_rates() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.overlay = Some(Overlay::Pricing);
        let output = render_app(&mut app, 100, 20);
        assert!(
            output.contains("claude-haiku-4-5"),
            "overlay should contain model name; got:\n{output}",
        );
        assert!(
            output.contains("$0.80"),
            "overlay should contain rate value; got:\n{output}",
        );
    }

    #[test]
    fn pricing_overlay_renders_cache_staleness() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.overlay = Some(Overlay::Pricing);
        let output = render_app(&mut app, 100, 20);
        assert!(
            output.contains("catalog:"),
            "overlay footer should contain cache staleness; got:\n{output}",
        );
    }

    #[test]
    fn pricing_overlay_renders_block_border() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.overlay = Some(Overlay::Pricing);
        let output = render_app(&mut app, 100, 20);
        assert!(
            output.contains("Pricing ($/MTok)"),
            "overlay should contain block title; got:\n{output}",
        );
    }

    #[test]
    fn pricing_overlay_does_not_render_when_closed() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 100, 20);
        assert!(
            !output.contains("Pricing ($/MTok)"),
            "overlay should not appear when closed; got:\n{output}",
        );
    }

    // --- Pricing refresh tests ---

    #[test]
    fn r_in_overlay_pushes_pricing_refresh_request() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('p')));
        handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert!(app.pricing_refresh_in_flight);
        assert_eq!(app.pending_loads.len(), 1);
        assert!(matches!(app.pending_loads[0], LoadRequest::RefreshPricing));
    }

    #[test]
    fn r_in_overlay_while_refresh_in_flight_is_noop() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.overlay = Some(Overlay::Pricing);
        app.pricing_refresh_in_flight = true;
        handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert!(app.pending_loads.is_empty());
    }

    #[test]
    fn r_without_overlay_does_not_trigger_pricing_refresh() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert!(
            !app.pending_loads
                .iter()
                .any(|r| matches!(r, LoadRequest::RefreshPricing))
        );
    }

    #[test]
    fn handle_load_result_pricing_success_updates_data() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.pricing_refresh_in_flight = true;
        let new_pricing = PricingData {
            entries: vec![(
                "claude-test".to_string(),
                fixture_pricing_data().entries[0].1,
            )],
            cache_info: CacheInfo {
                path: None,
                exists: true,
                last_modified: Some(SystemTime::now()),
                size: 2048,
                entry_count: Some(1),
            },
        };
        hlr(
            &mut app,
            LoadResult::Pricing {
                result: Ok((Arc::new(PricingCatalog::default()), new_pricing)),
            },
        );
        assert_eq!(app.pricing.entries.len(), 1);
        assert_eq!(app.pricing.entries[0].0, "claude-test");
        assert!(!app.pricing_refresh_in_flight);
    }

    #[test]
    fn handle_load_result_pricing_error_keeps_old_data() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.pricing_refresh_in_flight = true;
        let original_len = app.pricing.entries.len();
        hlr(
            &mut app,
            LoadResult::Pricing {
                result: Err(anyhow::anyhow!("network error")),
            },
        );
        assert_eq!(app.pricing.entries.len(), original_len);
        assert!(!app.pricing_refresh_in_flight);
    }

    #[test]
    fn pricing_overlay_footer_shows_refresh_hint() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.overlay = Some(Overlay::Pricing);
        let output = render_app(&mut app, 100, 20);
        assert!(
            output.contains("r refresh"),
            "overlay footer should contain 'r refresh'; got:\n{output}",
        );
    }

    #[test]
    fn pricing_overlay_footer_shows_refreshing_state() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.overlay = Some(Overlay::Pricing);
        app.pricing_refresh_in_flight = true;
        let output = render_app(&mut app, 100, 20);
        assert!(
            output.contains("refreshing"),
            "overlay footer should contain 'refreshing' during refresh; got:\n{output}",
        );
    }

    // --- Error state tests ---

    #[test]
    fn inputs_error_retry_on_enter() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Error("test error".to_string()));
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(matches!(app.inputs, Some(InputsData::Loading)));
        assert!(matches!(app.pending_loads[0], LoadRequest::InputsRefresh));
    }

    #[test]
    fn inputs_error_retry_on_r() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Error("test error".to_string()));
        handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert!(matches!(app.inputs, Some(InputsData::Loading)));
        assert!(matches!(app.pending_loads[0], LoadRequest::InputsRefresh));
    }

    #[test]
    fn inputs_error_quit_on_q() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Error("test error".to_string()));
        assert!(handle_key_event(&mut app, key_event(KeyCode::Char('q'))));
    }

    #[test]
    fn show_error_esc_returns_to_list() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
            message: "load failed".to_string(),
        };
        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn show_error_backspace_returns_to_list() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
            message: "load failed".to_string(),
        };
        handle_key_event(&mut app, key_event(KeyCode::Backspace));
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn show_error_enter_retries() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
            message: "load failed".to_string(),
        };
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(matches!(app.view, View::ShowLoading { .. }));
        assert!(matches!(
            app.pending_loads[0],
            LoadRequest::ShowDetail { .. }
        ));
    }

    #[test]
    fn show_error_retry_on_r() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
            message: "load failed".to_string(),
        };
        handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert!(matches!(app.view, View::ShowLoading { .. }));
        assert!(matches!(
            app.pending_loads[0],
            LoadRequest::ShowDetail { .. }
        ));
    }

    #[test]
    fn inputs_error_renders_error_message() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Error("test error".to_string()));
        let output = render_app(&mut app, 80, 10);
        assert!(
            output.contains("Error:") && output.contains("test error"),
            "should render error message; got:\n{output}",
        );
        assert!(
            output.contains("retry"),
            "should render retry hint; got:\n{output}",
        );
    }

    #[test]
    fn show_error_renders_error_message() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "\"First session\" (alpha)".to_string(),
            message: "connection failed".to_string(),
        };
        let output = render_app(&mut app, 80, 10);
        assert!(
            output.contains("Error:") && output.contains("connection failed"),
            "should render error message; got:\n{output}",
        );
        assert!(
            output.contains("retry"),
            "should render retry hint; got:\n{output}",
        );
    }

    #[test]
    fn inputs_error_header_shows_error_indicator() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Error("test error".to_string()));
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("cclens — !"),
            "header should show '!' indicator for error state; got: {first_line}",
        );
    }

    #[test]
    fn show_error_header_shows_session_context() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "\"First session\" (alpha)".to_string(),
            message: "connection failed".to_string(),
        };
        let output = render_app(&mut app, 80, 10);
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("First session") && first_line.contains("alpha"),
            "header should show session context; got: {first_line}",
        );
    }

    // --- Context versioning (generation) tests ---

    #[test]
    fn stale_generation_result_is_discarded() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
        };
        app.ctx_generation = 1;
        handle_load_result(
            &mut app,
            StampedResult {
                generation: 0,
                result: LoadResult::ShowDetail {
                    session_id: "aaa".to_string(),
                    header_label: "test".to_string(),
                    result: Ok(fixture_prepared_exchanges()),
                    refresh_fingerprint: None,
                    is_refresh: false,
                },
            },
        );
        assert!(
            matches!(app.view, View::ShowLoading { .. }),
            "a result stamped below the current generation must not apply",
        );
    }

    #[test]
    fn stale_generation_result_still_clears_refresh_in_flight() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx_generation = 1;
        app.refresh_in_flight = true;
        handle_load_result(
            &mut app,
            StampedResult {
                generation: 0,
                result: LoadResult::NoChange,
            },
        );
        assert!(
            !app.refresh_in_flight,
            "a discarded stale result must still clear backpressure or the timer never refreshes again",
        );
    }

    #[test]
    fn current_generation_result_applies() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
        };
        let generation = app.ctx_generation;
        handle_load_result(
            &mut app,
            StampedResult {
                generation,
                result: LoadResult::ShowDetail {
                    session_id: "aaa".to_string(),
                    header_label: "test".to_string(),
                    result: Ok(fixture_prepared_exchanges()),
                    refresh_fingerprint: None,
                    is_refresh: false,
                },
            },
        );
        assert!(matches!(app.view, View::Show { .. }));
    }

    // --- apply_pricing / current_scope tests ---

    #[test]
    fn apply_pricing_success_bumps_generation_and_swaps_catalog() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let original_generation = app.ctx_generation;
        let new_catalog = Arc::new(PricingCatalog::default());
        apply_pricing(
            &mut app,
            Ok((Arc::clone(&new_catalog), fixture_pricing_data())),
        );
        assert_eq!(app.ctx_generation, original_generation + 1);
        assert!(Arc::ptr_eq(&app.ctx.catalog, &new_catalog));
    }

    #[test]
    fn apply_pricing_success_pushes_forced_refresh_of_current_scope() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        apply_pricing(
            &mut app,
            Ok((Arc::new(PricingCatalog::default()), fixture_pricing_data())),
        );
        assert_eq!(app.pending_loads.len(), 1);
        assert!(matches!(
            app.pending_loads[0],
            LoadRequest::Refresh {
                scope: RefreshScope::Sessions,
                guard: None,
            }
        ));
    }

    #[test]
    fn apply_pricing_failure_does_not_bump_generation() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let original_generation = app.ctx_generation;
        apply_pricing(&mut app, Err(anyhow::anyhow!("network error")));
        assert_eq!(app.ctx_generation, original_generation);
        assert!(app.pending_loads.is_empty());
        assert!(!app.pricing_refresh_in_flight);
    }

    #[test]
    fn apply_pricing_failure_sets_status_immediately() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        apply_pricing(&mut app, Err(anyhow::anyhow!("network error")));
        assert!(
            app.status.is_some(),
            "a user-triggered pricing failure must set status with no threshold",
        );
    }

    #[test]
    fn apply_show_detail_user_triggered_failure_sets_status() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
        };
        apply_show_detail(
            &mut app,
            "aaa".to_string(),
            "test".to_string(),
            Err(anyhow::anyhow!("load failed")),
            None,
            false,
        );
        assert!(matches!(app.view, View::ShowError { .. }));
        assert!(
            app.status.is_some(),
            "a user-triggered show-detail failure must set status",
        );
    }

    #[test]
    fn apply_inputs_data_user_triggered_failure_sets_status() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loading);
        apply_inputs_data(&mut app, Err(anyhow::anyhow!("load failed")), None);
        assert!(matches!(app.inputs, Some(InputsData::Error(_))));
        assert!(
            app.status.is_some(),
            "a user-triggered inputs-load failure must set status",
        );
    }

    #[test]
    fn no_change_does_not_clear_an_existing_status() {
        // Regression test: `NoChange` fires on the overwhelmingly
        // common idle refresh tick. A user-triggered error (here, a
        // failed pricing refresh — the status footer is its only
        // display surface) must survive one of those ticks landing
        // moments later, or it vanishes before it can be read.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        apply_pricing(&mut app, Err(anyhow::anyhow!("network error")));
        assert!(app.status.is_some());
        apply_refresh_failed(&mut app, "boom");
        assert_eq!(app.consecutive_refresh_failures, 1);

        let generation = app.ctx_generation;
        handle_load_result(
            &mut app,
            StampedResult {
                generation,
                result: LoadResult::NoChange,
            },
        );

        assert!(
            app.status.is_some(),
            "an unrelated background NoChange must not clear an existing status",
        );
        assert_eq!(
            app.consecutive_refresh_failures, 0,
            "NoChange still proves the refresh pipeline healthy, resetting the failure counter",
        );
    }

    // --- Status footer tests ---

    #[test]
    fn status_footer_shows_hint_when_status_none() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.status.is_none());
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("Enter open"),
            "footer should show the keybinding hint when status is unset; got: {last_line}",
        );
    }

    #[test]
    fn status_footer_shows_message_when_status_some() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.status = Some(StatusMessage {
            text: "something went wrong".to_string(),
            from_refresh_failure: false,
        });
        let output = render_app(&mut app, 80, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("something went wrong"),
            "footer should show the status message in place of the hint; got: {last_line}",
        );
        assert!(
            !last_line.contains("Enter open"),
            "the keybinding hint must not appear alongside a status message; got: {last_line}",
        );
    }

    #[test]
    fn status_footer_shows_message_on_show_view() {
        // Covers the fourth footer site — `render_show_content`'s
        // formerly-inline footer was the only one not behind a named
        // helper, so it's the one most likely to have been missed.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        app.status = Some(StatusMessage {
            text: "show refresh failed".to_string(),
            from_refresh_failure: false,
        });
        let output = render_app(&mut app, 120, 20);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("show refresh failed"),
            "Show view footer should render status; got: {last_line}",
        );
    }

    #[test]
    fn status_footer_shows_message_on_loaded_inputs_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        app.status = Some(StatusMessage {
            text: "inputs refresh failed".to_string(),
            from_refresh_failure: false,
        });
        let output = render_app(&mut app, 120, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("inputs refresh failed"),
            "Inputs view footer should render status; got: {last_line}",
        );
    }

    // --- Consecutive refresh-failure threshold tests ---

    #[test]
    fn two_consecutive_refresh_failures_leave_status_none() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        apply_refresh_failed(&mut app, "boom");
        apply_refresh_failed(&mut app, "boom");
        assert_eq!(app.consecutive_refresh_failures, 2);
        assert!(app.status.is_none(), "two failures should stay silent");
    }

    #[test]
    fn third_consecutive_refresh_failure_sets_status() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        apply_refresh_failed(&mut app, "boom");
        apply_refresh_failed(&mut app, "boom");
        apply_refresh_failed(&mut app, "boom");
        assert_eq!(app.consecutive_refresh_failures, 3);
        assert!(
            app.status.is_some(),
            "the third consecutive failure must break silence",
        );
    }

    #[test]
    fn successful_load_after_failures_clears_status_and_resets_counter() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        apply_refresh_failed(&mut app, "boom");
        apply_refresh_failed(&mut app, "boom");
        apply_refresh_failed(&mut app, "boom");
        assert!(app.status.is_some());

        apply_sessions_data(
            &mut app,
            Ok(fixture_sessions()),
            Some(RefreshFingerprint::default()),
        );

        assert!(app.status.is_none(), "a successful load must clear status");
        assert_eq!(
            app.consecutive_refresh_failures, 0,
            "a successful load must reset the failure counter",
        );
    }

    #[test]
    fn no_change_clears_a_refresh_failure_status() {
        // Regression test for a manually-observed bug: recovering a
        // deleted `--projects-dir` via `mv away && mv back` restores
        // the file with its *original* mtime, so the guarded refresh
        // that follows can only ever see an unchanged fingerprint
        // (`NoChange`) — never a changed-data reload. A status that
        // `apply_refresh_failed` raised must clear on that signal, or
        // a fully-recovered directory leaves the "may be stale"
        // warning stuck forever.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        apply_refresh_failed(&mut app, "boom");
        apply_refresh_failed(&mut app, "boom");
        apply_refresh_failed(&mut app, "boom");
        assert!(app.status.is_some());

        let generation = app.ctx_generation;
        handle_load_result(
            &mut app,
            StampedResult {
                generation,
                result: LoadResult::NoChange,
            },
        );

        assert!(
            app.status.is_none(),
            "NoChange must clear a status that apply_refresh_failed itself raised",
        );
        assert_eq!(app.consecutive_refresh_failures, 0);
    }

    #[test]
    fn current_scope_maps_list_view_to_sessions() {
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(matches!(current_scope(&app), Some(RefreshScope::Sessions)));
    }

    #[test]
    fn current_scope_maps_show_view_to_show_with_session_id() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        assert!(matches!(
            current_scope(&app),
            Some(RefreshScope::Show { session_id }) if session_id == "aaa"
        ));
    }

    #[test]
    fn current_scope_maps_loaded_inputs_tab_to_inputs() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert!(matches!(current_scope(&app), Some(RefreshScope::Inputs)));
    }

    #[test]
    fn current_scope_none_for_loading_inputs_tab() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loading);
        assert!(current_scope(&app).is_none());
    }

    // --- handle_load_result tests ---

    #[test]
    fn load_result_show_detail_success() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
        };
        hlr(
            &mut app,
            LoadResult::ShowDetail {
                session_id: "aaa".to_string(),
                header_label: "test".to_string(),
                result: Ok(fixture_prepared_exchanges()),
                refresh_fingerprint: None,
                is_refresh: false,
            },
        );
        assert!(matches!(app.view, View::Show { .. }));
    }

    #[test]
    fn load_result_show_detail_failure() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
        };
        hlr(
            &mut app,
            LoadResult::ShowDetail {
                session_id: "aaa".to_string(),
                header_label: "test".to_string(),
                result: Err(anyhow::anyhow!("load failed")),
                refresh_fingerprint: None,
                is_refresh: false,
            },
        );
        assert!(matches!(app.view, View::ShowError { .. }));
    }

    #[test]
    fn load_result_show_detail_stale_is_discarded() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        hlr(
            &mut app,
            LoadResult::ShowDetail {
                session_id: "aaa".to_string(),
                header_label: "test".to_string(),
                result: Ok(fixture_prepared_exchanges()),
                refresh_fingerprint: None,
                is_refresh: false,
            },
        );
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn load_result_inputs_success() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loading);
        hlr(
            &mut app,
            LoadResult::InputsData {
                result: Ok((fixture_attribution_rows(), fixture_coverage_stats())),
                refresh_fingerprint: None,
            },
        );
        assert!(matches!(app.inputs, Some(InputsData::Loaded(_))));
    }

    #[test]
    fn load_result_inputs_failure() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loading);
        hlr(
            &mut app,
            LoadResult::InputsData {
                result: Err(anyhow::anyhow!("load failed")),
                refresh_fingerprint: None,
            },
        );
        assert!(matches!(app.inputs, Some(InputsData::Error(_))));
    }

    #[test]
    /// A `refresh_fingerprint: None` result landing while the inputs
    /// table is already `Loaded` is a *forced* reload (catalog swap),
    /// not a stray duplicate — it must apply, not be discarded. This
    /// is what makes a pricing refresh recompute costs on a visible
    /// Inputs tab; see `apply_inputs_data`'s `is_refresh_triggered`.
    fn forced_inputs_refresh_applies_while_loaded() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        hlr(
            &mut app,
            LoadResult::InputsData {
                result: Ok((vec![], fixture_coverage_stats())),
                refresh_fingerprint: None,
            },
        );
        assert!(matches!(app.inputs, Some(InputsData::Loaded(_))));
        assert_eq!(
            inputs_table_selected(&app),
            None,
            "empty rows clear selection"
        );
    }

    // --- Loading state rendering tests ---

    #[test]
    fn show_loading_renders_loading_message() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "\"First session\" (alpha)".to_string(),
        };
        let output = render_app(&mut app, 80, 10);
        assert!(
            output.contains("Loading session..."),
            "should render loading message; got:\n{output}",
        );
    }

    #[test]
    fn inputs_loading_renders_loading_message() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loading);
        let output = render_app(&mut app, 80, 10);
        assert!(
            output.contains("Loading inputs..."),
            "should render loading message; got:\n{output}",
        );
    }

    // --- Refresh fingerprint tests ---

    #[test]
    fn refresh_fingerprint_eq_same_entries() {
        let t = SystemTime::UNIX_EPOCH;
        let mut a = RefreshFingerprint::default();
        a.entries.insert(PathBuf::from("/a.jsonl"), (100, t));
        a.entries.insert(PathBuf::from("/b.jsonl"), (200, t));
        let mut b = RefreshFingerprint::default();
        b.entries.insert(PathBuf::from("/b.jsonl"), (200, t));
        b.entries.insert(PathBuf::from("/a.jsonl"), (100, t));
        assert_eq!(a, b);
    }

    #[test]
    fn refresh_fingerprint_ne_different_size() {
        let t = SystemTime::UNIX_EPOCH;
        let mut a = RefreshFingerprint::default();
        a.entries.insert(PathBuf::from("/a.jsonl"), (100, t));
        let mut b = RefreshFingerprint::default();
        b.entries.insert(PathBuf::from("/a.jsonl"), (200, t));
        assert_ne!(a, b);
    }

    #[test]
    fn refresh_fingerprint_ne_new_path() {
        let t = SystemTime::UNIX_EPOCH;
        let mut a = RefreshFingerprint::default();
        a.entries.insert(PathBuf::from("/a.jsonl"), (100, t));
        let mut b = a.clone();
        b.entries.insert(PathBuf::from("/c.jsonl"), (300, t));
        assert_ne!(a, b);
    }

    // --- selected_session_id tests ---

    #[test]
    fn selected_session_id_returns_id() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.list_state.select(Some(1));
        assert_eq!(app.selected_session_id(), Some("bbb"));
    }

    #[test]
    fn selected_session_id_none_when_empty() {
        let app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        assert_eq!(app.selected_session_id(), None);
    }

    // --- apply_sessions_refresh tests ---

    #[test]
    fn apply_sessions_refresh_preserves_selection_by_id() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.list_state.select(Some(1)); // "bbb"

        let mut reordered = fixture_sessions();
        reordered.reverse(); // ccc, bbb, aaa
        app.apply_sessions_refresh(reordered, RefreshFingerprint::default());

        assert_eq!(app.list_state.selected(), Some(1)); // "bbb" is now at index 1
    }

    #[test]
    fn apply_sessions_refresh_falls_back_on_removed_session() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.list_state.select(Some(2)); // "ccc"

        let mut shorter = fixture_sessions();
        shorter.retain(|s| s.id != "ccc");
        app.apply_sessions_refresh(shorter, RefreshFingerprint::default());

        // Old index 2 is clamped to new_len - 1 = 1
        assert_eq!(app.list_state.selected(), Some(1));
    }

    #[test]
    fn apply_sessions_refresh_updates_totals() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let mut updated = fixture_sessions();
        updated[0].total_billable = 5000;
        updated[0].cost_breakdown = Some(CostBreakdown {
            output: 0.05,
            ..CostBreakdown::default()
        });
        app.apply_sessions_refresh(updated, RefreshFingerprint::default());

        assert_eq!(app.total_tokens, 5000 + 2500 + 3000);
        let expected = 0.05 + 0.02 + 0.03;
        assert!(
            (app.total_cost.unwrap() - expected).abs() < 1e-10,
            "expected {expected}, got {:?}",
            app.total_cost,
        );
    }

    // --- handle_load_result refresh tests ---

    #[test]
    fn handle_sessions_data_applies_in_list_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.refresh_in_flight = true;
        let mut updated = fixture_sessions();
        updated[0].total_billable = 9999;
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Ok(updated),
                fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        assert_eq!(app.sessions[0].total_billable, 9999);
        assert!(!app.refresh_in_flight);
    }

    #[test]
    fn handle_sessions_data_discarded_in_show_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        app.refresh_in_flight = true;
        let original_count = app.sessions.len();
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Ok(vec![]),
                fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        assert_eq!(app.sessions.len(), original_count);
        assert!(!app.refresh_in_flight);
    }

    #[test]
    fn handle_no_change_clears_in_flight() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.refresh_in_flight = true;
        hlr(&mut app, LoadResult::NoChange);
        assert!(!app.refresh_in_flight);
    }

    // --- build_refresh_request tests ---

    #[test]
    fn build_refresh_request_returns_sessions_in_list() {
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let request = build_refresh_request(&app);
        assert!(matches!(
            request,
            Some(LoadRequest::Refresh {
                scope: RefreshScope::Sessions,
                guard: Some(_),
            })
        ));
    }

    #[test]
    fn build_refresh_request_returns_none_in_show_loading() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
        };
        assert!(build_refresh_request(&app).is_none());
    }

    #[test]
    fn build_refresh_request_returns_none_in_show_error() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
            message: "fail".to_string(),
        };
        assert!(build_refresh_request(&app).is_none());
    }

    // --- Phase 2: Show view refresh tests ---

    #[test]
    fn build_refresh_request_returns_show_in_show_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        let request = build_refresh_request(&app);
        assert!(matches!(
            request,
            Some(LoadRequest::Refresh {
                scope: RefreshScope::Show { session_id },
                guard: Some(_),
            }) if session_id == "aaa"
        ));
    }

    #[test]
    fn refresh_show_auto_scrolls_when_at_end() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        // Select last row (index 2 — 3 rows from fixture_prepared_exchanges)
        if let View::Show { table_state, .. } = &mut app.view {
            table_state.select(Some(2));
        }
        app.refresh_in_flight = true;

        let mut new_prepared = fixture_prepared_exchanges();
        new_prepared.push(PreparedExchange {
            origin: TurnOrigin::Parent,
            rows: vec![
                PreparedRow {
                    timestamp: Some("2026-04-01T10:03:00Z".parse::<DateTime<Utc>>().unwrap()),
                    role: PreparedRowRole::User,
                    tokens: Some(100),
                    cost: None,
                    cumulative_tokens: 1100,
                    cumulative_cost: None,
                    content: "New exchange".to_string(),
                    tool_use_count: 0,
                },
                PreparedRow {
                    timestamp: Some("2026-04-01T10:04:00Z".parse::<DateTime<Utc>>().unwrap()),
                    role: PreparedRowRole::Assistant,
                    tokens: Some(100),
                    cost: None,
                    cumulative_tokens: 1200,
                    cumulative_cost: None,
                    content: "Response".to_string(),
                    tool_use_count: 0,
                },
            ],
        });
        let total_rows = new_prepared.iter().map(|e| e.rows.len()).sum::<usize>();

        hlr(
            &mut app,
            LoadResult::ShowDetail {
                session_id: "aaa".to_string(),
                header_label: String::new(),
                result: Ok(new_prepared),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
                is_refresh: true,
            },
        );

        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(
                table_state.selected(),
                Some(total_rows - 1),
                "should auto-scroll to new last row",
            );
        } else {
            panic!("expected View::Show");
        }
        assert!(!app.refresh_in_flight);
    }

    #[test]
    fn refresh_show_holds_position_when_scrolled_up() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        // Select middle row (index 1)
        if let View::Show { table_state, .. } = &mut app.view {
            table_state.select(Some(1));
        }
        app.refresh_in_flight = true;

        let mut new_prepared = fixture_prepared_exchanges();
        new_prepared.push(PreparedExchange {
            origin: TurnOrigin::Parent,
            rows: vec![PreparedRow {
                timestamp: Some("2026-04-01T10:05:00Z".parse::<DateTime<Utc>>().unwrap()),
                role: PreparedRowRole::User,
                tokens: Some(100),
                cost: None,
                cumulative_tokens: 1100,
                cumulative_cost: None,
                content: "Another one".to_string(),
                tool_use_count: 0,
            }],
        });

        hlr(
            &mut app,
            LoadResult::ShowDetail {
                session_id: "aaa".to_string(),
                header_label: String::new(),
                result: Ok(new_prepared),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
                is_refresh: true,
            },
        );

        if let View::Show { table_state, .. } = &app.view {
            assert_eq!(
                table_state.selected(),
                Some(1),
                "should hold position when scrolled up",
            );
        } else {
            panic!("expected View::Show");
        }
        assert!(!app.refresh_in_flight);
    }

    #[test]
    fn refresh_show_discarded_when_session_mismatch() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app); // viewing session "aaa"
        app.refresh_in_flight = true;

        hlr(
            &mut app,
            LoadResult::ShowDetail {
                session_id: "bbb".to_string(),
                header_label: String::new(),
                result: Ok(vec![]),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
                is_refresh: true,
            },
        );

        // View should be unchanged — still showing "aaa"'s data
        if let View::Show {
            session_id,
            prepared,
            ..
        } = &app.view
        {
            assert_eq!(session_id, "aaa");
            assert!(!prepared.is_empty(), "data should be unchanged");
        } else {
            panic!("expected View::Show");
        }
        assert!(!app.refresh_in_flight);
    }

    // --- Phase 3: Inputs view refresh tests ---

    #[test]
    fn build_refresh_request_returns_inputs_when_loaded() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        let request = build_refresh_request(&app);
        assert!(matches!(
            request,
            Some(LoadRequest::Refresh {
                scope: RefreshScope::Inputs,
                guard: Some(_),
            })
        ));
    }

    #[test]
    fn build_refresh_request_returns_none_inputs_loading() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loading);
        assert!(build_refresh_request(&app).is_none());
    }

    #[test]
    fn refresh_inputs_preserves_selection_by_path() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        // Select the second row (skill at index 1)
        if let Some(InputsData::Loaded(state)) = &mut app.inputs {
            state.table_state.select(Some(1));
        }
        app.refresh_in_flight = true;

        let mut reordered = fixture_attribution_rows();
        reordered.reverse(); // project, skill, global → skill is now at index 1
        let new_coverage = fixture_coverage_stats();

        hlr(
            &mut app,
            LoadResult::InputsData {
                result: Ok((reordered, new_coverage)),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
            },
        );

        // skill path is "/home/user/.claude/skills/foo/SKILL.md" — find its new index
        if let Some(InputsData::Loaded(state)) = &app.inputs {
            let selected = state.table_state.selected();
            let selected_path = selected.and_then(|idx| state.rows.get(idx));
            assert!(
                selected_path.is_some_and(|r| r.file.path.to_string_lossy().contains("skills")),
                "selection should track by file path to the skill row",
            );
        } else {
            panic!("expected InputsData::Loaded");
        }
        assert!(!app.refresh_in_flight);
    }

    #[test]
    fn refresh_inputs_falls_back_on_removed_row() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        // Select the last row (index 2)
        if let Some(InputsData::Loaded(state)) = &mut app.inputs {
            state.table_state.select(Some(2));
        }
        app.refresh_in_flight = true;

        // Refresh with only one row — the previously selected row is gone
        let single_row = vec![fixture_attribution_rows().remove(0)];
        hlr(
            &mut app,
            LoadResult::InputsData {
                result: Ok((single_row, fixture_coverage_stats())),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
            },
        );

        // Old index 2 should fall back to min(2, 0) = 0
        assert_eq!(inputs_table_selected(&app), Some(0));
        assert!(!app.refresh_in_flight);
    }
}
