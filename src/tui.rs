//! Interactive TUI rendering for `cclens list`, `cclens inputs`, and
//! `cclens agents` with tabbed navigation, session drill-down, and
//! pricing, filter, and comparison overlays.
//!
//! Cell data comes from `views` (shared with `rendering`); this
//! module handles ratatui widget construction, styling, layout
//! constraints, and the event loop. Data loading is `loading`'s
//! responsibility — this module consumes `loading::DataContext` and
//! its load functions directly, with no dependency-inversion layer.
//!
//! Public API:
//! - `Tab` — `Sessions` | `Inputs` | `Agents` — the active tab.
//! - `run_tui(DataContext, Vec<Session>, RefreshFingerprint,
//!   PricingData, Tab) -> anyhow::Result<()>` — fullscreen TUI with
//!   tab switching (1/2/3 keys), scrollable tables, session drill-down
//!   (Enter/Esc within Sessions tab), attribution table with coverage
//!   footer (Inputs tab), agent rows with a pinning-slice line
//!   (Agents tab), pricing overlay (`p` toggles, `r` refreshes,
//!   Esc/q closes), and timer-driven auto-refresh (3s interval,
//!   fingerprint-gated).
//!
//! Agents-tab bindings beyond navigation: `c` opens the comparison
//! modal for the selected row's agent type, re-presenting that agent's
//! rows over a repricing panel whose target model `↑`/`↓` switch; `a`
//! widens `ctx.query.pinning` to every kind and back, committing
//! through `invalidate_data` so the generation bump and the slot
//! clearing happen at one site.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use crossterm::event::EventStream;
use futures_util::StreamExt;
use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Row, Table, TableState};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc;
use tokio::time::{self, MissedTickBehavior};

use crate::agents::{AgentRow, Pinning, PinningFilter, repriced_delta};
use crate::aggregation::{PreparedExchange, PreparedRow};
use crate::attribution::{AttributionRow, CoverageStats};
use crate::domain::Session;
use crate::filter::{
    FilterComponent, QueryScope, parse_filter_datetime, parse_min_cost, render_filter_datetime,
};
use crate::formatting::{coverage_line, format_cost_opt, format_tokens, tiers_differ};
use crate::inventory::InventoryConfig;
use crate::loading::{self, DataContext, PricingData, Query, RefreshFingerprint};
use crate::pricing::{CacheInfo, PricingCatalog};
use crate::views::{
    agents_cells, inputs_cells, pricing_view_rows, repriced_cells, session_cells, session_totals,
    show_row_cells,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Sessions,
    Inputs,
    Agents,
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

/// The Agents tab's rows and the selection that indexes them,
/// constructed together so a selection cannot outlive its rows.
/// Mirrors `InputsState`.
struct AgentsState {
    rows: Vec<AgentRow>,
    table_state: TableState,
}

impl AgentsState {
    fn new(rows: Vec<AgentRow>) -> Self {
        let mut table_state = TableState::default();
        if !rows.is_empty() {
            table_state.select_first();
        }
        Self { rows, table_state }
    }

    /// The key `apply_agents_data` preserves a selection across a
    /// refresh by. A path is not available here — a row is an
    /// accumulation, not a file — so the row key stands in for one.
    fn selected_key(&self) -> Option<(String, Option<String>, Option<String>, Pinning)> {
        let row = self.rows.get(self.table_state.selected()?)?;
        Some((
            row.agent_type.clone(),
            row.model.clone(),
            row.effort.clone(),
            row.pinning.clone(),
        ))
    }
}

enum AgentsData {
    Loading,
    Loaded(AgentsState),
    Error(String),
}

/// The comparison modal's state: one agent type's visible rows, and
/// an index into the catalog's bare `claude-*` keys naming the target
/// being compared against.
///
/// One target at a time with a switcher, rather than several columns:
/// the panel then carries the full label set for the figure on screen,
/// where side-by-side columns would have to drop it for width at
/// exactly the terminal sizes where a stripped number is most likely
/// to be misread.
struct CompareState {
    agent_type: String,
    rows: Vec<AgentRow>,
    /// An index into `targets_from`'s output, not a snapshot of it.
    /// A pricing refresh can land while this overlay is open, and a
    /// snapshot would then name models the live catalog no longer
    /// prices — a mixed-catalog delta with no sign it was one.
    target_idx: usize,
}

/// The models a comparison can target: the catalog's bare `claude-*`
/// keys, which are the ones that match transcript model strings.
fn compare_targets(catalog: &PricingCatalog) -> Vec<String> {
    catalog
        .sorted_entries(false)
        .into_iter()
        .map(|(model, _)| model.to_string())
        .collect()
}

impl CompareState {
    fn target(&self, catalog: &PricingCatalog) -> Option<String> {
        compare_targets(catalog).into_iter().nth(self.target_idx)
    }

    /// Move the target by `delta`, wrapping. Reads the catalog rather
    /// than a stored length for the same reason `target` does.
    fn move_target(&mut self, catalog: &PricingCatalog, forward: bool) {
        let len = compare_targets(catalog).len();
        if len == 0 {
            return;
        }
        self.target_idx = if forward {
            (self.target_idx + 1) % len
        } else {
            self.target_idx.checked_sub(1).unwrap_or(len - 1)
        };
    }
}

/// The Sessions tab's data and everything recomputed alongside it.
/// Mirrors `InputsState`: rows plus the selection state that indexes
/// them, constructed together so a selection cannot outlive its rows.
struct SessionsState {
    sessions: Vec<Session>,
    list_state: TableState,
    total_tokens: u64,
    total_cost: Option<f64>,
}

impl SessionsState {
    fn new(sessions: Vec<Session>) -> Self {
        let (total_tokens, total_cost) =
            session_totals(&sessions).map_or((0, None), |t| (t.total_tokens, t.total_cost));
        let mut list_state = TableState::default();
        if !sessions.is_empty() {
            list_state.select_first();
        }
        Self {
            sessions,
            list_state,
            total_tokens,
            total_cost,
        }
    }

    fn selected_session_id(&self) -> Option<&str> {
        let idx = self.list_state.selected()?;
        self.sessions.get(idx).map(|s| s.id.as_str())
    }

    fn apply_refresh(&mut self, new_sessions: Vec<Session>) {
        let prev_id = self.selected_session_id().map(str::to_owned);
        let prev_idx = self.list_state.selected();

        let (total_tokens, total_cost) =
            session_totals(&new_sessions).map_or((0, None), |t| (t.total_tokens, t.total_cost));
        self.sessions = new_sessions;
        self.total_tokens = total_tokens;
        self.total_cost = total_cost;

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

/// The Sessions tab's data slot. `Loading` means the slot was
/// invalidated or never filled and a load is expected; `Error` carries
/// a failure the user can retry. Mirrors `InputsData` minus its
/// `Option` wrapper — startup always loads sessions.
enum SessionsData {
    Loading,
    Loaded(SessionsState),
    Error(String),
}

enum Overlay {
    Pricing,
    /// Boxed for the same reason `Filter` is: the variant would
    /// otherwise inflate every `Option<Overlay>` by a `Vec` of rows.
    Compare(Box<CompareState>),
    /// Boxed so the variant does not inflate every `Option<Overlay>`
    /// by the editor's six fields — one allocation per `f` press
    /// against a value that is moved only on open and close.
    Filter(Box<FilterEditor>),
}

/// The six editable filter fields, in `Tab` order. `Session` is
/// clear-only — it displays the committed UUID and accepts only the
/// keystroke that empties it, because the constraint is exact
/// equality on a 36-character value that appears nowhere on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterFieldKind {
    Session,
    Project,
    Since,
    Until,
    MinTokens,
    MinCost,
}

impl FilterFieldKind {
    fn label(self) -> &'static str {
        match self {
            FilterFieldKind::Session => "session",
            FilterFieldKind::Project => "project",
            FilterFieldKind::Since => "since",
            FilterFieldKind::Until => "until",
            FilterFieldKind::MinTokens => "min-tokens",
            FilterFieldKind::MinCost => "min-cost",
        }
    }

    /// Whether `Char` input reaches this field. False only for
    /// `Session`, which is clear-only.
    fn accepts_text(self) -> bool {
        match self {
            FilterFieldKind::Session => false,
            FilterFieldKind::Project
            | FilterFieldKind::Since
            | FilterFieldKind::Until
            | FilterFieldKind::MinTokens
            | FilterFieldKind::MinCost => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ParsedFilterValue {
    Text(String),
    Date(DateTime<Utc>),
    Tokens(u64),
    Cost(f64),
}

struct FilterField {
    kind: FilterFieldKind,
    /// The field's text, seeded on open by rendering the committed
    /// `Query` in its shortest round-tripping spelling. Append-only:
    /// `Char` pushes, `Backspace` pops, and no cursor offset exists.
    input: String,
    /// Reparsed on every keystroke. `Err` while the text is non-empty
    /// and unparseable; `Ok(None)` means the field is empty, which is
    /// how a filter is cleared. `Enter` is inert while any field is
    /// `Err`.
    parsed: Result<Option<ParsedFilterValue>, String>,
}

struct FilterEditor {
    fields: [FilterField; 6],
    focused: usize,
}

/// Parse one field's text. Empty (or whitespace-only) is `Ok(None)` —
/// the way a filter is cleared — so fallibility never reaches
/// `loading`: `Query` is only ever built from already-parsed values.
fn parse_filter_field(
    kind: FilterFieldKind,
    input: &str,
) -> Result<Option<ParsedFilterValue>, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match kind {
        FilterFieldKind::Session | FilterFieldKind::Project => {
            Ok(Some(ParsedFilterValue::Text(trimmed.to_string())))
        }
        FilterFieldKind::Since | FilterFieldKind::Until => {
            parse_filter_datetime(trimmed).map(|d| Some(ParsedFilterValue::Date(d)))
        }
        FilterFieldKind::MinTokens => u64::from_str(trimmed)
            .map(|t| Some(ParsedFilterValue::Tokens(t)))
            .map_err(|_| "expected a whole number".to_string()),
        FilterFieldKind::MinCost => {
            parse_min_cost(trimmed).map(|c| Some(ParsedFilterValue::Cost(c)))
        }
    }
}

fn seeded_field(kind: FilterFieldKind, input: String) -> FilterField {
    FilterField {
        kind,
        parsed: parse_filter_field(kind, &input),
        input,
    }
}

impl FilterEditor {
    /// Seed every field from the committed `Query`, rendering each
    /// value in the shortest spelling that reparses to it. A seeded
    /// editor therefore starts entirely `Ok` — the only way to reach
    /// an `Err` field is to type into it.
    fn from_query(query: &Query) -> Self {
        let text = |v: Option<&String>| v.cloned().unwrap_or_default();
        Self {
            fields: [
                seeded_field(
                    FilterFieldKind::Session,
                    text(query.inputs_session_id.as_ref()),
                ),
                seeded_field(
                    FilterFieldKind::Project,
                    text(query.sessions.project_name.as_ref()),
                ),
                seeded_field(
                    FilterFieldKind::Since,
                    query
                        .sessions
                        .since
                        .map(render_filter_datetime)
                        .unwrap_or_default(),
                ),
                seeded_field(
                    FilterFieldKind::Until,
                    query
                        .sessions
                        .until
                        .map(render_filter_datetime)
                        .unwrap_or_default(),
                ),
                seeded_field(
                    FilterFieldKind::MinTokens,
                    query
                        .thresholds
                        .min_tokens
                        .map(|t| t.to_string())
                        .unwrap_or_default(),
                ),
                seeded_field(
                    FilterFieldKind::MinCost,
                    query
                        .thresholds
                        .min_cost
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                ),
            ],
            // Not `Session`: it is the one field that rejects `Char`
            // input, so opening there would silently discard a user's
            // first keystrokes.
            focused: 1,
        }
    }

    /// Project the parsed fields back into a `Query`. Each field is
    /// read with `if let` rather than a `match` on `ParsedFilterValue`
    /// so `wildcard_enum_match_arm` is satisfied without six
    /// unreachable arms per field — a kind/value mismatch is
    /// impossible by construction, and dropping it is the safe
    /// direction if one ever appeared.
    /// Project the parsed fields back onto `base`.
    ///
    /// The editor's six input fields write three of `Query`'s four
    /// fields; `pinning` has no editor field and must survive a
    /// commit. Basing the projection on the committed query makes the
    /// round trip lossless by construction rather than for the fields
    /// someone remembered to add — a field introduced later is
    /// preserved without its author having to find this site.
    fn to_query(&self, base: &Query) -> Query {
        let mut query = base.clone();
        // The editor owns these three: an emptied field must clear the
        // committed value rather than inherit it from `base`.
        query.inputs_session_id = None;
        query.sessions = crate::filter::SessionFilter::default();
        query.thresholds = crate::filter::ThresholdsFilter::default();
        for field in &self.fields {
            let Ok(Some(value)) = &field.parsed else {
                continue;
            };
            match field.kind {
                FilterFieldKind::Session => {
                    if let ParsedFilterValue::Text(t) = value {
                        query.inputs_session_id = Some(t.clone());
                    }
                }
                FilterFieldKind::Project => {
                    if let ParsedFilterValue::Text(t) = value {
                        query.sessions.project_name = Some(t.clone());
                    }
                }
                FilterFieldKind::Since => {
                    if let ParsedFilterValue::Date(d) = value {
                        query.sessions.since = Some(*d);
                    }
                }
                FilterFieldKind::Until => {
                    if let ParsedFilterValue::Date(d) = value {
                        query.sessions.until = Some(*d);
                    }
                }
                FilterFieldKind::MinTokens => {
                    if let ParsedFilterValue::Tokens(t) = value {
                        query.thresholds.min_tokens = Some(*t);
                    }
                }
                FilterFieldKind::MinCost => {
                    if let ParsedFilterValue::Cost(c) = value {
                        query.thresholds.min_cost = Some(*c);
                    }
                }
            }
        }
        query
    }

    fn is_committable(&self) -> bool {
        self.fields.iter().all(|f| f.parsed.is_ok())
    }

    fn focus_next(&mut self) {
        self.focused = (self.focused + 1) % self.fields.len();
    }

    fn focus_prev(&mut self) {
        self.focused = (self.focused + self.fields.len() - 1) % self.fields.len();
    }

    /// Append to the focused field, unless it is clear-only. This is
    /// what keeps `q` from quitting while a project name is typed.
    fn push_char(&mut self, c: char) {
        let field = &mut self.fields[self.focused];
        if !field.kind.accepts_text() {
            return;
        }
        field.input.push(c);
        field.parsed = parse_filter_field(field.kind, &field.input);
    }

    /// Empty the focused field. The one shell chord an append-only
    /// editor can honor literally — without it, correcting the front
    /// of an RFC 3339 timestamp costs 25 backspaces.
    fn clear_field(&mut self) {
        let field = &mut self.fields[self.focused];
        field.input.clear();
        field.parsed = parse_filter_field(field.kind, &field.input);
    }

    /// One key means "remove" throughout the editor: pop a character
    /// from a text field, empty a clear-only one.
    fn backspace(&mut self) {
        let field = &mut self.fields[self.focused];
        if field.kind.accepts_text() {
            field.input.pop();
        } else {
            field.input.clear();
        }
        field.parsed = parse_filter_field(field.kind, &field.input);
    }
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
    Agents,
}

/// The unit of in-flight accounting: at most one load per slot runs at
/// a time. `Show` is keyed by session id rather than by variant —
/// opening B while A's load is still running is two different loads,
/// not a duplicate, and dropping B's dispatch would strand
/// `View::ShowLoading`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum LoadSlot {
    Sessions,
    Show(String),
    Inputs,
    Agents,
    Pricing,
}

/// The loads currently holding one slot, split by whether each can
/// answer for a view that is waiting.
///
/// A guarded load resolves to `NoChange` when the filesystem is
/// unchanged, so it satisfies nobody; an unguarded one always produces
/// a result an applier will consume. That difference is the whole
/// reason both counts exist — see `App::try_reserve`.
#[derive(Clone, Copy, Debug, Default)]
struct Holders {
    guarded: u32,
    forced: u32,
}

impl Holders {
    fn total(self) -> u32 {
        self.guarded + self.forced
    }
}

enum LoadRequest {
    ShowDetail {
        session_id: String,
        header_label: String,
    },
    InputsRefresh,
    AgentsRefresh,
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
    AgentsData {
        result: anyhow::Result<Vec<AgentRow>>,
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
    slot: LoadSlot,
    /// Whether the dispatch was guarded, so `handle_load_result`
    /// releases the same holder `spawn_load` reserved.
    guarded: bool,
    result: LoadResult,
}

/// The severity of a status message, which decides both its styling
/// and its rank in `render_status_footer`'s precedence.
enum StatusKind {
    Error,
    /// No producer yet — the variant exists so that adding one does not
    /// reopen the precedence question in `render_status_footer`. Scoped
    /// to the non-test build because `clippy --all-targets` compiles the
    /// lib twice, and a variant constructed only under `#[cfg(test)]` is
    /// genuinely unreachable in the other build. A plain `expect` would
    /// be unfulfilled in the test build and trip
    /// `unfulfilled_lint_expectation` instead.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "reserved for the first Info-severity producer")
    )]
    Info,
}

/// One message for the status footer. Every producer today is an error
/// path (`apply_show_detail`, `apply_inputs_data`, `apply_sessions_data`,
/// `apply_pricing`, `note_refresh_failure`), so `kind` is uniformly
/// `Error` — but the footer now has a second content source (the
/// derived missing-model hint), which is what makes the severity
/// ordering a question the row has to answer regardless of how many
/// producers exist. See `render_status_footer` for the precedence.
struct StatusMessage {
    text: String,
    kind: StatusKind,
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
    sessions: SessionsData,
    view: View,
    inputs: Option<InputsData>,
    agents: Option<AgentsData>,
    /// The pinning slice the process started with, so the Agents
    /// tab's `a` binding can restore it rather than the library
    /// default. Immutable for the app's lifetime.
    launch_pinning: PinningFilter,
    /// `--compare-model`, when the run supplied one. Seeds the
    /// comparison modal's target so the flag is not silently ignored
    /// on the interactive surface.
    compare_model: Option<String>,
    overlay: Option<Overlay>,
    pricing: PricingData,
    pending_loads: Vec<LoadRequest>,
    refresh_fingerprint: RefreshFingerprint,
    /// Which loads are currently running, per slot. Reserved in
    /// `spawn_load`, released in `handle_load_result`, and cleared
    /// wholesale by `bump_ctx_generation`. Counted rather than
    /// flagged because a guarded and an unguarded load can hold one
    /// slot at once — and the first result to arrive must not free it
    /// out from under the second.
    in_flight: HashMap<LoadSlot, Holders>,
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
        Self {
            launch_pinning: ctx.query.pinning.clone(),
            compare_model: None,
            ctx,
            ctx_generation: 0,
            tab: default_tab,
            sessions: SessionsData::Loaded(SessionsState::new(sessions)),
            view: View::List,
            inputs: None,
            agents: None,
            overlay: None,
            pricing,
            pending_loads: Vec::new(),
            refresh_fingerprint: RefreshFingerprint::default(),
            in_flight: HashMap::new(),
            status: None,
            consecutive_refresh_failures: 0,
        }
    }

    /// Supersede the current data context. Every reservation in the
    /// ledger belongs to a load whose result `handle_load_result` will
    /// now discard, so it protects nothing — and holding it would
    /// block the replacement dispatch, stranding whatever view is
    /// waiting on it.
    fn bump_ctx_generation(&mut self) {
        self.ctx_generation += 1;
        self.in_flight.clear();
    }

    /// Reserve `slot` for one dispatch, reporting whether it may
    /// proceed.
    ///
    /// A dispatch is redundant only when a load already running can
    /// answer for it, and guardedness decides that. A guarded dispatch
    /// is background work whose worst outcome is a tick's delay, so it
    /// yields to any holder. An unguarded dispatch is one somebody is
    /// waiting on, so it yields only to another unguarded load —
    /// yielding to a guarded one would strand a `ShowLoading` or a
    /// `SessionsData::Loading` the moment that load answered
    /// `NoChange`, with no tick able to recover it (`current_scope`
    /// reports `None` for the former and keeps answering `NoChange`
    /// for the latter).
    fn try_reserve(&mut self, slot: LoadSlot, guarded: bool) -> bool {
        let holders = self.in_flight.entry(slot).or_default();
        if guarded {
            if holders.total() > 0 {
                return false;
            }
            holders.guarded += 1;
        } else {
            if holders.forced > 0 {
                return false;
            }
            holders.forced += 1;
        }
        true
    }

    /// Release the holder `guarded` describes, symmetric with
    /// `try_reserve`. The kind matters: freeing a forced holder when a
    /// guarded one finished would let a later forced dispatch be
    /// suppressed by a load that cannot answer for it.
    fn release_slot(&mut self, slot: &LoadSlot, guarded: bool) {
        let Some(holders) = self.in_flight.get_mut(slot) else {
            return;
        };
        if guarded {
            holders.guarded = holders.guarded.saturating_sub(1);
        } else {
            holders.forced = holders.forced.saturating_sub(1);
        }
        if holders.total() == 0 {
            self.in_flight.remove(slot);
        }
    }

    fn slot_busy(&self, slot: &LoadSlot) -> bool {
        self.in_flight.contains_key(slot)
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
}

/// Maps the current tab/view to the `RefreshScope` a timer refresh
/// should target, or `None` when nothing on screen has loaded data to
/// refresh. Its one caller is `build_refresh_request`; forced reloads
/// go through `invalidate_data`, which branches on `tab`/`view`
/// instead.
///
/// The two arms are deliberately asymmetric. `Inputs` is
/// content-sensitive — it requires `InputsData::Loaded` — while
/// The three tabs carry different content-sensitivity policies:
/// Sessions ignores slot contents entirely, while Inputs and Agents
/// each require `Loaded` before a guarded tick targets them.
///
/// `(Tab::Sessions, View::List, _)` ignores slot contents, so the
/// timer keeps issuing guarded Sessions refreshes while that slot sits
/// in `Loading` or `Error`. That is wanted: a guarded tick is the
/// cheap path that lets a failed sessions load self-heal the moment
/// the filesystem changes, and the in-flight ledger absorbs the ticks
/// that would otherwise pile up.
fn current_scope(app: &App) -> Option<RefreshScope> {
    match (&app.tab, &app.view, &app.inputs) {
        (Tab::Sessions, View::List, _) => Some(RefreshScope::Sessions),
        (Tab::Sessions, View::Show { session_id, .. }, _) => Some(RefreshScope::Show {
            session_id: session_id.clone(),
        }),
        (Tab::Inputs, _, Some(InputsData::Loaded(_))) => Some(RefreshScope::Inputs),
        // Requires `Loaded` for the same content-sensitivity reason
        // the Inputs arm does: a guarded refresh into a slot that is
        // still `Loading` has nothing to compare against.
        (Tab::Agents, _, _) if matches!(app.agents, Some(AgentsData::Loaded(_))) => {
            Some(RefreshScope::Agents)
        }
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
    run_tui_with_compare_model(
        ctx,
        sessions,
        initial_fingerprint,
        pricing,
        default_tab,
        None,
    )
    .await
}

/// `run_tui` plus the `--compare-model` target the run supplied, which
/// seeds the Agents tab's comparison modal so the flag is honored on
/// the interactive surface rather than silently ignored.
///
/// # Errors
///
/// Propagates a terminal init/restore failure or an unrecoverable
/// event-loop error.
pub async fn run_tui_with_compare_model(
    ctx: DataContext,
    sessions: Vec<Session>,
    initial_fingerprint: RefreshFingerprint,
    pricing: PricingData,
    default_tab: Tab,
    compare_model: Option<String>,
) -> anyhow::Result<()> {
    let mut terminal = ratatui::try_init()?;
    let result = run_event_loop(
        &mut terminal,
        ctx,
        sessions,
        initial_fingerprint,
        pricing,
        default_tab,
        compare_model,
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
    compare_model: Option<String>,
) -> anyhow::Result<()> {
    let mut app = App::new(ctx, sessions, pricing, default_tab);
    app.compare_model = compare_model;
    app.refresh_fingerprint = initial_fingerprint;
    let mut event_stream = EventStream::new();

    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<StampedResult>();

    let mut refresh_interval = time::interval(std::time::Duration::from_secs(3));
    refresh_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    refresh_interval.tick().await;

    // Only the Sessions tab arrives with data — `run_tui` is handed a
    // session list, and nothing else. A tab launched into directly
    // must dispatch its own first load or it renders an empty slot
    // that no navigation event recovers.
    if default_tab == Tab::Inputs {
        app.inputs = Some(InputsData::Loading);
        app.pending_loads.push(LoadRequest::InputsRefresh);
    }
    if default_tab == Tab::Agents {
        app.agents = Some(AgentsData::Loading);
        app.pending_loads.push(LoadRequest::AgentsRefresh);
    }

    loop {
        drain_pending_loads(&mut app, &result_tx);
        terminal.draw(|frame| render(&mut app, frame))?;

        tokio::select! {
            biased;
            event = event_stream.next() => {
                match event {
                    // Press only: Windows consoles and terminals with
                    // crossterm's keyboard-enhancement flags active
                    // also deliver Release, which would insert every
                    // character typed into a filter field twice.
                    Some(Ok(Event::Key(key)))
                        if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                    {
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
                if let Some(request) = build_refresh_request(&app) {
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
    app: &mut App,
    slot: LoadSlot,
    guard: Option<RefreshFingerprint>,
    tx: &mpsc::UnboundedSender<StampedResult>,
    load: L,
    map: M,
) where
    T: Send + 'static,
    L: FnOnce(&DataContext) -> anyhow::Result<T> + Send + 'static,
    M: FnOnce(anyhow::Result<T>, Option<RefreshFingerprint>) -> LoadResult + Send + 'static,
{
    // Reserve. A guarded dispatch yields to a load already running
    // for this slot — that would be redundant work on a single-
    // threaded loop. An unguarded one never yields; see
    // `App::try_reserve`. Released in `handle_load_result`.
    let guarded = guard.is_some();
    if !app.try_reserve(slot.clone(), guarded) {
        return;
    }
    let ctx = app.ctx.clone();
    let generation = app.ctx_generation;
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
        let _ = tx.send(StampedResult {
            generation,
            slot,
            guarded,
            result,
        });
    });
}

/// Dispatch an agents load. Extracted from `drain_pending_loads`
/// because the guarded and unguarded arms differ only in the guard,
/// and inlining both pushed that function past its line cap.
fn spawn_agents_load(
    app: &mut App,
    guard: Option<RefreshFingerprint>,
    tx: &mpsc::UnboundedSender<StampedResult>,
) {
    spawn_load(
        app,
        LoadSlot::Agents,
        guard,
        tx,
        loading::load_agents,
        |result, fp| LoadResult::AgentsData {
            result,
            refresh_fingerprint: fp,
        },
    );
}

fn drain_pending_loads(app: &mut App, tx: &mpsc::UnboundedSender<StampedResult>) {
    // `mem::take` rather than `drain(..)`: `spawn_load` needs `&mut
    // App`, which a live drain iterator would keep borrowed.
    for request in std::mem::take(&mut app.pending_loads) {
        match request {
            LoadRequest::ShowDetail {
                session_id,
                header_label,
            } => {
                let sid = session_id.clone();
                let slot = LoadSlot::Show(session_id.clone());
                spawn_load(
                    app,
                    slot,
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
                    app,
                    LoadSlot::Inputs,
                    None,
                    tx,
                    loading::load_inputs,
                    |result, _fp| LoadResult::InputsData {
                        result,
                        refresh_fingerprint: None,
                    },
                );
            }
            LoadRequest::AgentsRefresh => spawn_agents_load(app, None, tx),
            LoadRequest::Refresh { scope, guard } => match scope {
                RefreshScope::Sessions => {
                    spawn_load(
                        app,
                        LoadSlot::Sessions,
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
                    let slot = LoadSlot::Show(session_id.clone());
                    spawn_load(
                        app,
                        slot,
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
                        app,
                        LoadSlot::Inputs,
                        guard,
                        tx,
                        loading::load_inputs,
                        |result, fp| LoadResult::InputsData {
                            result,
                            refresh_fingerprint: fp,
                        },
                    );
                }
                RefreshScope::Agents => spawn_agents_load(app, guard, tx),
            },
            LoadRequest::RefreshPricing => {
                spawn_load(
                    app,
                    LoadSlot::Pricing,
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
        // Superseded context: the payload was computed against stale
        // state. This result's reservation was already dropped by
        // `bump_ctx_generation`, so releasing by key here would free a
        // *fresh* holder's slot and license a duplicate dispatch.
        return;
    }
    // Release, symmetric with `spawn_load`'s reserve and reached
    // identically by every applier below.
    app.release_slot(&stamped.slot, stamped.guarded);
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
        LoadResult::AgentsData {
            result,
            refresh_fingerprint,
        } => apply_agents_data(app, result, refresh_fingerprint),
        LoadResult::SessionsData {
            result,
            fingerprint,
        } => apply_sessions_data(app, result, fingerprint),
        LoadResult::Pricing { result } => apply_pricing(app, result),
        LoadResult::RefreshFailed { message } => apply_refresh_failed(app, &message),
        LoadResult::NoChange => {
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
                    kind: StatusKind::Error,
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
    if let Some(fp) = refresh_fingerprint {
        app.refresh_fingerprint = fp;
    }
}

fn apply_inputs_data(
    app: &mut App,
    result: anyhow::Result<(Vec<AttributionRow>, CoverageStats)>,
    refresh_fingerprint: Option<RefreshFingerprint>,
) {
    let is_user_triggered =
        refresh_fingerprint.is_none() && matches!(&app.inputs, Some(InputsData::Loading));
    // Not gated on `refresh_fingerprint.is_some()`, though the catalog
    // swap that used to justify that no longer reaches here: a visible
    // Inputs tab is routed through `InputsData::Loading` by
    // `invalidate_data`, so its reload lands as `is_user_triggered`.
    // The branch stays reachable from the guarded timer, and stays
    // ungated because a fingerprint-free result must still apply to an
    // already-loaded table rather than be silently dropped.
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
                    kind: StatusKind::Error,
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
    }
}

fn apply_sessions_data(
    app: &mut App,
    result: anyhow::Result<Vec<Session>>,
    fingerprint: Option<RefreshFingerprint>,
) {
    if app.tab != Tab::Sessions || !matches!(app.view, View::List) {
        return;
    }
    match result {
        Ok(sessions) => {
            match &mut app.sessions {
                SessionsData::Loaded(state) => state.apply_refresh(sessions),
                SessionsData::Loading | SessionsData::Error(_) => {
                    app.sessions = SessionsData::Loaded(SessionsState::new(sessions));
                }
            }
            // A forced reload (catalog swap) carries no fresh
            // fingerprint — the filesystem didn't change, so the
            // existing one stands.
            app.refresh_fingerprint =
                fingerprint.unwrap_or_else(|| app.refresh_fingerprint.clone());
            app.clear_status();
        }
        Err(e) => {
            let message = format!("{e}");
            if matches!(app.sessions, SessionsData::Loaded(_)) {
                // A background refresh that fails must not discard
                // rows the user is reading — that is the prior
                // design's silent-refresh rationale, narrowed to where
                // it still applies. It is not owed an immediate footer
                // message either: it is a failure of the same refresh
                // pipeline `apply_refresh_failed` accounts for, so it
                // gets the same silence budget.
                note_refresh_failure(app, &message);
            } else {
                // An empty slot has nothing left to render, so the
                // user is owed an answer now, with no threshold.
                app.status = Some(StatusMessage {
                    text: message.clone(),
                    from_refresh_failure: false,
                    kind: StatusKind::Error,
                });
                app.sessions = SessionsData::Error(message);
            }
        }
    }
}

/// Dispatch whatever the Sessions tab's current view needs when its
/// data is absent or errored.
///
/// Redundant calls are free, and the ledger is what makes them so:
/// both dispatches here are unguarded, and `App::try_reserve` refuses
/// an unguarded dispatch when another unguarded load already holds the
/// slot — that one will produce a result an applier consumes. So
/// key-repeat on `1` cannot queue a `load_sessions` walk per keypress,
/// and callers ask only "is this slot loaded?", which the type forces
/// them to handle, rather than "is a load already running?", which
/// nothing forces anyone to ask. Keeping that question at the dispatch
/// chokepoint instead of here is the whole point of the ledger.
///
/// Retrying from `Error` matches `try_switch_to_inputs`, which
/// re-dispatches from its own error state, and enters `Loading` for
/// the same reason it does: an invisible retry reads as a dead
/// keypress.
fn ensure_sessions_data(app: &mut App) {
    match &app.view {
        View::List => {
            if !matches!(app.sessions, SessionsData::Loaded(_)) {
                app.sessions = SessionsData::Loading;
                app.pending_loads.push(LoadRequest::Refresh {
                    scope: RefreshScope::Sessions,
                    guard: None,
                });
            }
        }
        View::ShowLoading {
            session_id,
            header_label,
        } => {
            app.pending_loads.push(LoadRequest::ShowDetail {
                session_id: session_id.clone(),
                header_label: header_label.clone(),
            });
        }
        View::Show { .. } | View::ShowError { .. } => {}
    }
}

/// Leave a Show-family view for the list, reloading when the sessions
/// slot was invalidated while the user was inside Show.
fn return_to_list(app: &mut App) {
    app.view = View::List;
    ensure_sessions_data(app);
}

/// Empty every slot whose contents predate the new `ctx`, and
/// re-dispatch what the user is currently looking at.
///
/// Called from every `ctx` mutation site, and owns the generation bump
/// so the two cannot drift apart at a call site. That is a convention,
/// not a type-level guarantee — `ctx` is a plain field — held today by
/// there being two mutation sites, `apply_pricing` and
/// `commit_filter_edit`, each of which ends by calling this.
/// Branches on `app.tab` / `app.view` rather than on
/// `current_scope`, which derives the visible scope from slot
/// *contents* and so reports `None` for an Inputs tab sitting in
/// `Loading` or `Error` — clearing that tab with no re-dispatch leaves
/// a blank screen that no navigation event recovers.
///
/// Off-screen slots are cleared but not reloaded: they refill through
/// `ensure_sessions_data` and `try_switch_to_inputs` when the user
/// navigates to them.
/// Apply an agents load, mirroring `apply_inputs_data` including its
/// selection-preservation logic. The selection is keyed on the row key
/// — agent type, model, effort, pinning — rather than on a file path,
/// because an agents row is an accumulation and no path identifies it.
fn apply_agents_data(
    app: &mut App,
    result: anyhow::Result<Vec<AgentRow>>,
    refresh_fingerprint: Option<RefreshFingerprint>,
) {
    let is_user_triggered =
        refresh_fingerprint.is_none() && matches!(&app.agents, Some(AgentsData::Loading));
    let is_refresh_triggered = matches!(&app.agents, Some(AgentsData::Loaded(_)));

    if is_user_triggered {
        match result {
            Ok(rows) => {
                app.agents = Some(AgentsData::Loaded(AgentsState::new(rows)));
                app.clear_status();
            }
            Err(e) => {
                let message = format!("{e}");
                app.status = Some(StatusMessage {
                    text: message.clone(),
                    from_refresh_failure: false,
                    kind: StatusKind::Error,
                });
                app.agents = Some(AgentsData::Error(message));
            }
        }
    } else if is_refresh_triggered
        && let Ok(new_rows) = result
        && let Some(AgentsData::Loaded(state)) = &mut app.agents
    {
        let prev_key = state.selected_key();
        let prev_idx = state.table_state.selected();

        state.rows = new_rows;

        if let Some(prev_key) = prev_key {
            let found = state.rows.iter().position(|r| {
                (
                    r.agent_type.clone(),
                    r.model.clone(),
                    r.effort.clone(),
                    r.pinning.clone(),
                ) == prev_key
            });
            if let Some(new_idx) = found {
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
    // Stored exactly as the three sibling appliers do. Without it the
    // guarded tick keeps comparing against a stale fingerprint, so
    // after any single filesystem change every 3s tick reruns a full
    // `load_agents` instead of resolving to `NoChange`.
    if let Some(fp) = refresh_fingerprint {
        app.refresh_fingerprint = fp;
    }
}

fn invalidate_data(app: &mut App) {
    app.bump_ctx_generation();
    app.sessions = SessionsData::Loading;

    // A `Show` holds costs baked in at load time, a `ShowLoading`
    // holds a promise the bump has just doomed, and a `ShowError` was
    // computed against a superseded context and may not reproduce.
    // All three renew into `ShowLoading`; retrying is cheaper than
    // explaining a stale failure.
    let renewed = match &app.view {
        View::List => None,
        View::Show {
            session_id,
            header_label,
            ..
        }
        | View::ShowLoading {
            session_id,
            header_label,
        }
        | View::ShowError {
            session_id,
            header_label,
            ..
        } => Some((session_id.clone(), header_label.clone())),
    };
    if let Some((session_id, header_label)) = renewed {
        app.view = View::ShowLoading {
            session_id,
            header_label,
        };
    }

    if app.tab == Tab::Inputs {
        app.inputs = Some(InputsData::Loading);
        app.pending_loads.push(LoadRequest::InputsRefresh);
    } else {
        app.inputs = None;
    }

    // Branch on `app.tab`, not on `current_scope`: that derives the
    // visible scope from slot *contents* and so reports `None` for an
    // Agents tab sitting in `Loading` or `Error`, leaving a blank
    // screen no navigation event recovers.
    if app.tab == Tab::Agents {
        app.agents = Some(AgentsData::Loading);
        app.pending_loads.push(LoadRequest::AgentsRefresh);
    } else {
        app.agents = None;
    }

    if app.tab == Tab::Sessions {
        ensure_sessions_data(app);
    }
}

fn apply_pricing(app: &mut App, result: anyhow::Result<(Arc<PricingCatalog>, PricingData)>) {
    match result {
        Ok((catalog, data)) => {
            app.pricing = data;
            app.ctx.catalog = catalog;
            invalidate_data(app);
            app.clear_status();
        }
        Err(e) => {
            // User pressed `r` in the overlay and is owed an answer
            // immediately — no threshold, unlike `apply_refresh_failed`.
            app.status = Some(StatusMessage {
                text: format!("{e}"),
                from_refresh_failure: false,
                kind: StatusKind::Error,
            });
        }
    }
}

/// Account one background-refresh failure against the consecutive
/// threshold, breaking silence only once "wait for the next tick" has
/// stopped being a credible remedy.
///
/// Shared by the two ways a background refresh can fail: the guarded
/// fingerprint build (`apply_refresh_failed`) and the data load behind
/// it (`apply_sessions_data`'s `Err` arm, when rows are already on
/// screen). Both are failures of the same pipeline, so both are owed
/// the same silence budget and the same `from_refresh_failure` stamp —
/// which is what lets a later `NoChange` clear the message.
fn note_refresh_failure(app: &mut App, message: &str) {
    // Single transient failures stay silent — the prior design's
    // rationale holds: valid data is still displayed and the next
    // tick retries. Only a persistent, repeated failure breaks
    // silence.
    app.consecutive_refresh_failures = app.consecutive_refresh_failures.saturating_add(1);
    if app.consecutive_refresh_failures >= CONSECUTIVE_REFRESH_FAILURE_THRESHOLD {
        app.status = Some(StatusMessage {
            // Diagnostic first: the footer is one row, so on a narrow
            // terminal a long `message` truncates — put the part
            // worth reading before the boilerplate, not after it.
            text: format!("{message} — refresh failing, data may be stale"),
            from_refresh_failure: true,
            kind: StatusKind::Error,
        });
    }
}

fn apply_refresh_failed(app: &mut App, message: &str) {
    note_refresh_failure(app, message);
}

fn handle_key_event(app: &mut App, key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return true;
    }
    // The match binds nothing out of the scrutinee, so the borrow of
    // `app.overlay` ends at the arm and the handler can take `&mut App`.
    match &app.overlay {
        Some(Overlay::Pricing) => return handle_pricing_overlay_key(app, key),
        Some(Overlay::Compare(_)) => return handle_compare_overlay_key(app, key),
        Some(Overlay::Filter(_)) => return handle_filter_overlay_key(app, key),
        None => {}
    }
    match key.code {
        KeyCode::Char('1') => {
            app.tab = Tab::Sessions;
            ensure_sessions_data(app);
            return false;
        }
        KeyCode::Char('2') => {
            try_switch_to_inputs(app);
            return false;
        }
        KeyCode::Char('3') => {
            try_switch_to_agents(app);
            return false;
        }
        KeyCode::Char('p') => {
            app.overlay = Some(Overlay::Pricing);
            return false;
        }
        KeyCode::Char('f') => {
            app.overlay = Some(Overlay::Filter(Box::new(FilterEditor::from_query(
                &app.ctx.query,
            ))));
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
        Tab::Agents => handle_agents_key(app, key),
    }
}

/// Keys while the pricing overlay is open. Swallows everything the
/// overlay does not bind, so no global binding fires behind it.
fn handle_pricing_overlay_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc | KeyCode::Char('p' | 'q') => {
            app.overlay = None;
        }
        KeyCode::Char('r') => {
            app.pending_loads.push(LoadRequest::RefreshPricing);
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
    false
}

/// Keys while the filter editor is open. Swallows everything it does
/// not bind, which is the whole point of the overlay owning the
/// keyboard: `q`, `p`, and `r` are ordinary characters here.
fn handle_filter_overlay_key(app: &mut App, key: KeyEvent) -> bool {
    if key.code == KeyCode::Esc {
        // Discards the uncommitted edit; `ctx.query` is untouched.
        app.overlay = None;
        return false;
    }
    if key.code == KeyCode::Enter {
        commit_filter_edit(app);
        return false;
    }
    let Some(Overlay::Filter(editor)) = &mut app.overlay else {
        return false;
    };
    match key.code {
        KeyCode::Tab => editor.focus_next(),
        KeyCode::BackTab => editor.focus_prev(),
        KeyCode::Backspace => editor.backspace(),
        KeyCode::Char(c) => {
            // Ctrl-chords are habitual in a text field (Ctrl+W and
            // Ctrl+U kill a word / the line in most shells). Ctrl+U is
            // the one this editor can honor literally, since clearing
            // needs no cursor; the rest do nothing, because inserting
            // a literal `w` is worse. Ctrl-C never reaches here —
            // `handle_key_event` intercepts it.
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                if matches!(c, 'u' | 'U') {
                    editor.clear_field();
                }
            } else {
                editor.push_char(c);
            }
        }
        KeyCode::Enter
        | KeyCode::Left
        | KeyCode::Right
        | KeyCode::Up
        | KeyCode::Down
        | KeyCode::Home
        | KeyCode::End
        | KeyCode::PageUp
        | KeyCode::PageDown
        | KeyCode::Delete
        | KeyCode::Insert
        | KeyCode::F(_)
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
    false
}

/// Apply the editor's fields to `ctx.query`. Inert while any field
/// fails to parse — the overlay stays open with its error text
/// visible, which is the only feedback an inert `Enter` gives.
fn commit_filter_edit(app: &mut App) {
    let committable = match &app.overlay {
        Some(Overlay::Filter(editor)) => editor.is_committable(),
        Some(Overlay::Pricing | Overlay::Compare(_)) | None => false,
    };
    if !committable {
        return;
    }
    // `take()` ends the borrow on the editor *and* clears the overlay
    // slot, so no separate `app.overlay = None` is needed.
    //
    // `invalidate_data` owns the generation bump, the slot clearing,
    // the Show-view renewal, and the re-dispatch for the visible tab.
    // Do not bump `ctx_generation` here (it would double-count) and do
    // not push a `Refresh` (it carries `is_refresh: true`, which
    // `apply_show_detail` refuses into the `ShowLoading` state that
    // invalidation has just installed).
    let Some(Overlay::Filter(editor)) = app.overlay.take() else {
        return;
    };
    app.ctx.query = editor.to_query(&app.ctx.query);
    invalidate_data(app);
    // As `apply_pricing` does: a user-initiated commit is a fresh
    // action, and a standing error would outrank the empty-state and
    // missing-model feedback the reload is about to produce.
    app.clear_status();
}

fn handle_list_loading_key(key: KeyEvent) -> bool {
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

fn handle_list_error_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Enter | KeyCode::Char('r') => {
            // Enter `Loading` before dispatching, mirroring
            // `handle_show_error_key` and `try_switch_to_inputs`: the
            // retry is otherwise invisible, and the loading handler
            // ignores a second Enter while it runs.
            app.sessions = SessionsData::Loading;
            app.pending_loads.push(LoadRequest::Refresh {
                scope: RefreshScope::Sessions,
                guard: None,
            });
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
    }
}

fn handle_list_key(app: &mut App, key: KeyEvent) -> bool {
    match &app.sessions {
        SessionsData::Loading => return handle_list_loading_key(key),
        SessionsData::Error(_) => return handle_list_error_key(app, key),
        SessionsData::Loaded(_) => {}
    }
    // Handled before the mutable borrow below: `try_open_show` needs
    // `&mut App`, which cannot coexist with a borrow of `app.sessions`.
    if key.code == KeyCode::Enter {
        try_open_show(app);
        return false;
    }
    let SessionsData::Loaded(state) = &mut app.sessions else {
        return false;
    };
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Down | KeyCode::Char('j') => {
            state.list_state.select_next();
            false
        }
        KeyCode::Up | KeyCode::Char('k') => {
            state.list_state.select_previous();
            false
        }
        KeyCode::Home | KeyCode::Char('g') => {
            state.list_state.select_first();
            false
        }
        KeyCode::End | KeyCode::Char('G') => {
            state.list_state.select_last();
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

fn try_open_show(app: &mut App) {
    let SessionsData::Loaded(state) = &app.sessions else {
        return;
    };
    let Some(idx) = state.list_state.selected() else {
        return;
    };
    let Some(session) = state.sessions.get(idx) else {
        return;
    };
    let session_id = session.id.clone();
    let header_label = format!("\"{}\" ({})", session.title, session.project_short_name);
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
            return_to_list(app);
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
            return_to_list(app);
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
            return_to_list(app);
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

fn try_switch_to_agents(app: &mut App) {
    if matches!(app.agents, None | Some(AgentsData::Error(_))) {
        app.agents = Some(AgentsData::Loading);
        app.pending_loads.push(LoadRequest::AgentsRefresh);
    }
    app.tab = Tab::Agents;
}

/// Widen `ctx.query.pinning` to every kind, or back to the slice the
/// process launched with.
///
/// Back means `launch_pinning`, not `PinningFilter::default()`: a user
/// who ran `cclens agents --pinning pinned` and pressed `a` twice
/// would otherwise be moved silently to a slice they never asked for,
/// with no pinning field in the filter overlay to get back from.
///
/// Commits through `invalidate_data`, the same path
/// `commit_filter_edit` uses, so the generation bump and the slot
/// clearing happen at one site rather than two.
fn toggle_agents_pinning(app: &mut App) {
    app.ctx.query.pinning = if app.ctx.query.pinning == PinningFilter::everything() {
        app.launch_pinning.clone()
    } else {
        PinningFilter::everything()
    };
    invalidate_data(app);
    app.clear_status();
}

/// Open the comparison modal for the selected row's agent type.
///
/// Carries the visible rows for that agent type only. The targets are
/// the catalog's bare `claude-*` keys — the same set `pricing list`
/// shows without `--all`, which are the keys that match transcript
/// model strings.
fn open_compare_overlay(app: &mut App) {
    let Some(AgentsData::Loaded(state)) = &app.agents else {
        return;
    };
    let Some(selected) = state.table_state.selected() else {
        return;
    };
    let Some(agent_type) = state.rows.get(selected).map(|r| r.agent_type.clone()) else {
        return;
    };
    let rows: Vec<AgentRow> = state
        .rows
        .iter()
        .filter(|r| r.agent_type == agent_type)
        .cloned()
        .collect();
    // `target_idx` seeds from `--compare-model` when the run supplied
    // one, so the modal opens on the model the user already named
    // rather than on whatever sorts first.
    let target_idx = app
        .compare_model
        .as_deref()
        .and_then(|wanted| {
            compare_targets(&app.ctx.catalog)
                .iter()
                .position(|t| t == wanted)
        })
        .unwrap_or(0);
    app.overlay = Some(Overlay::Compare(Box::new(CompareState {
        agent_type,
        rows,
        target_idx,
    })));
}

/// Keys while the comparison overlay is open. Swallows everything the
/// overlay does not bind, so no global binding fires behind it —
/// matching `handle_pricing_overlay_key`'s contract.
fn handle_compare_overlay_key(app: &mut App, key: KeyEvent) -> bool {
    // Cloned before the overlay borrow: `move_target` reads the live
    // catalog, and `state` borrows `app` mutably. An `Arc` bump.
    let catalog = Arc::clone(&app.ctx.catalog);
    let Some(Overlay::Compare(state)) = &mut app.overlay else {
        return false;
    };
    match key.code {
        KeyCode::Char('c' | 'q') | KeyCode::Esc => {
            app.overlay = None;
        }
        KeyCode::Down | KeyCode::Char('j') => state.move_target(&catalog, true),
        KeyCode::Up | KeyCode::Char('k') => state.move_target(&catalog, false),
        KeyCode::Enter
        | KeyCode::Backspace
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
        | KeyCode::Modifier(_) => {}
    }
    false
}

fn handle_agents_loading_key(key: KeyEvent) -> bool {
    handle_inputs_loading_key(key)
}

/// Keys on an `AgentsData::Error` slot: retry, quit, or nothing.
fn handle_agents_error_key(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Enter | KeyCode::Char('r') => {
            try_switch_to_agents(app);
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
    }
}

/// Navigation on a loaded Agents table. Split from
/// `handle_agents_key` because `c` and `a` need `&mut App` while these
/// arms hold a `&mut AgentsState` borrowed out of it.
fn handle_agents_loaded_key(agents: &mut AgentsState, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Down | KeyCode::Char('j') => {
            agents.table_state.select_next();
            false
        }
        KeyCode::Up | KeyCode::Char('k') => {
            agents.table_state.select_previous();
            false
        }
        KeyCode::Home | KeyCode::Char('g') => {
            agents.table_state.select_first();
            false
        }
        KeyCode::End | KeyCode::Char('G') => {
            agents.table_state.select_last();
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

fn handle_agents_key(app: &mut App, key: KeyEvent) -> bool {
    if matches!(&app.agents, Some(AgentsData::Loading)) {
        return handle_agents_loading_key(key);
    }
    if matches!(&app.agents, Some(AgentsData::Error(_))) {
        return handle_agents_error_key(app, key);
    }
    // `c` and `a` need `&mut App`, so they are dispatched before the
    // `&mut AgentsState` borrow below begins.
    match key.code {
        KeyCode::Char('c') => {
            open_compare_overlay(app);
            return false;
        }
        KeyCode::Char('a') => {
            toggle_agents_pinning(app);
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
    let Some(AgentsData::Loaded(agents)) = &mut app.agents else {
        return false;
    };
    handle_agents_loaded_key(agents, key)
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
                    " 1/2/3 tabs  Esc back  q quit",
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
                    " 1/2/3 tabs  Enter retry  Esc back  q quit",
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
        Tab::Agents => render_agents_content(app, frame, content_area),
    }

    match &app.overlay {
        Some(Overlay::Pricing) => render_pricing_overlay(app, frame),
        Some(Overlay::Filter(editor)) => render_filter_overlay(editor, frame),
        Some(Overlay::Compare(state)) => render_compare_overlay(state, &app.ctx.catalog, frame),
        None => {}
    }
}

/// The leading components that fit within `width`, plus how many were
/// dropped. Whole components only: every rendered component must still
/// reparse to what `Query` holds, and a mid-value ellipsis would break
/// that. Measured with `chars().count()` — project names are
/// user-authored and frequently multi-byte.
fn fit_filter_components(
    components: &[FilterComponent],
    width: usize,
) -> (Vec<&FilterComponent>, usize) {
    let measure = |(i, c): (usize, &FilterComponent)| {
        // Every component after the first carries a separating space.
        c.text.chars().count() + usize::from(i > 0)
    };
    let total: usize = components.iter().enumerate().map(measure).sum();
    if total <= width {
        return (components.iter().collect(), 0);
    }

    // Something must drop, so the ` +N` marker is certain. Reserve its
    // widest possible width up front — `N` can never exceed the
    // component count — so the marker itself cannot overflow the half.
    let reserved = format!(" +{}", components.len()).chars().count();
    let budget = width.saturating_sub(reserved);
    let mut fitted = Vec::new();
    let mut used = 0usize;
    for (i, component) in components.iter().enumerate() {
        let cost = measure((i, component));
        if used + cost > budget {
            break;
        }
        used += cost;
        fitted.push(component);
    }
    let dropped = components.len() - fitted.len();
    (fitted, dropped)
}

fn render_tab_header(app: &App, frame: &mut Frame, area: ratatui::layout::Rect) {
    // Bracketed when active, bare when not — rather than the padded
    // `" Sessions "` a two-tab strip could afford. A third tab costs
    // the left area about ten columns, which at 80 (the common
    // terminal width) is the difference between the filter indicator
    // rendering and vanishing entirely.
    let label = |tab: Tab, name: &str| {
        if app.tab == tab {
            format!("[{name}]")
        } else {
            name.to_string()
        }
    };
    let sessions_label = label(Tab::Sessions, "Sessions");
    let inputs_label = label(Tab::Inputs, "Inputs");
    let agents_label = label(Tab::Agents, "Agents");

    let context = match app.tab {
        Tab::Sessions => match &app.view {
            View::Show { header_label, .. }
            | View::ShowLoading { header_label, .. }
            | View::ShowError { header_label, .. } => header_label.clone(),
            View::List => match &app.sessions {
                SessionsData::Loaded(state) => {
                    let count = state.sessions.len();
                    let label = if count == 1 { "session" } else { "sessions" };
                    format!("{count} {label}")
                }
                SessionsData::Loading => "...".to_string(),
                SessionsData::Error(_) => "!".to_string(),
            },
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
        Tab::Agents => match &app.agents {
            Some(AgentsData::Loaded(agents)) => {
                let count = agents.rows.len();
                let label = if count == 1 { "agent" } else { "agents" };
                format!("{count} {label}")
            }
            Some(AgentsData::Loading) => "...".to_string(),
            Some(AgentsData::Error(_)) => "!".to_string(),
            None => "0 agents".to_string(),
        },
    };

    // The right label claims what it needs rather than half the row.
    // An even split leaves the left half too narrow at 80 columns —
    // the common terminal width — to fit even `--project alpha`,
    // while the right half renders mostly blanks.
    //
    // Capped at half, because `context` is Show's `header_label` and
    // carries an untruncated session title: unclamped, `Length` is
    // satisfied first and `Fill` gets zero columns, taking the tabs
    // and the whole filter indicator off screen.
    let right_text = format!("cclens — {context} ");
    let right_width = u16::try_from(right_text.chars().count())
        .unwrap_or(u16::MAX)
        .min(area.width / 2);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Fill(1), Constraint::Length(right_width)]).areas(area);

    let tabs = format!(" {sessions_label} {inputs_label} {agents_label} ");
    let mut left_spans = vec![Span::raw(tabs.clone()).bold()];

    // The indicator describes all of `Query` bar one documented
    // exception. Over-claiming is the safe direction: a component that
    // does not constrain the visible tab is dimmed, never omitted,
    // because a hidden filter is one actively narrowing what the user
    // is reading with nothing on screen to say so.
    //
    // The exception is `Query::pinning` at its default, which narrows
    // but emits no component — `PinningFilter::describe_active`
    // explains why it must, and the agents view names its own slice
    // unconditionally in the places this header cannot.
    let components = app.ctx.query.describe_active();
    if !components.is_empty() {
        const MARKER: &str = "  filter: ";
        let available = (left_area.width as usize)
            .saturating_sub(tabs.chars().count() + MARKER.chars().count());
        let (fitted, dropped) = fit_filter_components(&components, available);
        // `+N` on its own still renders when no component fits: a
        // narrow terminal is exactly where a silently hidden filter
        // does the most damage. Below the width that fits the marker
        // too, the count goes on alone — it is the information; the
        // word `filter:` is only the label, and pushing both would
        // have ratatui clip the count off the end of the row.
        let room_for_marker = (left_area.width as usize).saturating_sub(tabs.chars().count())
            >= MARKER.chars().count() + format!("+{dropped}").chars().count();
        let count_alone = fitted.is_empty() && !room_for_marker;
        if !fitted.is_empty() || dropped > 0 {
            if count_alone {
                left_spans.push(Span::raw(format!(" +{dropped}")).dim());
            } else {
                left_spans.push(Span::raw(MARKER).dim());
            }
            for (i, component) in fitted.iter().enumerate() {
                if i > 0 {
                    left_spans.push(Span::raw(" "));
                }
                // Identical text on both tabs; only the styling
                // differs, which is what keeps the TUI's indicator
                // from diverging from the CLI's flag-shaped hint.
                let span = Span::raw(component.text.clone());
                let applies = component.honored_by.honors(tab_scope(app.tab));
                left_spans.push(if applies { span } else { span.dim() });
            }
            if dropped > 0 && !count_alone {
                // The separating space has nothing to separate when
                // every component dropped. `fit_filter_components`
                // reserved room for it either way, so spending one
                // less column here can only help.
                let lead = if fitted.is_empty() { "" } else { " " };
                left_spans.push(Span::raw(format!("{lead}+{dropped}")).dim());
            }
        }
    }

    let left = Line::from(left_spans);
    let right = Line::from(right_text).bold().right_aligned();
    frame.render_widget(Paragraph::new(left), left_area);
    frame.render_widget(Paragraph::new(right), right_area);
}

/// The loader behind a tab, for styling the header indicator. Only
/// the header goes through this: Show renders under the Sessions tab
/// but loads by session id, so the empty states name their own
/// `QueryScope` rather than deriving one from the tab.
fn tab_scope(tab: Tab) -> QueryScope {
    match tab {
        Tab::Sessions => QueryScope::Sessions,
        Tab::Inputs => QueryScope::Inputs,
        Tab::Agents => QueryScope::Agents,
    }
}

/// Where a view's emptiness comes from when no filter caused it. The
/// views differ: List and Inputs really are reading an empty
/// `projects_dir`, while a Show with no exchanges is one session that
/// has none — naming the directory there explains nothing.
#[derive(Clone, Copy)]
enum EmptySource<'a> {
    ProjectsDir(&'a Path),
    /// The `~/.claude` tree. Inputs rows come one per inventory file
    /// (`compute_rows`), so `projects_dir` cannot be what emptied it.
    Inventory(&'a Path),
    Session,
}

/// What a zero-row view says. Branches on whether any filter is
/// active, because the two cases have different remedies: a filtered
/// view is fixed by pressing `f`, an empty `EmptySource` is not fixed
/// by any filter change.
fn empty_state_lines(
    query: &Query,
    noun: &str,
    source: EmptySource<'_>,
    scope: QueryScope,
) -> Vec<Line<'static>> {
    // Unlike the header indicator, which over-claims deliberately (a
    // dimmed component the tab context explains), the empty state
    // must under-claim: it asserts causation, so a component that
    // could not have caused this emptiness has no place in it.
    let components: Vec<FilterComponent> = query
        .describe_active()
        .into_iter()
        .filter(|c| c.honored_by.honors(scope))
        .collect();
    if components.is_empty() {
        let explanation = match source {
            EmptySource::ProjectsDir(dir) | EmptySource::Inventory(dir) => {
                format!(" No {noun} found in {}.", dir.display())
            }
            EmptySource::Session => format!(" No {noun} in this session."),
        };
        return vec![Line::raw(""), Line::from(explanation)];
    }
    let joined = components
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    vec![
        Line::raw(""),
        Line::from(format!(" No {noun} match {joined}")),
        Line::raw(""),
        Line::from(" Press f to change filters.").dim(),
    ]
}

fn render_list_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    // Each arm yields owned `Line<'static>`s rather than rendering in
    // place, so the borrow of `app.sessions` ends with the match and
    // `app` is free to be reborrowed for the status footer below.
    let (message_lines, footer) = match &app.sessions {
        SessionsData::Loaded(_) => {
            render_loaded_list_content(app, frame, area);
            return;
        }
        SessionsData::Loading => (
            vec![Line::raw(""), Line::from(" Loading sessions...")],
            " 1/2/3 tabs  q quit",
        ),
        SessionsData::Error(msg) => (
            vec![
                Line::raw(""),
                Line::from(format!(" Error: {msg}")),
                Line::raw(""),
                Line::from(" Press Enter or r to retry.").dim(),
            ],
            " 1/2/3 tabs  Enter retry  q quit",
        ),
    };
    render_feedback_content(app, frame, area, message_lines, footer);
}

fn render_loaded_list_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    // A filtered-to-empty slot is `Loaded` with zero rows, not
    // `Loading` — the branch belongs here so the two stay
    // distinguishable during the reload window a commit opens.
    let is_empty = match &app.sessions {
        SessionsData::Loaded(state) => state.sessions.is_empty(),
        SessionsData::Loading | SessionsData::Error(_) => false,
    };
    if is_empty {
        let lines = empty_state_lines(
            &app.ctx.query,
            "sessions",
            EmptySource::ProjectsDir(&app.ctx.projects_dir),
            QueryScope::Sessions,
        );
        // Navigation and open keys are omitted — there is nothing to
        // navigate or open.
        render_feedback_content(
            app,
            frame,
            area,
            lines,
            " 1/2/3 tabs  f filter  p pricing  q quit",
        );
        return;
    }

    let [table_area, totals_area, footer_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(area);

    // Derived over the same rows the table just rendered:
    // `load_sessions` applies the thresholds before the slot is
    // filled, so this is never a superset of what is on screen.
    let derived = if let SessionsData::Loaded(state) = &mut app.sessions {
        render_sessions_table(state, frame, table_area);
        render_totals(state, frame, totals_area);
        state
            .sessions
            .iter()
            .any(|s| s.cost_breakdown.is_none())
            .then_some(MISSING_MODEL_HINT)
    } else {
        None
    };
    render_status_footer(
        app,
        frame,
        footer_area,
        " 1/2/3 tabs  ↑↓ navigate  Enter open  f filter  p pricing  q quit",
        derived,
    );
}

fn render_sessions_table(
    state: &mut SessionsState,
    frame: &mut Frame,
    area: ratatui::layout::Rect,
) {
    let rows: Vec<Row> = state
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

    frame.render_stateful_widget(table, area, &mut state.list_state);
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

fn render_totals(state: &SessionsState, frame: &mut Frame, area: ratatui::layout::Rect) {
    if state.sessions.len() < 2 {
        return;
    }
    let row = Row::new(vec![
        Line::raw(""),
        Line::raw(""),
        Line::raw("total").right_aligned(),
        Line::raw(format_tokens(state.total_tokens)).right_aligned(),
        Line::raw(format_cost_opt(state.total_cost)).right_aligned(),
    ]);

    let table = Table::new(vec![row], session_table_widths());
    frame.render_widget(table, area);
}

/// One footer row, four possible producers, in this precedence:
/// 1. `app.status` with `StatusKind::Error` — red, takes the row
/// 2. `app.status` with `StatusKind::Info` — dim, takes the row
/// 3. `hint` plus `derived`, when both fit — dim
/// 4. `hint` alone — dim
///
/// A status outranks the row because it is the newer, user-triggered
/// fact, and because it clears. `derived` describes a standing
/// condition that will still be true after the status clears, so it
/// shares the row with the key hints rather than evicting them.
/// `derived` is computed by the callers that render priced rows and
/// passed in, not derived here — a feedback view has no rows and
/// correctly reports nothing.
fn render_status_footer(
    app: &App,
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    hint: &str,
    derived: Option<DerivedHint>,
) {
    let line = if let Some(status) = &app.status {
        match status.kind {
            StatusKind::Error => Line::from(format!(" {}", status.text)).red(),
            StatusKind::Info => Line::from(format!(" {}", status.text)).dim(),
        }
    } else {
        // A standing condition shares the row rather than taking it.
        // A status clears; this does not, and the key hints are the
        // only place `q quit` and `f filter` are advertised on a
        // loaded view — evicting them permanently costs the user the
        // way out. The derived half is what drops when the row is too
        // narrow for both.
        // Longest spelling that fits, rather than all-or-nothing:
        // the full remedy needs 114 columns beside the list hints, so
        // an 80-column terminal would never see the condition at all.
        let text = derived
            .and_then(|d| {
                [d.long, d.short]
                    .into_iter()
                    .map(|form| format!("{hint}  {form}"))
                    .find(|combined| combined.chars().count() <= area.width as usize)
            })
            .unwrap_or_else(|| hint.to_string());
        Line::from(text).dim()
    };
    frame.render_widget(Paragraph::new(line), area);
}

/// Named once and rendered from two views.
///
/// The remedy is stated conditionally because the condition has causes
/// a refresh cannot fix: `cost_for_components` returns `None` both when
/// the catalog lacks the model *and* when the transcript's model string
/// is absent or unpriceable (Claude Code writes `<synthetic>` on
/// API-error turns, which no catalog will ever carry). Promising
/// `press p then r` outright would advertise a remedy that, for those
/// sessions, can never work — permanently, since the condition is
/// recomputed from the same rows every frame.
const MISSING_MODEL_HINT: DerivedHint = DerivedHint {
    long: "some rows unpriced — p then r if catalog is stale",
    // Sized to fit: the list hints are 63 columns, leaving 15 at an
    // 80-column terminal once the two-space separator is paid for.
    short: "unpriced rows",
};

/// A standing condition the footer reports beside the key hints, in
/// two spellings. `long` names the remedy; `short` names only the
/// condition, so a narrow row still says that something is unpriced
/// instead of silently saying nothing.
#[derive(Clone, Copy)]
struct DerivedHint {
    long: &'static str,
    short: &'static str,
}

// ---- show view ----

fn render_show_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    // `load_show` applies `ctx.query.thresholds` through
    // `prepare_exchanges`, so a threshold committed from inside Show
    // can empty it. Whatever the `f` binding makes reachable, this
    // view has to explain.
    let is_empty = match &app.view {
        View::Show { prepared, .. } => prepared.is_empty(),
        View::List | View::ShowLoading { .. } | View::ShowError { .. } => false,
    };
    if is_empty {
        let lines = empty_state_lines(
            &app.ctx.query,
            "exchanges",
            EmptySource::Session,
            QueryScope::Show,
        );
        render_feedback_content(
            app,
            frame,
            area,
            lines,
            " 1/2/3 tabs  Esc back  f filter  p pricing  q quit",
        );
        return;
    }

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

    // Deliberately `None`: a `PreparedExchange` is a per-turn exchange
    // rather than a priced session, so "unpriced" is undefined here.
    render_status_footer(
        app,
        frame,
        footer_area,
        " 1/2/3 tabs  ↑↓ navigate  Esc back  f filter  p pricing  q quit",
        None,
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
    // A feedback view has no rows, so it has no standing condition to
    // report — the derived slot is correctly empty here.
    render_status_footer(app, frame, footer_area, footer_text, None);
}

// ---- inputs view ----

fn render_inputs_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    // Checked before the `&mut app.inputs` borrow below, which would
    // otherwise conflict with reading `app.ctx`. `run_inputs` has no
    // startup empty guard, so this is the only explanation that
    // launch path ever produces for an empty attribution table.
    let is_empty = match &app.inputs {
        Some(InputsData::Loaded(inputs)) => inputs.rows.is_empty(),
        Some(InputsData::Loading | InputsData::Error(_)) | None => false,
    };
    if is_empty {
        // Resolved here rather than carried on `ctx`: this is the
        // only site that needs it, and only on a path that renders no
        // rows.
        let claude_home = InventoryConfig::default().claude_home;
        let lines = empty_state_lines(
            &app.ctx.query,
            "files",
            EmptySource::Inventory(&claude_home),
            QueryScope::Inputs,
        );
        render_feedback_content(
            app,
            frame,
            area,
            lines,
            " 1/2/3 tabs  f filter  p pricing  q quit",
        );
        return;
    }

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
            // Bound before the call for the same borrow reason the
            // `Error` arm's comment below documents: `inputs` borrows
            // this match's scrutinee, and that borrow must end before
            // `app` is reborrowed for the footer.
            let derived = inputs
                .rows
                .iter()
                .any(|r| r.attributed_cost.is_none())
                .then_some(MISSING_MODEL_HINT);
            render_status_footer(
                app,
                frame,
                footer_area,
                " 1/2/3 tabs  ↑↓ navigate  f filter  p pricing  q quit",
                derived,
            );
        }
        Some(InputsData::Loading) => {
            render_feedback_content(
                app,
                frame,
                area,
                vec![Line::raw(""), Line::from(" Loading inputs...")],
                " 1/2/3 tabs  q quit",
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
            render_feedback_content(app, frame, area, lines, " 1/2/3 tabs  Enter retry  q quit");
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

// ---- agents tab ----

/// The active pinning slice, spelled for the footer and empty state.
///
/// Appended on both paths regardless of whether `describe_active`
/// emitted a component: at the default it emits none, so without this
/// the empty state would report "No agents found in …" for a view
/// whose `Pinned` and `Fork` rows the filter is holding back.
fn agents_slice_note(app: &App) -> String {
    format!("pinning: {}", app.ctx.query.pinning.describe_slice())
}

fn render_agents_content(app: &mut App, frame: &mut Frame, area: ratatui::layout::Rect) {
    let is_empty = match &app.agents {
        Some(AgentsData::Loaded(agents)) => agents.rows.is_empty(),
        Some(AgentsData::Loading | AgentsData::Error(_)) | None => false,
    };
    if is_empty {
        // `ProjectsDir`, not `Inventory`: the agents view reads
        // transcripts, so `projects_dir` is what can empty it.
        let projects_dir = app.ctx.projects_dir.clone();
        let mut lines = empty_state_lines(
            &app.ctx.query,
            "agents",
            EmptySource::ProjectsDir(&projects_dir),
            QueryScope::Agents,
        );
        lines.push(Line::raw(""));
        lines.push(Line::from(format!(" Covering {}.", agents_slice_note(app))).dim());
        lines.push(Line::from(" Press a to widen to every kind.").dim());
        render_feedback_content(
            app,
            frame,
            area,
            lines,
            " 1/2/3 tabs  a widen  f filter  p pricing  q quit",
        );
        return;
    }

    let slice = agents_slice_note(app);
    match &mut app.agents {
        Some(AgentsData::Loaded(agents)) => {
            let [table_area, slice_area, footer_area] = Layout::vertical([
                Constraint::Fill(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .areas(area);

            render_agents_table(agents, frame, table_area);
            frame.render_widget(Paragraph::new(Line::from(format!(" {slice}"))), slice_area);
            let derived = agents
                .rows
                .iter()
                .any(|r| r.cost.is_none())
                .then_some(MISSING_MODEL_HINT);
            render_status_footer(
                app,
                frame,
                footer_area,
                " 1/2/3 tabs  ↑↓ navigate  c compare  a widen  f filter  p pricing  q quit",
                derived,
            );
        }
        Some(AgentsData::Loading) => {
            render_feedback_content(
                app,
                frame,
                area,
                vec![Line::raw(""), Line::from(" Loading agents...")],
                " 1/2/3 tabs  q quit",
            );
        }
        Some(AgentsData::Error(msg)) => {
            let lines = vec![
                Line::raw(""),
                Line::from(format!(" Error: {msg}")),
                Line::raw(""),
                Line::from(" Press Enter or r to retry.").dim(),
            ];
            render_feedback_content(app, frame, area, lines, " 1/2/3 tabs  Enter retry  q quit");
        }
        None => {}
    }
}

/// Column widths, weighted so the two identifying columns win the
/// flexible space. At 80 columns — the common terminal width — the
/// fixed columns leave little to share, and a row whose `agent` cell
/// is truncated past recognition cannot be acted on at all, whereas a
/// clipped `dispatches` header still reads from its numbers.
fn agents_table_widths() -> [Constraint; 8] {
    [
        Constraint::Fill(3),    // agent
        Constraint::Fill(2),    // model
        Constraint::Length(6),  // effort
        Constraint::Length(8),  // declared
        Constraint::Length(13), // pinning
        Constraint::Length(6),  // disp
        Constraint::Length(8),  // tokens
        Constraint::Length(9),  // cost
    ]
}

fn render_agents_table(agents: &mut AgentsState, frame: &mut Frame, area: ratatui::layout::Rect) {
    let rows: Vec<Row> = agents
        .rows
        .iter()
        .map(|row| {
            let cells = agents_cells(row);
            Row::new(vec![
                Line::raw(cells.agent),
                Line::raw(cells.model),
                Line::raw(cells.effort),
                Line::raw(cells.declared_effort),
                Line::raw(cells.pinning),
                Line::raw(cells.dispatches).right_aligned(),
                Line::raw(cells.tokens).right_aligned(),
                Line::raw(cells.cost).right_aligned(),
            ])
        })
        .collect();

    // `declared` / `disp` rather than the plain renderer's
    // `declared_effort` / `dispatches`: the TUI is width-constrained
    // in a way the plain table is not, and a header wider than its
    // column buys nothing.
    let header = Row::new([
        "agent", "model", "effort", "declared", "pinning", "disp", "tokens", "cost",
    ])
    .style(Style::new().bold());

    let table = Table::new(rows, agents_table_widths())
        .header(header)
        .row_highlight_style(Style::new().reversed());

    frame.render_stateful_widget(table, area, &mut agents.table_state);
}

// ---- comparison overlay ----

/// One agent's rows as a model × effort grid, over a repricing panel.
///
/// A cell holding more than one row is an agent whose dispatches
/// disagree about pinning; the cell says so rather than merging them,
/// because the disagreement is the finding.
fn render_compare_overlay(state: &CompareState, catalog: &PricingCatalog, frame: &mut Frame) {
    let area = frame.area();
    // Wide enough for the row format below (73 columns) plus the
    // border. Narrower clipped the delta column and the upper-bound
    // label — and a repriced figure whose label is cut off is exactly
    // the number this modal exists to keep qualified.
    let popup_width = 78.min(area.width);
    let popup_height = u16::try_from(state.rows.len() + 13)
        .unwrap_or(u16::MAX)
        .min(area.height);
    let popup_area = centered_rect(popup_width, popup_height, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::bordered().title(format!(" Compare — {} ", state.agent_type));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let mut lines: Vec<Line> = Vec::new();
    let target = state.target(catalog);
    let target = target.as_deref();

    lines.push(
        Line::from(format!(
            " {:<20} {:<7} {:<13} {:>9} {:>9} {:>9}",
            "model", "effort", "pinning", "cost", "repriced", "delta",
        ))
        .bold(),
    );
    // Truncated, not just padded: a real catalog key such as
    // `claude-sonnet-4-5-20250929` is 26 characters and would shift
    // every column right, pushing `delta` past the panel's inner
    // width — the exact clipping the popup width exists to prevent.
    // Scalar-aware, per the crate's count-scalars-not-bytes rule.
    let fit_model = |model: &str| -> String {
        const MAX: usize = 20;
        if model.chars().count() <= MAX {
            return model.to_string();
        }
        let mut out: String = model.chars().take(MAX - 1).collect();
        out.push('\u{2026}');
        out
    };
    for row in &state.rows {
        let cells = agents_cells(row);
        let (repriced, delta) = target.map_or_else(
            || ("—".to_string(), "—".to_string()),
            |t| repriced_cells(row, t, catalog),
        );
        lines.push(Line::from(format!(
            " {:<20} {:<7} {:<13} {:>9} {:>9} {:>9}",
            fit_model(&cells.model),
            cells.effort,
            cells.pinning,
            cells.cost,
            repriced,
            delta,
        )));
    }

    // A model × effort pair carrying more than one row is a pinning
    // disagreement; label it rather than merging the rows away.
    let mut seen: Vec<(Option<String>, Option<String>)> = Vec::new();
    let mut duplicated = false;
    for row in &state.rows {
        let key = (row.model.clone(), row.effort.clone());
        if seen.contains(&key) {
            duplicated = true;
        } else {
            seen.push(key);
        }
    }
    if duplicated {
        lines.push(Line::raw(""));
        lines.push(Line::from(" Rows sharing a model and effort disagree about pinning.").dim());
    }

    lines.push(Line::raw(""));
    match target {
        None => lines.push(Line::from(" No catalog models to compare against.")),
        Some(t) => {
            let d = repriced_delta(&state.rows, t, catalog);
            lines.push(Line::from(format!(" vs {t}")).bold());
            if d.rows_counted == 0 {
                lines.push(Line::from(" No comparable rows."));
            } else {
                let direction = if d.delta >= 0.0 { "more" } else { "less" };
                let noun = if d.rows_counted == 1 { "row" } else { "rows" };
                lines.push(Line::from(format!(
                    " {} {direction} across {} {noun}",
                    format_cost_opt(Some(d.delta.abs())),
                    d.rows_counted,
                )));
            }
            // Split across two lines rather than one long one: at
            // this width a single line clips, and a repriced figure
            // whose qualifier is cut off reads as a promise.
            lines.push(Line::from(" Upper bound — the same work on another model").dim());
            lines.push(Line::from(" produces a smaller token bundle than this one.").dim());
            let fork_noun = if d.rows_forked == 1 { "row" } else { "rows" };
            let unpriced_noun = if d.rows_unpriced == 1 { "row" } else { "rows" };
            lines.push(
                Line::from(format!(
                    " Excluded {} fork {fork_noun} and {} unpriced {unpriced_noun}.",
                    d.rows_forked, d.rows_unpriced,
                ))
                .dim(),
            );
        }
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(" ↑↓ target   c/Esc close").dim());

    frame.render_widget(Paragraph::new(lines), inner);
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

    let footer = if app.slot_busy(&LoadSlot::Pricing) {
        Line::from(" refreshing\u{2026}  Esc close").dim()
    } else {
        let staleness = format_cache_staleness(&app.pricing.cache_info);
        Line::from(format!(" {staleness}  r refresh  Esc close")).dim()
    };
    frame.render_widget(Paragraph::new(footer), footer_area);
}

/// The filter editor popup. Centered over the underlying view rather
/// than replacing it, so the rows the commit is about to change stay
/// visible while the fields are edited.
fn render_filter_overlay(editor: &FilterEditor, frame: &mut Frame) {
    // ` > ` plus the 10-column label and its trailing space.
    const PREFIX_WIDTH: usize = 14;

    let area = frame.area();
    // Wide enough that the `session` row — a 36-character UUID plus
    // its `(clear-only)` marker — is not clipped, with headroom for a
    // parse-error message beside a short value.
    let popup_width = 72.min(area.width);
    // Six field rows, a blank spacer, a footer, and two border rows.
    let popup_height = 10.min(area.height);

    let popup_area = centered_rect(popup_width, popup_height, area);
    frame.render_widget(Clear, popup_area);

    let block = Block::bordered().title(" Filter ");
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let [fields_area, footer_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

    let row_width = fields_area.width as usize;

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, field) in editor.fields.iter().enumerate() {
        let marker = if i == editor.focused { '>' } else { ' ' };
        let label = field.kind.label();
        let suffix = match &field.parsed {
            Err(message) => Some(format!("  {message}")),
            // The marker explains why a visible UUID cannot be typed
            // over. With the field empty there is nothing to explain,
            // so it renders blank like the other five.
            Ok(_) if !field.kind.accepts_text() && !field.input.is_empty() => {
                Some("  (clear-only)".to_string())
            }
            Ok(_) => None,
        };
        // The suffix keeps its place and the value gives ground: a
        // value long enough to push the parse error off the row would
        // leave "fix errors to apply" pointing at nothing. The tail is
        // what survives, being what the user just typed.
        let budget = row_width
            .saturating_sub(PREFIX_WIDTH + suffix.as_ref().map_or(0, |s| s.chars().count()));
        let mut spans = vec![
            Span::raw(format!(" {marker} {label:<10} ")),
            Span::raw(elide_front(&field.input, budget)),
        ];
        if let Some(suffix) = suffix {
            spans.push(Span::raw(suffix).dim());
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::raw(""));
    // The popup is capped to the terminal height, so on a short
    // terminal the six field rows outnumber the rows they have.
    // Scroll to keep the focused field — and the parse error rendered
    // beside it — on screen: the footer's "fix errors to apply" is a
    // dead end while the field it refers to is clipped away.
    let visible = fields_area.height as usize;
    let offset = if visible == 0 || editor.focused < visible {
        0
    } else {
        editor.focused + 1 - visible
    };
    frame.render_widget(
        Paragraph::new(lines).scroll((u16::try_from(offset).unwrap_or(0), 0)),
        fields_area,
    );

    let footer = if editor.is_committable() {
        Line::from(" Tab move  Enter apply  Esc cancel").dim()
    } else {
        Line::from(" Tab move  fix errors to apply  Esc cancel").dim()
    };
    frame.render_widget(Paragraph::new(footer), footer_area);
}

/// The last `width` scalars of `s`, marked with a leading `…` when
/// anything was dropped. Scalar-counted, like every other width
/// decision here — filter values are user-authored text.
fn elide_front(s: &str, width: usize) -> String {
    let count = s.chars().count();
    if count <= width {
        return s.to_string();
    }
    if width <= 1 {
        return "…".chars().take(width).collect();
    }
    std::iter::once('…')
        .chain(s.chars().skip(count - (width - 1)))
        .collect()
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
    use crate::filter::{HonoredBy, SessionFilter, ThresholdsFilter};
    use crate::inventory::{ContextFile, ContextFileKind, Scope};
    use crate::pricing::{ClaudePricing, TieredRate};

    /// A `DataContext` pointed at a nonexistent directory with an
    /// empty pricing catalog — every test in this module drives `App`
    /// state directly rather than through a real load, so the context
    /// itself is never dereferenced.
    fn fixture_ctx() -> DataContext {
        DataContext {
            projects_dir: PathBuf::from("/nonexistent"),
            // Holds the same two models `fixture_pricing_data` lists.
            // In production `ctx.catalog` and `pricing.entries` are
            // derived from one catalog and swapped together, so a
            // fixture where they disagree would test a state the app
            // never reaches.
            catalog: Arc::new(
                PricingCatalog::from_raw_json(
                    r#"{"claude-haiku-4-5":{"input_cost_per_token":0.0000008,
                        "output_cost_per_token":0.0000008,
                        "cache_read_input_token_cost":0.0000008,
                        "cache_creation_input_token_cost":0.0000008},
                        "claude-sonnet-4-5":{"input_cost_per_token":0.000003,
                        "output_cost_per_token":0.000015,
                        "cache_read_input_token_cost":0.0000003,
                        "cache_creation_input_token_cost":0.00000375}}"#,
                )
                .expect("fixture catalog parses"),
            ),
            inventory: Arc::new(InventoryConfig::default()),
            query: crate::loading::Query::default(),
        }
    }

    fn new_app(sessions: Vec<Session>, pricing: PricingData, default_tab: Tab) -> App {
        App::new(fixture_ctx(), sessions, pricing, default_tab)
    }

    /// Reach the loaded sessions slot. Every test that touches rows,
    /// totals, or the selection expects a `Loaded` slot — an empty one
    /// at that point is the test's own setup being wrong, so this
    /// panics rather than silently returning a default.
    fn sessions_state(app: &App) -> &SessionsState {
        match &app.sessions {
            SessionsData::Loaded(state) => state,
            SessionsData::Loading | SessionsData::Error(_) => panic!("sessions slot not loaded"),
        }
    }

    fn sessions_state_mut(app: &mut App) -> &mut SessionsState {
        match &mut app.sessions {
            SessionsData::Loaded(state) => state,
            SessionsData::Loading | SessionsData::Error(_) => panic!("sessions slot not loaded"),
        }
    }

    fn in_flight(app: &App, slot: &LoadSlot) -> bool {
        app.slot_busy(slot)
    }

    /// How many dispatches hold `slot`. Asserting on the count rather
    /// than on the channel keeps these tests deterministic — a
    /// dispatch that *was* made may not have reached `tx.send` yet
    /// when `try_recv` runs, so "nothing was sent" cannot distinguish
    /// suppression from a task that simply has not started.
    fn holders(app: &App, slot: &LoadSlot) -> u32 {
        app.in_flight.get(slot).map_or(0, |h| h.total())
    }

    fn forced_holders(app: &App, slot: &LoadSlot) -> u32 {
        app.in_flight.get(slot).map_or(0, |h| h.forced)
    }

    /// Wraps `handle_load_result` for call sites that don't care about
    /// generation staleness — stamps with the app's current
    /// generation, which is never considered stale.
    fn hlr_with_slot(app: &mut App, slot: LoadSlot, result: LoadResult) {
        let generation = app.ctx_generation;
        handle_load_result(
            app,
            StampedResult {
                generation,
                slot,
                guarded: false,
                result,
            },
        );
    }

    /// `hlr` with the slot derived from the result variant — the slot
    /// a real dispatch of that load would have acquired. `NoChange`
    /// and `RefreshFailed` carry no scope of their own, so they take
    /// `Sessions`: the outcome a guarded Sessions refresh produces,
    /// which is what every existing call site means by them.
    fn hlr(app: &mut App, result: LoadResult) {
        let slot = match &result {
            LoadResult::ShowDetail { session_id, .. } => LoadSlot::Show(session_id.clone()),
            LoadResult::InputsData { .. } => LoadSlot::Inputs,
            LoadResult::AgentsData { .. } => LoadSlot::Agents,
            LoadResult::Pricing { .. } => LoadSlot::Pricing,
            LoadResult::SessionsData { .. }
            | LoadResult::RefreshFailed { .. }
            | LoadResult::NoChange => LoadSlot::Sessions,
        };
        hlr_with_slot(app, slot, result);
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
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
    }

    #[test]
    fn app_new_no_selection_when_empty() {
        let app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        assert_eq!(sessions_state(&app).list_state.selected(), None);
    }

    #[test]
    fn app_new_computes_totals() {
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert_eq!(sessions_state(&app).total_tokens, 1500 + 2500 + 3000);
        let expected_cost = 0.01 + 0.02 + 0.03;
        assert!(
            (sessions_state(&app).total_cost.unwrap() - expected_cost).abs() < 1e-10,
            "expected {expected_cost}, got {:?}",
            sessions_state(&app).total_cost,
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
        sessions_state_mut(&mut app).list_state.select(Some(2));
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert_eq!(sessions_state(&app).list_state.selected(), Some(2));
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
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Down));
        assert_eq!(sessions_state(&app).list_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_up_retreats_selection() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        sessions_state_mut(&mut app).list_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Up));
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_j_advances_like_down() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::Char('j')));
        assert_eq!(sessions_state(&app).list_state.selected(), Some(1));
    }

    #[test]
    fn handle_key_k_retreats_like_up() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        sessions_state_mut(&mut app).list_state.select(Some(1));
        handle_key_event(&mut app, key_event(KeyCode::Char('k')));
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_home_selects_first() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        sessions_state_mut(&mut app).list_state.select(Some(2));
        handle_key_event(&mut app, key_event(KeyCode::Home));
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
    }

    #[test]
    fn handle_key_end_selects_last() {
        let sessions = fixture_sessions();
        let last = sessions.len() - 1;
        let mut app = new_app(sessions, fixture_pricing_data(), Tab::Sessions);
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
        handle_key_event(&mut app, key_event(KeyCode::End));
        render_app(&mut app, 80, 10);
        assert_eq!(sessions_state(&app).list_state.selected(), Some(last));
    }

    #[test]
    fn handle_key_unknown_is_noop() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('x')));
        assert!(!quit);
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
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

    /// `fixture_attribution_rows` deliberately includes one row with
    /// `attributed_cost: None`. Tests about the footer's *static* hint
    /// need rows that are all priced, since an unpriced row is a
    /// standing condition that outranks the key hints.
    fn priced_attribution_rows() -> Vec<AttributionRow> {
        let mut rows = fixture_attribution_rows();
        for row in &mut rows {
            row.attributed_cost = Some(0.001);
        }
        rows
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
            priced_attribution_rows(),
            fixture_coverage_stats(),
        )));
        app.tab = Tab::Inputs;
        let output = render_app(&mut app, 120, 10);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("1/2/3 tabs") && last_line.contains("q quit"),
            "footer should contain tab hint and quit; got: {last_line}",
        );
    }

    // --- Agents tab ---

    use crate::agents::PinningKind;
    use crate::domain::{CacheCreation, Usage};

    fn fixture_agent_rows() -> Vec<AgentRow> {
        vec![
            AgentRow {
                agent_type: "tw:code-reviewer".to_string(),
                model: Some("claude-opus-5".to_string()),
                effort: Some("high".to_string()),
                declared_effort: None,
                pinning: Pinning::Unpinned,
                dispatches: 3,
                usage: Usage {
                    input: 1000,
                    output: 0,
                    cache_creation: CacheCreation::default(),
                    cache_read: 0,
                },
                cost: Some(CostBreakdown {
                    input: 1.5,
                    ..CostBreakdown::default()
                }),
            },
            AgentRow {
                agent_type: "general-purpose".to_string(),
                model: None,
                effort: None,
                declared_effort: None,
                pinning: Pinning::NoAgentFile,
                dispatches: 1,
                usage: Usage {
                    input: 10,
                    output: 0,
                    cache_creation: CacheCreation::default(),
                    cache_read: 0,
                },
                cost: None,
            },
        ]
    }

    fn agents_app(rows: Vec<AgentRow>) -> App {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Agents);
        app.agents = Some(AgentsData::Loaded(AgentsState::new(rows)));
        app.tab = Tab::Agents;
        app
    }

    #[test]
    fn key_3_switches_to_agents_loading() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('3')));
        assert_eq!(app.tab, Tab::Agents);
        assert!(matches!(app.agents, Some(AgentsData::Loading)));
        assert_eq!(app.pending_loads.len(), 1);
        assert!(matches!(app.pending_loads[0], LoadRequest::AgentsRefresh));
    }

    #[test]
    fn key_3_reuses_cached_agents() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('3')));
        hlr(
            &mut app,
            LoadResult::AgentsData {
                result: Ok(fixture_agent_rows()),
                refresh_fingerprint: None,
            },
        );
        assert!(matches!(app.agents, Some(AgentsData::Loaded(_))));
        app.pending_loads.clear();
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
        handle_key_event(&mut app, key_event(KeyCode::Char('3')));
        assert!(matches!(app.agents, Some(AgentsData::Loaded(_))));
        assert!(
            app.pending_loads.is_empty(),
            "a loaded slot must not re-dispatch",
        );
    }

    #[test]
    fn app_starts_on_agents_tab_when_requested() {
        let app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Agents);
        assert_eq!(app.tab, Tab::Agents);
    }

    #[test]
    fn agents_tab_renders_table_columns() {
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 140, 12);
        // `declared` / `disp`, not the plain renderer's
        // `declared_effort` / `dispatches`: the TUI abbreviates both
        // to fit a width-constrained column.
        for column in [
            "agent", "model", "effort", "declared", "pinning", "disp", "tokens", "cost",
        ] {
            assert!(output.contains(column), "missing {column}; got:\n{output}");
        }
        assert!(output.contains("tw:code-reviewer"), "{output}");
    }

    #[test]
    fn agents_tab_renders_pinning_labels() {
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 140, 12);
        assert!(output.contains("unpinned"), "{output}");
        assert!(output.contains("no-agent-file"), "{output}");
    }

    #[test]
    fn agents_tab_renders_dash_for_unknown_cost() {
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 140, 12);
        assert!(
            output.contains('\u{2014}'),
            "an unpriced row renders an em dash; got:\n{output}",
        );
    }

    #[test]
    fn agents_tab_renders_footer_with_tab_hint() {
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 140, 12);
        let last_line = output.lines().last().unwrap();
        assert!(
            last_line.contains("1/2/3 tabs") && last_line.contains("q quit"),
            "footer should contain tab hint and quit; got: {last_line}",
        );
    }

    #[test]
    fn agents_tab_names_the_pinning_slice_above_the_footer() {
        // The narrowing default emits no filter component, so this
        // line is the only place a reader learns which kinds the
        // visible rows cover.
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 140, 12);
        assert!(
            output.contains("pinning: inherit,unpinned,no-agent-file"),
            "{output}",
        );
    }

    #[test]
    fn agents_empty_state_names_the_pinning_slice_at_the_default() {
        // A filtered-to-nothing Agents tab at the default slice must
        // still say which pinning kinds it covered.
        let mut app = agents_app(Vec::new());
        let output = render_app(&mut app, 140, 12);
        assert!(
            output.contains("pinning: inherit,unpinned,no-agent-file"),
            "{output}",
        );
        assert!(output.contains("Press a to widen"), "{output}");
    }

    #[test]
    fn tab_header_shows_agents_active() {
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 140, 12);
        let header = output.lines().next().unwrap();
        assert!(header.contains("[Agents]"), "got: {header}");
        assert!(!header.contains("[Sessions]"), "got: {header}");
    }

    #[test]
    fn tab_header_shows_agent_count() {
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 140, 12);
        let header = output.lines().next().unwrap();
        assert!(header.contains("2 agents"), "got: {header}");

        let mut one = agents_app(vec![fixture_agent_rows().remove(0)]);
        let output = render_app(&mut one, 140, 12);
        let header = output.lines().next().unwrap();
        assert!(header.contains("1 agent"), "singular; got: {header}");
    }

    #[test]
    fn agents_key_navigation_moves_selection() {
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Down));
        let Some(AgentsData::Loaded(state)) = &app.agents else {
            panic!("agents loaded");
        };
        assert_eq!(state.table_state.selected(), Some(1));
    }

    #[test]
    fn agents_key_unknown_is_noop() {
        let mut app = agents_app(fixture_agent_rows());
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('z')));
        assert!(!quit);
        let Some(AgentsData::Loaded(state)) = &app.agents else {
            panic!("agents loaded");
        };
        assert_eq!(state.table_state.selected(), Some(0));
    }

    #[test]
    fn agents_refresh_stores_the_fingerprint() {
        // Every sibling applier stores it. Without this, the guarded
        // tick keeps comparing against a stale fingerprint and reruns
        // a full `load_agents` every 3s forever.
        let mut app = agents_app(fixture_agent_rows());
        let mut fp = RefreshFingerprint::default();
        fp.entries
            .insert(PathBuf::from("/x.jsonl"), (1, std::time::UNIX_EPOCH));
        hlr(
            &mut app,
            LoadResult::AgentsData {
                result: Ok(fixture_agent_rows()),
                refresh_fingerprint: Some(fp.clone()),
            },
        );
        assert_eq!(app.refresh_fingerprint, fp);
    }

    #[test]
    fn agents_refresh_preserves_selection_by_row_key() {
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Down));
        // The selected row is `general-purpose`; a refresh that
        // reorders the rows must follow it rather than the index.
        let mut reordered = fixture_agent_rows();
        reordered.reverse();
        hlr(
            &mut app,
            LoadResult::AgentsData {
                result: Ok(reordered),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        let Some(AgentsData::Loaded(state)) = &app.agents else {
            panic!("agents loaded");
        };
        assert_eq!(state.table_state.selected(), Some(0));
        assert_eq!(state.rows[0].agent_type, "general-purpose");
    }

    #[test]
    fn agents_refresh_falls_back_when_the_selected_row_is_gone() {
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Down));
        hlr(
            &mut app,
            LoadResult::AgentsData {
                result: Ok(vec![fixture_agent_rows().remove(0)]),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        let Some(AgentsData::Loaded(state)) = &app.agents else {
            panic!("agents loaded");
        };
        assert_eq!(
            state.table_state.selected(),
            Some(0),
            "the index is clamped to the shorter list",
        );
    }

    #[test]
    fn agents_refresh_to_empty_clears_the_selection() {
        // A dangling index into an empty table is the one state a
        // renderer cannot draw.
        let mut app = agents_app(fixture_agent_rows());
        hlr(
            &mut app,
            LoadResult::AgentsData {
                result: Ok(Vec::new()),
                refresh_fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        let Some(AgentsData::Loaded(state)) = &app.agents else {
            panic!("agents loaded");
        };
        assert_eq!(state.table_state.selected(), None);
    }

    #[test]
    fn agents_table_stays_legible_at_eighty_columns() {
        // Eight columns cannot all be full width at 80, so the
        // question is which one wins. `agent` outweighs `model`
        // because a row whose agent name is unrecognizable cannot be
        // acted on, while a truncated model still reads as which
        // family it is.
        let mut app = agents_app(fixture_agent_rows());
        let output = render_app(&mut app, 80, 12);
        assert!(
            output.contains("tw:code-review"),
            "the agent cell must stay recognizable; got:\n{output}",
        );
        assert!(
            output.contains("general-purpos"),
            "and must distinguish the rows; got:\n{output}",
        );
        // The model cell yields first: it is truncated here while the
        // agent cell is not yet exhausted.
        assert!(
            !output.contains("claude-opus-5 "),
            "the model cell is the one that yields; got:\n{output}",
        );
        assert!(output.contains("no-agent-file"), "{output}");
    }

    #[test]
    fn a_restores_the_launch_slice_not_the_library_default() {
        // A run started with `--pinning pinned` must come back to
        // `pinned`, not to the default — the filter overlay has no
        // pinning field to recover it from.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Agents);
        let launch = PinningFilter::new(&[PinningKind::Pinned]);
        app.ctx.query.pinning = launch.clone();
        app.launch_pinning = launch.clone();
        app.agents = Some(AgentsData::Loaded(AgentsState::new(fixture_agent_rows())));

        handle_key_event(&mut app, key_event(KeyCode::Char('a')));
        assert_eq!(app.ctx.query.pinning, PinningFilter::everything());
        hlr(
            &mut app,
            LoadResult::AgentsData {
                result: Ok(fixture_agent_rows()),
                refresh_fingerprint: None,
            },
        );
        handle_key_event(&mut app, key_event(KeyCode::Char('a')));
        assert_eq!(app.ctx.query.pinning, launch);
    }

    #[test]
    fn compare_overlay_seeds_its_target_from_compare_model() {
        let mut app = agents_app(fixture_agent_rows());
        let wanted = compare_targets(&app.ctx.catalog)
            .into_iter()
            .nth(1)
            .expect("the fixture catalog has two or more models");
        app.compare_model = Some(wanted.clone());
        handle_key_event(&mut app, key_event(KeyCode::Char('c')));
        let Some(Overlay::Compare(state)) = &app.overlay else {
            panic!("compare overlay not open");
        };
        assert_eq!(
            state.target(&app.ctx.catalog).as_deref(),
            Some(wanted.as_str())
        );
    }

    #[test]
    fn invalidate_data_reloads_a_visible_agents_tab() {
        let mut app = agents_app(fixture_agent_rows());
        invalidate_data(&mut app);
        assert!(matches!(app.agents, Some(AgentsData::Loading)));
        assert!(
            app.pending_loads
                .iter()
                .any(|r| matches!(r, LoadRequest::AgentsRefresh)),
            "a visible tab must re-dispatch",
        );
    }

    #[test]
    fn invalidate_data_clears_an_offscreen_agents_tab() {
        let mut app = agents_app(fixture_agent_rows());
        app.tab = Tab::Sessions;
        invalidate_data(&mut app);
        assert!(app.agents.is_none(), "an offscreen slot is cleared");
        assert!(
            !app.pending_loads
                .iter()
                .any(|r| matches!(r, LoadRequest::AgentsRefresh)),
            "and must not re-dispatch",
        );
    }

    #[test]
    fn a_widens_pinning_to_every_kind() {
        let mut app = agents_app(fixture_agent_rows());
        let before = app.ctx_generation;
        handle_key_event(&mut app, key_event(KeyCode::Char('a')));
        assert_eq!(app.ctx.query.pinning, PinningFilter::everything());
        assert!(
            app.ctx_generation > before,
            "widening must bump the generation, or an in-flight load \
             computed against the old slice could overwrite the new one",
        );
        // Widening invalidates the slot, so the tab sits in
        // `Loading` until the reload lands — `a` is a loaded-state
        // binding and must wait for it, as `handle_agents_key`'s
        // dispatch order says.
        assert!(matches!(app.agents, Some(AgentsData::Loading)));
        hlr(
            &mut app,
            LoadResult::AgentsData {
                result: Ok(fixture_agent_rows()),
                refresh_fingerprint: None,
            },
        );
        handle_key_event(&mut app, key_event(KeyCode::Char('a')));
        assert_eq!(
            app.ctx.query.pinning,
            PinningFilter::default(),
            "a second press narrows back",
        );
    }

    #[test]
    fn c_opens_compare_overlay_from_agents_tab() {
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Char('c')));
        let Some(Overlay::Compare(state)) = &app.overlay else {
            panic!("compare overlay not open");
        };
        assert_eq!(state.agent_type, "tw:code-reviewer");
        assert_eq!(state.rows.len(), 1, "only the selected agent's rows");
    }

    #[test]
    fn esc_closes_compare_overlay() {
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Char('c')));
        assert!(app.overlay.is_some());
        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert!(app.overlay.is_none());
    }

    #[test]
    fn compare_overlay_swallows_unhandled_keys() {
        // No global binding may fire behind the overlay — matching
        // `handle_pricing_overlay_key`'s contract.
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Char('c')));
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
        assert_eq!(app.tab, Tab::Agents, "the tab binding must not fire");
        assert!(app.overlay.is_some(), "and the overlay stays open");
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        assert!(
            matches!(app.overlay, Some(Overlay::Compare(_))),
            "the filter overlay must not replace it",
        );
    }

    #[test]
    fn compare_overlay_switches_target_model() {
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Char('c')));
        let catalog = Arc::clone(&app.ctx.catalog);
        let first = match &app.overlay {
            Some(Overlay::Compare(s)) => s.target(&catalog),
            _ => panic!("compare overlay not open"),
        };
        handle_key_event(&mut app, key_event(KeyCode::Down));
        let second = match &app.overlay {
            Some(Overlay::Compare(s)) => s.target(&catalog),
            _ => panic!("compare overlay not open"),
        };
        assert_ne!(first, second, "the target must move");
    }

    #[test]
    fn filter_commit_preserves_a_non_default_pinning_slice() {
        // The regression the `to_query` base change exists to
        // prevent: the editor writes three of `Query`'s four fields,
        // and a commit must not reset the fourth.
        let mut app = agents_app(fixture_agent_rows());
        handle_key_event(&mut app, key_event(KeyCode::Char('a')));
        assert_eq!(app.ctx.query.pinning, PinningFilter::everything());

        app.overlay = Some(Overlay::Filter(Box::new(FilterEditor::from_query(
            &app.ctx.query,
        ))));
        commit_filter_edit(&mut app);
        assert_eq!(
            app.ctx.query.pinning,
            PinningFilter::everything(),
            "the committed slice must survive a filter commit",
        );
    }

    #[test]
    fn filter_commit_still_clears_an_emptied_editor_field() {
        // The other half of the base change: the three fields the
        // editor *does* own must not inherit `base`'s values, or
        // clearing a filter in the overlay would silently do nothing.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        app.overlay = Some(Overlay::Filter(Box::new(FilterEditor::from_query(
            &app.ctx.query,
        ))));
        // Clear every field through the editor's own key handling, so
        // the test exercises the path a user takes.
        for _ in 0..6 {
            for _ in 0..64 {
                handle_key_event(&mut app, key_event(KeyCode::Backspace));
            }
            handle_key_event(&mut app, key_event(KeyCode::Tab));
        }
        commit_filter_edit(&mut app);
        assert_eq!(app.ctx.query.sessions, SessionFilter::default());
        assert_eq!(app.ctx.query.thresholds, ThresholdsFilter::default());
        assert!(app.ctx.query.inputs_session_id.is_none());
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
            last_line.contains("1/2/3 tabs"),
            "list footer should contain '1/2/3 tabs'; got: {last_line}",
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
        let original_selected = sessions_state(&app).list_state.selected();
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
        assert_eq!(
            sessions_state(&app).list_state.selected(),
            original_selected
        );
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
        assert_eq!(app.pending_loads.len(), 1);
        assert!(matches!(app.pending_loads[0], LoadRequest::RefreshPricing));
    }

    #[test]
    fn r_in_overlay_pushes_even_while_a_pricing_load_is_in_flight() {
        // Suppression moved from the key handler to the dispatch
        // chokepoint: `r` always pushes, and `spawn_load` drops the
        // dispatch for an already-held slot. See
        // `occupied_slot_skips_dispatch` for the suppression itself.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.overlay = Some(Overlay::Pricing);
        assert!(app.try_reserve(LoadSlot::Pricing, false));
        handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert!(matches!(app.pending_loads[0], LoadRequest::RefreshPricing));
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
        assert!(app.try_reserve(LoadSlot::Pricing, false));
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
    }

    #[test]
    fn handle_load_result_pricing_error_keeps_old_data() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.try_reserve(LoadSlot::Pricing, false));
        let original_len = app.pricing.entries.len();
        hlr(
            &mut app,
            LoadResult::Pricing {
                result: Err(anyhow::anyhow!("network error")),
            },
        );
        assert_eq!(app.pricing.entries.len(), original_len);
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
        assert!(app.try_reserve(LoadSlot::Pricing, false));
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
                guarded: false,
                slot: LoadSlot::Show("aaa".to_string()),
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
    fn stale_generation_result_does_not_release_a_fresh_reservation() {
        // Inverts the rule the single in-flight boolean needed. A
        // generation bump already cleared every reservation it held,
        // so any `Sessions` entry standing now belongs to a
        // *replacement* dispatch — releasing it here would license a
        // duplicate load while that one is still running.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx_generation = 1;
        assert!(app.try_reserve(LoadSlot::Sessions, false));
        handle_load_result(
            &mut app,
            StampedResult {
                generation: 0,
                guarded: false,
                slot: LoadSlot::Sessions,
                result: LoadResult::NoChange,
            },
        );
        assert!(
            in_flight(&app, &LoadSlot::Sessions),
            "a stale result must not free a reservation held by a fresher dispatch",
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
                guarded: false,
                slot: LoadSlot::Show("aaa".to_string()),
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
                guarded: false,
                slot: LoadSlot::Sessions,
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
            kind: StatusKind::Error,
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
            kind: StatusKind::Error,
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
            kind: StatusKind::Error,
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
                guarded: false,
                slot: LoadSlot::Sessions,
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
        sessions_state_mut(&mut app).list_state.select(Some(1));
        assert_eq!(sessions_state(&app).selected_session_id(), Some("bbb"));
    }

    #[test]
    fn selected_session_id_none_when_empty() {
        let app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        assert_eq!(sessions_state(&app).selected_session_id(), None);
    }

    // --- apply_sessions_refresh tests ---

    #[test]
    fn apply_sessions_refresh_preserves_selection_by_id() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        sessions_state_mut(&mut app).list_state.select(Some(1)); // "bbb"

        let mut reordered = fixture_sessions();
        reordered.reverse(); // ccc, bbb, aaa
        sessions_state_mut(&mut app).apply_refresh(reordered);

        assert_eq!(sessions_state(&app).list_state.selected(), Some(1)); // "bbb" is now at index 1
    }

    #[test]
    fn apply_sessions_refresh_falls_back_on_removed_session() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        sessions_state_mut(&mut app).list_state.select(Some(2)); // "ccc"

        let mut shorter = fixture_sessions();
        shorter.retain(|s| s.id != "ccc");
        sessions_state_mut(&mut app).apply_refresh(shorter);

        // Old index 2 is clamped to new_len - 1 = 1
        assert_eq!(sessions_state(&app).list_state.selected(), Some(1));
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
        sessions_state_mut(&mut app).apply_refresh(updated);

        assert_eq!(sessions_state(&app).total_tokens, 5000 + 2500 + 3000);
        let expected = 0.05 + 0.02 + 0.03;
        assert!(
            (sessions_state(&app).total_cost.unwrap() - expected).abs() < 1e-10,
            "expected {expected}, got {:?}",
            sessions_state(&app).total_cost,
        );
    }

    // --- handle_load_result refresh tests ---

    #[test]
    fn handle_sessions_data_applies_in_list_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.try_reserve(LoadSlot::Sessions, false));
        let mut updated = fixture_sessions();
        updated[0].total_billable = 9999;
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Ok(updated),
                fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        assert_eq!(sessions_state(&app).sessions[0].total_billable, 9999);
    }

    #[test]
    fn handle_sessions_data_discarded_in_show_view() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        assert!(app.try_reserve(LoadSlot::Sessions, false));
        let original_count = sessions_state(&app).sessions.len();
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Ok(vec![]),
                fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        assert_eq!(sessions_state(&app).sessions.len(), original_count);
    }

    #[test]
    fn handle_no_change_clears_in_flight() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.try_reserve(LoadSlot::Sessions, false));
        hlr(&mut app, LoadResult::NoChange);
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
        assert!(app.try_reserve(LoadSlot::Show("aaa".to_string()), false));

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
    }

    #[test]
    fn refresh_show_holds_position_when_scrolled_up() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        // Select middle row (index 1)
        if let View::Show { table_state, .. } = &mut app.view {
            table_state.select(Some(1));
        }
        assert!(app.try_reserve(LoadSlot::Show("aaa".to_string()), false));

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
    }

    #[test]
    fn refresh_show_discarded_when_session_mismatch() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app); // viewing session "aaa"
        assert!(app.try_reserve(LoadSlot::Show("bbb".to_string()), false));

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
        assert!(app.try_reserve(LoadSlot::Inputs, false));

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
        assert!(app.try_reserve(LoadSlot::Inputs, false));

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
    }

    // --- Invalidation tests ---

    fn swap_catalog(app: &mut App) {
        apply_pricing(
            app,
            Ok((Arc::new(PricingCatalog::default()), fixture_pricing_data())),
        );
    }

    fn pushed_sessions_refreshes(app: &App) -> usize {
        app.pending_loads
            .iter()
            .filter(|r| {
                matches!(
                    r,
                    LoadRequest::Refresh {
                        scope: RefreshScope::Sessions,
                        guard: None,
                    }
                )
            })
            .count()
    }

    fn pushed_show_detail_for(app: &App, id: &str) -> bool {
        app.pending_loads
            .iter()
            .any(|r| matches!(r, LoadRequest::ShowDetail { session_id, .. } if session_id == id))
    }

    #[test]
    fn stale_applied_data_does_not_survive_a_swap() {
        // The first failure trace: an off-screen Sessions tab kept
        // rendering rows priced against the superseded catalog,
        // because `1` was a bare tab assignment with no re-dispatch.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));

        swap_catalog(&mut app);
        assert!(matches!(app.sessions, SessionsData::Loading));

        app.pending_loads.clear();
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
        assert_eq!(pushed_sessions_refreshes(&app), 1);
    }

    #[test]
    fn pending_show_is_renewed_not_stranded() {
        // The second failure trace: the discard dropped the in-flight
        // result and nothing re-dispatched, so `ShowLoading` hung.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "\"First session\" (alpha)".to_string(),
        };

        swap_catalog(&mut app);

        assert!(
            matches!(&app.view, View::ShowLoading { session_id, .. } if session_id == "aaa"),
            "the pending view is renewed, not abandoned",
        );
        assert!(pushed_show_detail_for(&app, "aaa"));

        // The doomed generation-0 result now lands.
        assert!(app.try_reserve(LoadSlot::Show("aaa".to_string()), false));
        handle_load_result(
            &mut app,
            StampedResult {
                generation: 0,
                guarded: false,
                slot: LoadSlot::Show("aaa".to_string()),
                result: LoadResult::ShowDetail {
                    session_id: "aaa".to_string(),
                    header_label: "test".to_string(),
                    result: Ok(fixture_prepared_exchanges()),
                    refresh_fingerprint: None,
                    is_refresh: false,
                },
            },
        );
        assert!(matches!(app.view, View::ShowLoading { .. }));
        assert!(
            in_flight(&app, &LoadSlot::Show("aaa".to_string())),
            "the renewal's reservation survives the stale result",
        );
    }

    #[test]
    fn show_view_is_renewed_on_swap() {
        // `View::Show` is applied data with costs baked into
        // `prepared`; leaving it alone reproduces the first trace.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);

        swap_catalog(&mut app);

        assert!(
            matches!(&app.view, View::ShowLoading { session_id, .. } if session_id == "aaa"),
            "a loaded Show view must be reloaded, not kept",
        );
        assert!(pushed_show_detail_for(&app, "aaa"));
    }

    #[test]
    fn show_error_is_retried_on_swap() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.view = View::ShowError {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
            message: "load failed".to_string(),
        };

        swap_catalog(&mut app);

        assert!(matches!(app.view, View::ShowLoading { .. }));
        assert!(pushed_show_detail_for(&app, "aaa"));
    }

    #[test]
    fn visible_inputs_tab_reloads_while_loading() {
        // The hole a `current_scope`-driven invalidator leaves: it
        // reports `None` here, so the tab would be cleared to `None`
        // with no dispatch and `render_inputs_content` would paint
        // nothing, with no navigation event able to recover it.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loading);

        swap_catalog(&mut app);

        assert!(matches!(app.inputs, Some(InputsData::Loading)));
        assert!(
            app.pending_loads
                .iter()
                .any(|r| matches!(r, LoadRequest::InputsRefresh))
        );
    }

    #[test]
    fn visible_inputs_tab_reloads_while_errored() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Error("boom".to_string()));

        swap_catalog(&mut app);

        assert!(matches!(app.inputs, Some(InputsData::Loading)));
        assert!(
            app.pending_loads
                .iter()
                .any(|r| matches!(r, LoadRequest::InputsRefresh))
        );
    }

    #[test]
    fn offscreen_inputs_is_cleared_and_refills_on_navigation() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));

        swap_catalog(&mut app);
        assert!(app.inputs.is_none(), "an off-screen slot is cleared");

        app.pending_loads.clear();
        handle_key_event(&mut app, key_event(KeyCode::Char('2')));
        assert!(matches!(app.inputs, Some(InputsData::Loading)));
        assert!(
            app.pending_loads
                .iter()
                .any(|r| matches!(r, LoadRequest::InputsRefresh))
        );
    }

    #[test]
    fn escaping_show_after_invalidation_reloads_sessions() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);

        swap_catalog(&mut app);
        app.pending_loads.clear();

        handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert!(matches!(app.view, View::List));
        assert_eq!(pushed_sessions_refreshes(&app), 1);
    }

    #[test]
    fn swap_from_list_pushes_exactly_one_sessions_refresh() {
        // Guards against double-dispatch between the view renewal and
        // the tab branch.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        swap_catalog(&mut app);
        assert_eq!(pushed_sessions_refreshes(&app), 1);
        assert_eq!(app.pending_loads.len(), 1);
    }

    #[test]
    fn pricing_failure_does_not_invalidate() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        set_show_view(&mut app);
        let generation = app.ctx_generation;

        apply_pricing(&mut app, Err(anyhow::anyhow!("network error")));

        assert_eq!(app.ctx_generation, generation);
        assert!(matches!(app.sessions, SessionsData::Loaded(_)));
        assert!(matches!(app.inputs, Some(InputsData::Loaded(_))));
        assert!(matches!(app.view, View::Show { .. }));
        assert!(app.pending_loads.is_empty());
    }

    #[test]
    fn failed_reload_after_invalidation_is_retryable() {
        // Without an error state this leaves a permanent "Loading
        // sessions..." screen: the timer's guarded ticks answer
        // `NoChange` against an unchanged filesystem forever.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        swap_catalog(&mut app);
        assert!(matches!(app.sessions, SessionsData::Loading));

        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Err(anyhow::anyhow!("boom")),
                fingerprint: None,
            },
        );
        assert!(matches!(app.sessions, SessionsData::Error(_)));
        assert!(app.status.is_some());

        app.pending_loads.clear();
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert_eq!(pushed_sessions_refreshes(&app), 1);
        assert!(matches!(app.sessions, SessionsData::Loading));
    }

    #[test]
    fn tab_switch_during_a_forced_load_self_heals() {
        // The forced result lands while the user is on Inputs, where
        // `apply_sessions_data` discards it — so the Sessions tab must
        // re-dispatch when they come back, or it stays empty.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        swap_catalog(&mut app);

        handle_key_event(&mut app, key_event(KeyCode::Char('2')));
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Ok(fixture_sessions()),
                fingerprint: None,
            },
        );
        assert!(
            matches!(app.sessions, SessionsData::Loading),
            "the result is discarded off-tab",
        );

        app.pending_loads.clear();
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
        assert_eq!(pushed_sessions_refreshes(&app), 1);
    }

    #[tokio::test]
    async fn repeated_tab_switches_do_not_queue_duplicate_sessions_loads() {
        // Both of `ensure_sessions_data`'s dispatches are unguarded,
        // which the ledger never suppresses — so without its own
        // check, key-repeat on `1` during a post-swap reload queues a
        // full `load_sessions` walk per keypress.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, _rx) = load_channel();
        swap_catalog(&mut app);
        drain_pending_loads(&mut app, &tx);
        assert_eq!(holders(&app, &LoadSlot::Sessions), 1);

        for _ in 0..4 {
            handle_key_event(&mut app, key_event(KeyCode::Char('1')));
            drain_pending_loads(&mut app, &tx);
        }
        assert_eq!(holders(&app, &LoadSlot::Sessions), 1);
    }

    #[tokio::test]
    async fn repeated_tab_switches_do_not_queue_duplicate_show_loads() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, _rx) = load_channel();
        app.view = View::ShowLoading {
            session_id: "aaa".to_string(),
            header_label: "test".to_string(),
        };
        for _ in 0..3 {
            handle_key_event(&mut app, key_event(KeyCode::Char('1')));
            drain_pending_loads(&mut app, &tx);
        }
        assert_eq!(holders(&app, &LoadSlot::Show("aaa".to_string())), 1);
    }

    #[test]
    fn show_view_on_the_inputs_tab_renews_and_refills_on_navigation() {
        // `try_switch_to_inputs` leaves `app.view` alone, so
        // (Tab::Inputs, View::Show) is reachable. Invalidation renews
        // that view but dispatches nothing for it — the Sessions
        // branch is gated on `app.tab` — so the renewal is only made
        // good when the user navigates back.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Char('2')));
        assert_eq!(app.tab, Tab::Inputs);

        swap_catalog(&mut app);
        assert!(
            matches!(app.view, View::ShowLoading { .. }),
            "the off-screen Show view is still renewed",
        );
        assert!(
            !pushed_show_detail_for(&app, "aaa"),
            "but nothing is dispatched for an off-screen tab",
        );

        app.pending_loads.clear();
        handle_key_event(&mut app, key_event(KeyCode::Char('1')));
        assert!(pushed_show_detail_for(&app, "aaa"));
    }

    // --- In-flight ledger tests ---

    /// A channel whose receiver is dropped would make `spawn_load`'s
    /// send fail silently, so tests hold both ends.
    fn load_channel() -> (
        mpsc::UnboundedSender<StampedResult>,
        mpsc::UnboundedReceiver<StampedResult>,
    ) {
        mpsc::unbounded_channel()
    }

    #[test]
    fn bump_ctx_generation_clears_the_ledger() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.try_reserve(LoadSlot::Sessions, false));
        assert!(app.try_reserve(LoadSlot::Inputs, false));
        assert!(app.try_reserve(LoadSlot::Show("aaa".to_string()), false));
        let before = app.ctx_generation;

        app.bump_ctx_generation();

        assert!(app.in_flight.is_empty());
        assert_eq!(app.ctx_generation, before + 1);
    }

    #[tokio::test]
    async fn dispatch_acquires_its_slot() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, _rx) = load_channel();
        app.pending_loads.push(LoadRequest::Refresh {
            scope: RefreshScope::Sessions,
            guard: None,
        });
        drain_pending_loads(&mut app, &tx);
        assert!(in_flight(&app, &LoadSlot::Sessions));
    }

    #[tokio::test]
    async fn guarded_dispatch_yields_to_an_occupied_slot() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, mut rx) = load_channel();
        assert!(app.try_reserve(LoadSlot::Sessions, false));
        app.pending_loads.push(LoadRequest::Refresh {
            scope: RefreshScope::Sessions,
            guard: Some(RefreshFingerprint::default()),
        });
        drain_pending_loads(&mut app, &tx);
        assert_eq!(
            holders(&app, &LoadSlot::Sessions),
            1,
            "a guarded dispatch must not add a second equivalent load",
        );
        assert!(rx.try_recv().is_err(), "and must not have spawned at all");
    }

    #[tokio::test]
    async fn unguarded_dispatch_is_not_suppressed_by_a_guarded_load() {
        // A guarded load can answer `NoChange`, which satisfies no
        // waiting view — so it must never stand in for a load somebody
        // is waiting on.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, _rx) = load_channel();
        assert!(app.try_reserve(LoadSlot::Sessions, true));
        app.pending_loads.push(LoadRequest::Refresh {
            scope: RefreshScope::Sessions,
            guard: None,
        });
        drain_pending_loads(&mut app, &tx);
        assert_eq!(
            holders(&app, &LoadSlot::Sessions),
            2,
            "the forced load runs alongside the guarded one",
        );
    }

    #[tokio::test]
    async fn guarded_agents_dispatch_yields_to_a_running_load() {
        // Asserted on the reservation count, which is settled by the
        // time `drain_pending_loads` returns — an empty result channel
        // would pass both when the dispatch was suppressed and when it
        // merely has not reached its `send`.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Agents);
        let (tx, _rx) = load_channel();
        assert!(app.try_reserve(LoadSlot::Agents, false));
        app.pending_loads.push(LoadRequest::Refresh {
            scope: RefreshScope::Agents,
            guard: Some(RefreshFingerprint::default()),
        });
        drain_pending_loads(&mut app, &tx);
        assert_eq!(
            holders(&app, &LoadSlot::Agents),
            1,
            "a guarded dispatch must not add a second equivalent load",
        );
    }

    #[tokio::test]
    async fn unguarded_agents_dispatch_is_not_suppressed_by_a_guarded_one() {
        // A guarded load can answer `NoChange`, which no applier turns
        // into data — so it answers nobody, and suppressing the
        // unguarded dispatch behind it would strand the tab in
        // `Loading` forever.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Agents);
        let (tx, _rx) = load_channel();
        assert!(app.try_reserve(LoadSlot::Agents, true));
        app.pending_loads.push(LoadRequest::AgentsRefresh);
        drain_pending_loads(&mut app, &tx);
        assert_eq!(
            holders(&app, &LoadSlot::Agents),
            2,
            "the forced load runs alongside the guarded one",
        );
    }

    #[tokio::test]
    async fn stale_generation_agents_result_releases_nothing() {
        // `bump_ctx_generation` already dropped this result's
        // reservation; releasing by key on arrival would free a
        // *fresh* dispatch's slot and license a duplicate load.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Agents);
        app.bump_ctx_generation();
        assert!(app.try_reserve(LoadSlot::Agents, false));
        assert_eq!(holders(&app, &LoadSlot::Agents), 1);
        handle_load_result(
            &mut app,
            StampedResult {
                generation: 0,
                slot: LoadSlot::Agents,
                guarded: false,
                result: LoadResult::AgentsData {
                    result: Ok(fixture_agent_rows()),
                    refresh_fingerprint: None,
                },
            },
        );
        assert_eq!(
            holders(&app, &LoadSlot::Agents),
            1,
            "the fresh reservation must survive a stale result",
        );
    }

    #[tokio::test]
    async fn unguarded_dispatch_yields_to_another_unguarded_load() {
        // That one *will* produce a result an applier consumes, so a
        // second is pure duplicated work.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, _rx) = load_channel();
        assert!(app.try_reserve(LoadSlot::Sessions, false));
        app.pending_loads.push(LoadRequest::Refresh {
            scope: RefreshScope::Sessions,
            guard: None,
        });
        drain_pending_loads(&mut app, &tx);
        assert_eq!(holders(&app, &LoadSlot::Sessions), 1);
    }

    #[tokio::test]
    async fn opening_a_session_is_not_suppressed_by_its_guarded_refresh() {
        // Regression: a guarded `Show` refresh for "aaa" runs while
        // the user leaves Show and re-opens the same session. If the
        // ledger dropped that open, the guarded result — `is_refresh`,
        // so no applier accepts it into `ShowLoading`, and possibly
        // `NoChange` carrying nothing at all — would leave the view on
        // "Loading session..." with `current_scope` reporting `None`,
        // so no tick would ever retry it.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, _rx) = load_channel();
        assert!(app.try_reserve(LoadSlot::Show("aaa".to_string()), true));

        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(matches!(app.view, View::ShowLoading { .. }));
        drain_pending_loads(&mut app, &tx);

        assert_eq!(
            holders(&app, &LoadSlot::Show("aaa".to_string())),
            2,
            "the user's open must dispatch alongside the guarded refresh",
        );
    }

    #[test]
    fn a_shared_slot_frees_only_when_every_holder_releases() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.try_reserve(LoadSlot::Sessions, true));
        assert!(app.try_reserve(LoadSlot::Sessions, false));

        app.release_slot(&LoadSlot::Sessions, true);
        assert!(
            in_flight(&app, &LoadSlot::Sessions),
            "the first result must not free the slot the second load still holds",
        );

        app.release_slot(&LoadSlot::Sessions, false);
        assert!(!in_flight(&app, &LoadSlot::Sessions));
    }

    #[test]
    fn releasing_the_wrong_holder_kind_does_not_free_a_forced_load() {
        // Releasing by count alone would free the forced holder when a
        // guarded load finished, and a later forced dispatch would
        // then be suppressed by a load that cannot answer for it.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.try_reserve(LoadSlot::Sessions, false));

        app.release_slot(&LoadSlot::Sessions, true);

        assert_eq!(
            forced_holders(&app, &LoadSlot::Sessions),
            1,
            "a guarded release must leave the forced holder standing",
        );
    }

    #[test]
    fn releasing_an_unheld_slot_is_a_noop() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.release_slot(&LoadSlot::Sessions, false);
        assert!(app.in_flight.is_empty());
    }

    #[tokio::test]
    async fn show_slots_are_keyed_by_session_id() {
        // Opening B while A's load runs is two loads, not a duplicate.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let (tx, _rx) = load_channel();
        assert!(app.try_reserve(LoadSlot::Show("aaa".to_string()), false));
        app.pending_loads.push(LoadRequest::ShowDetail {
            session_id: "bbb".to_string(),
            header_label: "test".to_string(),
        });
        drain_pending_loads(&mut app, &tx);
        assert!(in_flight(&app, &LoadSlot::Show("bbb".to_string())));
        assert!(in_flight(&app, &LoadSlot::Show("aaa".to_string())));
    }

    #[test]
    fn backpressure_trace_survives_a_generation_bump() {
        // The trace this phase exists to fix: a guarded Sessions load
        // is in flight at generation 0 when the catalog swaps. The
        // bump drops its reservation, the replacement dispatch takes
        // the slot, and the generation-0 result must not free it —
        // doing so would let the very next timer tick start a second
        // guarded load alongside the one still running.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(app.try_reserve(LoadSlot::Sessions, false));

        app.bump_ctx_generation();
        assert!(app.try_reserve(LoadSlot::Sessions, false));

        handle_load_result(
            &mut app,
            StampedResult {
                generation: 0,
                guarded: false,
                slot: LoadSlot::Sessions,
                result: LoadResult::NoChange,
            },
        );

        assert!(
            in_flight(&app, &LoadSlot::Sessions),
            "the replacement dispatch keeps its reservation",
        );
    }

    #[test]
    fn every_result_kind_releases_its_slot() {
        let cases: Vec<(LoadSlot, LoadResult)> = vec![
            (
                LoadSlot::Sessions,
                LoadResult::SessionsData {
                    result: Ok(fixture_sessions()),
                    fingerprint: Some(RefreshFingerprint::default()),
                },
            ),
            (
                LoadSlot::Inputs,
                LoadResult::InputsData {
                    result: Ok((fixture_attribution_rows(), fixture_coverage_stats())),
                    refresh_fingerprint: None,
                },
            ),
            (
                LoadSlot::Show("aaa".to_string()),
                LoadResult::ShowDetail {
                    session_id: "aaa".to_string(),
                    header_label: "test".to_string(),
                    result: Ok(fixture_prepared_exchanges()),
                    refresh_fingerprint: None,
                    is_refresh: false,
                },
            ),
            (
                LoadSlot::Pricing,
                LoadResult::Pricing {
                    result: Ok((Arc::new(PricingCatalog::default()), fixture_pricing_data())),
                },
            ),
            (LoadSlot::Sessions, LoadResult::NoChange),
            (
                LoadSlot::Sessions,
                LoadResult::RefreshFailed {
                    message: "boom".to_string(),
                },
            ),
        ];
        for (slot, result) in cases {
            let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
            assert!(app.try_reserve(slot.clone(), false));
            hlr_with_slot(&mut app, slot.clone(), result);
            assert!(
                !in_flight(&app, &slot),
                "a current-generation result must release {slot:?}",
            );
        }
    }

    // --- Sessions slot state tests ---

    #[test]
    fn try_open_show_while_loading_is_noop() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(app.pending_loads.is_empty());
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn try_open_show_with_out_of_range_selection_is_noop() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        // A selection retained past the end of a shortened list — the
        // shape that used to panic on `app.sessions[idx]`.
        let state = sessions_state_mut(&mut app);
        state.sessions.truncate(1);
        state.list_state.select(Some(2));
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(app.pending_loads.is_empty());
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn loading_sessions_renders_feedback() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        let output = render_app(&mut app, 80, 10);
        assert!(
            output.contains("Loading sessions..."),
            "should render the loading message; got:\n{output}",
        );
        let first_line = output.lines().next().unwrap();
        assert!(
            first_line.contains("cclens — ..."),
            "header should show '...' while loading; got: {first_line}",
        );
    }

    #[test]
    fn errored_sessions_renders_message_and_retry_hint() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Error("boom".to_string());
        let output = render_app(&mut app, 80, 10);
        assert!(
            output.contains("Error: boom"),
            "should render the error message; got:\n{output}",
        );
        assert!(
            output.contains("Enter retry"),
            "should render the retry hint; got:\n{output}",
        );
    }

    #[test]
    fn sessions_error_retries_on_enter() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Error("boom".to_string());
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(matches!(
            app.pending_loads[0],
            LoadRequest::Refresh {
                scope: RefreshScope::Sessions,
                guard: None,
            }
        ));
        assert!(
            matches!(app.sessions, SessionsData::Loading),
            "the retry must be visible, matching Show's and Inputs' error retries",
        );
    }

    #[test]
    fn sessions_error_retries_on_r() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Error("boom".to_string());
        handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert!(matches!(
            app.pending_loads[0],
            LoadRequest::Refresh {
                scope: RefreshScope::Sessions,
                guard: None,
            }
        ));
        assert!(matches!(app.sessions, SessionsData::Loading));
    }

    #[test]
    fn loading_sessions_ignores_navigation_keys() {
        // The point of splitting `handle_list_key` is that a slot with
        // no rows cannot be navigated — there is no selection to move.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        for code in [
            KeyCode::Down,
            KeyCode::Up,
            KeyCode::Char('j'),
            KeyCode::Char('k'),
            KeyCode::Char('G'),
            KeyCode::Enter,
        ] {
            assert!(!handle_key_event(&mut app, key_event(code)));
        }
        assert!(app.pending_loads.is_empty());
        assert!(matches!(app.sessions, SessionsData::Loading));
        assert!(matches!(app.view, View::List));
    }

    #[test]
    fn loading_sessions_still_quits_on_q() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        assert!(handle_key_event(&mut app, key_event(KeyCode::Char('q'))));
    }

    #[test]
    fn errored_sessions_ignores_navigation_keys() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Error("boom".to_string());
        for code in [
            KeyCode::Down,
            KeyCode::Up,
            KeyCode::Char('j'),
            KeyCode::Char('k'),
            KeyCode::Char('G'),
        ] {
            assert!(!handle_key_event(&mut app, key_event(code)));
        }
        assert!(
            app.pending_loads.is_empty(),
            "only Enter/r retry from the error state",
        );
    }

    #[test]
    fn errored_sessions_still_quits_on_q() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Error("boom".to_string());
        assert!(handle_key_event(&mut app, key_event(KeyCode::Char('q'))));
    }

    #[test]
    fn sessions_load_failure_into_empty_slot_sets_error() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Err(anyhow::anyhow!("boom")),
                fingerprint: None,
            },
        );
        assert!(matches!(app.sessions, SessionsData::Error(_)));
        assert!(
            app.status.is_some(),
            "a failed load into an empty slot must also reach the status footer",
        );
    }

    #[test]
    fn sessions_load_failure_preserves_loaded_rows() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Err(anyhow::anyhow!("boom")),
                fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        assert!(
            matches!(app.sessions, SessionsData::Loaded(_)),
            "a failed background refresh must not discard rows the user is reading",
        );
        assert_eq!(sessions_state(&app).sessions.len(), 3);
        assert!(
            app.status.is_none(),
            "one transient background failure stays silent, like every other refresh failure",
        );
        assert_eq!(app.consecutive_refresh_failures, 1);
    }

    #[test]
    fn repeated_sessions_load_failures_break_silence_and_clear_on_no_change() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        for _ in 0..CONSECUTIVE_REFRESH_FAILURE_THRESHOLD {
            hlr(
                &mut app,
                LoadResult::SessionsData {
                    result: Err(anyhow::anyhow!("boom")),
                    fingerprint: Some(RefreshFingerprint::default()),
                },
            );
        }
        assert!(
            app.status.is_some(),
            "a persistent background failure must eventually speak up",
        );

        // Stamped `from_refresh_failure`, so the same signal that
        // proves the pipeline healthy can retire the warning.
        hlr(&mut app, LoadResult::NoChange);
        assert!(
            app.status.is_none(),
            "NoChange must clear a status this pipeline itself raised",
        );
        assert_eq!(app.consecutive_refresh_failures, 0);
    }

    #[test]
    fn sessions_data_into_empty_slot_constructs_state() {
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        hlr(
            &mut app,
            LoadResult::SessionsData {
                result: Ok(fixture_sessions()),
                fingerprint: Some(RefreshFingerprint::default()),
            },
        );
        assert_eq!(sessions_state(&app).sessions.len(), 3);
        assert_eq!(sessions_state(&app).list_state.selected(), Some(0));
    }

    // --- Filter editor: opening ---

    fn filter_editor(app: &App) -> &FilterEditor {
        match &app.overlay {
            Some(Overlay::Filter(editor)) => editor,
            Some(Overlay::Pricing | Overlay::Compare(_)) | None => {
                panic!("filter overlay not open")
            }
        }
    }

    fn populated_query() -> Query {
        Query {
            sessions: SessionFilter {
                project_name: Some("alpha".to_string()),
                // The instant that renders as a bare date is the
                // local day start, not midnight UTC.
                since: parse_filter_datetime("2026-04-10").ok(),
                until: Some(ts("2026-04-20T14:33:00Z")),
            },
            thresholds: ThresholdsFilter {
                min_tokens: Some(50_000),
                min_cost: Some(0.5),
            },
            inputs_session_id: Some("aaaa1111-2222-3333-4444-555555555555".to_string()),
            pinning: PinningFilter::default(),
        }
    }

    fn ts(s: &str) -> DateTime<Utc> {
        s.parse::<DateTime<Utc>>().unwrap()
    }

    #[test]
    fn f_opens_filter_overlay_from_every_state() {
        // The binding is global precisely so it works from a view the
        // user filtered into a dead end — including one still loading.
        let mut list = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut list, key_event(KeyCode::Char('f')));
        assert!(matches!(list.overlay, Some(Overlay::Filter(_))));

        let mut show = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut show);
        handle_key_event(&mut show, key_event(KeyCode::Char('f')));
        assert!(matches!(show.overlay, Some(Overlay::Filter(_))));

        let mut inputs = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        inputs.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        handle_key_event(&mut inputs, key_event(KeyCode::Char('f')));
        assert!(matches!(inputs.overlay, Some(Overlay::Filter(_))));

        let mut loading = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        loading.sessions = SessionsData::Loading;
        handle_key_event(&mut loading, key_event(KeyCode::Char('f')));
        assert!(matches!(loading.overlay, Some(Overlay::Filter(_))));
    }

    #[test]
    fn filter_editor_seeds_every_field_and_round_trips_to_the_same_query() {
        let query = populated_query();
        let editor = FilterEditor::from_query(&query);
        assert!(
            editor.fields.iter().all(|f| f.parsed.is_ok()),
            "a seeded editor must start entirely Ok",
        );
        assert!(editor.is_committable());
        assert_eq!(editor.to_query(&query), query);
    }

    #[test]
    fn filter_editor_seeds_midnight_since_as_bare_date() {
        // The shortest spelling that reparses to the same instant —
        // otherwise the editor shows a value the user did not type.
        let query = Query {
            sessions: SessionFilter {
                since: parse_filter_datetime("2026-04-10").ok(),
                ..Default::default()
            },
            ..Default::default()
        };
        let editor = FilterEditor::from_query(&query);
        let since = &editor.fields[2];
        assert_eq!(since.kind, FilterFieldKind::Since);
        assert_eq!(since.input, "2026-04-10");
    }

    #[test]
    fn filter_editor_seeds_non_midnight_until_as_rfc3339() {
        let editor = FilterEditor::from_query(&populated_query());
        let until = &editor.fields[3];
        assert_eq!(until.kind, FilterFieldKind::Until);
        // Asserted as a property, not a fixed spelling: the offset is
        // the local one, which differs per zone the test runs in.
        assert!(
            until.input.starts_with("2026-04-20T"),
            "got: {}",
            until.input
        );
        assert_eq!(
            parse_filter_datetime(&until.input).ok(),
            populated_query().sessions.until,
        );
    }

    // --- Filter editor: key handling ---

    #[test]
    fn typing_q_into_project_field_appends_rather_than_quitting() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        // The editor opens focused on `project`, the first field that
        // accepts text.
        let quit = handle_key_event(&mut app, key_event(KeyCode::Char('q')));
        assert!(!quit, "q must not quit while the editor owns the keyboard");
        assert!(matches!(app.overlay, Some(Overlay::Filter(_))));
        assert_eq!(filter_editor(&app).fields[1].input, "q");
    }

    #[test]
    fn overlay_swallows_pricing_and_refresh_keys() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for c in ['p', 'r'] {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        assert!(matches!(app.overlay, Some(Overlay::Filter(_))));
        assert!(app.pending_loads.is_empty(), "r must not trigger a refresh");
        assert_eq!(filter_editor(&app).fields[1].input, "pr");
    }

    #[test]
    fn backspace_pops_one_character_from_a_text_field() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        handle_key_event(&mut app, key_event(KeyCode::Backspace));
        assert_eq!(filter_editor(&app).fields[1].input, "alph");
    }

    #[test]
    fn backspace_clears_the_whole_session_field_in_one_press() {
        // One key means "remove" throughout the editor; on the
        // clear-only field it removes everything.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        handle_key_event(&mut app, key_event(KeyCode::BackTab)); // -> session
        handle_key_event(&mut app, key_event(KeyCode::Backspace));
        assert_eq!(filter_editor(&app).fields[0].input, "");
    }

    #[test]
    fn char_input_into_the_session_field_is_ignored() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        handle_key_event(&mut app, key_event(KeyCode::BackTab)); // -> session
        handle_key_event(&mut app, key_event(KeyCode::Char('x')));
        assert_eq!(
            filter_editor(&app).fields[0].input,
            "aaaa1111-2222-3333-4444-555555555555",
        );
    }

    #[test]
    fn tab_and_backtab_move_focus_and_wrap_at_both_ends() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        assert_eq!(
            filter_editor(&app).focused,
            1,
            "opens on the first text field"
        );

        for expected in 2..=5 {
            handle_key_event(&mut app, key_event(KeyCode::Tab));
            assert_eq!(filter_editor(&app).focused, expected);
        }
        handle_key_event(&mut app, key_event(KeyCode::Tab));
        assert_eq!(filter_editor(&app).focused, 0, "Tab wraps forward");

        handle_key_event(&mut app, key_event(KeyCode::BackTab));
        assert_eq!(filter_editor(&app).focused, 5, "BackTab wraps backward");
    }

    // --- Filter editor: commit ---

    #[test]
    fn enter_is_inert_while_a_field_fails_to_parse() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let before = app.ctx_generation;
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for _ in 0..4 {
            handle_key_event(&mut app, key_event(KeyCode::Tab)); // project -> min-cost
        }
        for c in ['a', 'b', 'c'] {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        assert!(!filter_editor(&app).is_committable());

        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(
            matches!(app.overlay, Some(Overlay::Filter(_))),
            "the overlay stays open so the error text remains visible",
        );
        assert_eq!(app.ctx.query, Query::default());
        assert_eq!(app.ctx_generation, before);
    }

    #[test]
    fn enter_commits_the_query_and_reloads_the_visible_tab() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let before = app.ctx_generation;
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for c in "beta".chars() {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        handle_key_event(&mut app, key_event(KeyCode::Enter));

        assert!(app.overlay.is_none());
        assert_eq!(
            app.ctx.query.sessions.project_name,
            Some("beta".to_string()),
        );
        // `invalidate_data` owns the bump — exactly one, not two.
        assert_eq!(app.ctx_generation, before + 1);
        assert!(matches!(app.sessions, SessionsData::Loading));
        assert_eq!(app.pending_loads.len(), 1);
    }

    #[test]
    fn commit_from_show_renews_the_view_and_dispatches_show_detail() {
        // Not a `Refresh { scope: Show }` — that carries
        // `is_refresh: true`, which `apply_show_detail` refuses into
        // the `ShowLoading` invalidation has just installed.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        handle_key_event(&mut app, key_event(KeyCode::Enter));

        assert!(matches!(app.view, View::ShowLoading { .. }));
        assert_eq!(app.pending_loads.len(), 1);
        assert!(matches!(
            app.pending_loads[0],
            LoadRequest::ShowDetail { .. }
        ));
    }

    #[test]
    fn esc_discards_the_edit() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let before = app.ctx_generation;
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for c in "beta".chars() {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        handle_key_event(&mut app, key_event(KeyCode::Esc));

        assert!(app.overlay.is_none());
        assert_eq!(app.ctx.query, Query::default());
        assert_eq!(app.ctx_generation, before);
    }

    #[test]
    fn committing_an_unchanged_query_still_reloads() {
        // Pressing Enter always reloads. A contract that silently did
        // nothing when the query happened to match would be harder to
        // explain than one that always costs a reload.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        let before = app.ctx_generation;
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        handle_key_event(&mut app, key_event(KeyCode::Enter));

        assert_eq!(app.ctx.query, populated_query());
        assert_eq!(app.ctx_generation, before + 1);
        assert_eq!(app.pending_loads.len(), 1);
    }

    // --- Active-filter indicator ---

    #[test]
    fn fit_filter_components_keeps_whole_components_only() {
        let components = vec![
            FilterComponent {
                text: "--project alpha".to_string(),
                honored_by: HonoredBy::EVERY_LOADER,
            },
            FilterComponent {
                text: "--min-tokens 50000".to_string(),
                honored_by: HonoredBy::EVERY_LOADER,
            },
            FilterComponent {
                text: "--min-cost 0.5".to_string(),
                honored_by: HonoredBy::EVERY_LOADER,
            },
        ];

        let (fitted, dropped) = fit_filter_components(&components, 200);
        assert_eq!(fitted.len(), 3);
        assert_eq!(dropped, 0);

        // Room for the first component plus the ` +N` marker only.
        let (fitted, dropped) = fit_filter_components(&components, 20);
        assert_eq!(fitted.len(), 1);
        assert_eq!(dropped, 2);

        let (fitted, dropped) = fit_filter_components(&components, 3);
        assert!(fitted.is_empty());
        assert_eq!(dropped, 3);
    }

    #[test]
    fn fit_filter_components_measures_scalars_not_bytes() {
        // Each `é` is two bytes but one column. Measuring bytes would
        // drop a component that fits.
        let name = "é".repeat(10);
        let components = vec![FilterComponent {
            text: format!("--project {name}"),
            honored_by: HonoredBy::EVERY_LOADER,
        }];
        assert_eq!(components[0].text.len(), 30, "20 bytes of accented text");
        let (fitted, dropped) = fit_filter_components(&components, 20);
        assert_eq!(fitted.len(), 1, "10 scalars + `--project ` fits in 20");
        assert_eq!(dropped, 0);
    }

    #[test]
    fn header_names_active_filters_and_omits_the_marker_when_none() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let unfiltered = render_app(&mut app, 120, 10);
        assert!(
            !unfiltered.lines().next().unwrap().contains("filter:"),
            "an empty Query must render no marker",
        );

        app.ctx.query = Query {
            sessions: SessionFilter {
                project_name: Some("alpha".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let filtered = render_app(&mut app, 120, 10);
        let header = filtered.lines().next().unwrap();
        assert!(header.contains("filter:"), "got: {header}");
        assert!(header.contains("--project alpha"), "got: {header}");
    }

    #[test]
    fn header_keeps_an_indicator_at_eighty_columns() {
        // 80 columns is the common terminal width, and the width an
        // even left/right split starved: the whole marker vanished,
        // `+N` included, leaving a filtered view with nothing on
        // screen to say so.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        let output = render_app(&mut app, 80, 10);
        let header = output.lines().next().unwrap();
        assert!(header.contains("filter:"), "got: {header}");
        assert!(
            header.contains("--session") || header.contains('+'),
            "either a component or the dropped count must survive; got: {header}",
        );
    }

    #[test]
    fn a_long_show_title_cannot_starve_the_header_indicator() {
        // `header_label` carries an untruncated session title. An
        // unclamped `Length` for the right label satisfies it first
        // and hands `Fill` zero columns, taking the tabs and the
        // filter indicator off screen entirely.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        set_show_view(&mut app);
        if let View::Show { header_label, .. } = &mut app.view {
            *header_label = format!("\"{}\" (alpha)", "x".repeat(120));
        }
        let output = render_app(&mut app, 80, 10);
        let header = output.lines().next().unwrap();
        assert!(header.contains("[Sessions]"), "got: {header}");
        assert!(header.contains("filter:"), "got: {header}");
    }

    #[test]
    fn a_very_narrow_header_still_reports_the_dropped_count() {
        // At 50 columns the marker and the count cannot both fit, and
        // pushing both had ratatui clip exactly the count — the row
        // read `filter:` and said nothing about how many.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        let output = render_app(&mut app, 50, 10);
        let header = output.lines().next().unwrap();
        assert!(
            header.contains('+'),
            "the count is the information and must survive: {header}",
        );
    }

    #[test]
    fn header_renders_the_session_component_on_both_tabs() {
        // The text is identical on both tabs — only the styling
        // differs, which is what keeps the indicator from diverging
        // from the CLI's flag-shaped hint.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = Query {
            inputs_session_id: Some("abc".to_string()),
            ..Default::default()
        };
        assert!(render_app(&mut app, 120, 10).contains("--session abc"));

        app.tab = Tab::Inputs;
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert!(render_app(&mut app, 120, 10).contains("--session abc"));
    }

    // --- Empty-result states ---

    #[test]
    fn zero_row_list_with_a_filter_names_it_and_offers_f() {
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = Query {
            sessions: SessionFilter {
                project_name: Some("nope".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let output = render_app(&mut app, 100, 10);
        assert!(output.contains("No sessions match"), "got: {output}");
        assert!(output.contains("--project nope"), "got: {output}");
        assert!(
            output.contains("Press f to change filters."),
            "got: {output}"
        );
    }

    #[test]
    fn zero_row_list_without_a_filter_names_the_projects_dir() {
        // The genuinely-empty case, which no filter change fixes.
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 100, 10);
        assert!(output.contains("No sessions found in"), "got: {output}");
        assert!(!output.contains("Press f"), "got: {output}");
    }

    #[test]
    fn zero_row_inputs_with_a_filter_names_it_and_offers_f() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            vec![],
            fixture_coverage_stats(),
        )));
        app.ctx.query = Query {
            thresholds: ThresholdsFilter {
                min_tokens: Some(999_999_999),
                min_cost: None,
            },
            ..Default::default()
        };
        let output = render_app(&mut app, 100, 10);
        assert!(output.contains("No files match"), "got: {output}");
        assert!(output.contains("--min-tokens 999999999"), "got: {output}");
        assert!(
            output.contains("Press f to change filters."),
            "got: {output}"
        );
    }

    #[test]
    fn zero_row_inputs_without_a_filter_names_the_projects_dir() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            vec![],
            fixture_coverage_stats(),
        )));
        let output = render_app(&mut app, 100, 10);
        assert!(output.contains("No files found in"), "got: {output}");
        assert!(!output.contains("Press f"), "got: {output}");
    }

    #[test]
    fn loading_sessions_slot_is_not_mistaken_for_an_empty_result() {
        // Loaded-with-zero-rows and Loading share one render path;
        // during the reload a commit opens they must stay
        // distinguishable.
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        let output = render_app(&mut app, 100, 10);
        assert!(output.contains("Loading sessions"), "got: {output}");
        assert!(!output.contains("No sessions"), "got: {output}");
    }

    #[test]
    fn commit_clears_a_standing_status_so_the_empty_state_is_not_masked() {
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        app.status = Some(StatusMessage {
            text: "stale error".to_string(),
            from_refresh_failure: true,
            kind: StatusKind::Error,
        });
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert!(app.status.is_none());
        assert_eq!(app.consecutive_refresh_failures, 0);
    }

    // --- Keybinding discoverability ---

    #[test]
    fn loaded_view_footers_advertise_f_filter() {
        let mut list = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(render_app(&mut list, 100, 10).contains("f filter"));

        let mut show = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut show);
        assert!(render_app(&mut show, 100, 10).contains("f filter"));

        let mut inputs = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        inputs.inputs = Some(InputsData::Loaded(InputsState::new(
            priced_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert!(render_app(&mut inputs, 100, 10).contains("f filter"));
    }

    #[test]
    fn feedback_view_footers_stay_terse() {
        // Deliberately unchanged: the empty-state copy names `f`
        // where it matters, and these rows are the narrow ones.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        let output = render_app(&mut app, 100, 10);
        assert!(!output.contains("f filter"), "got: {output}");
    }

    // --- Filter overlay rendering ---

    #[test]
    fn filter_overlay_renders_every_label_and_the_apply_footer() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = populated_query();
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        let output = render_app(&mut app, 100, 20);
        for label in [
            "session",
            "project",
            "since",
            "until",
            "min-tokens",
            "min-cost",
        ] {
            assert!(output.contains(label), "label `{label}` missing:\n{output}");
        }
        assert!(output.contains("Enter apply"), "got:\n{output}");
        assert!(output.contains("(clear-only)"), "got:\n{output}");
    }

    #[test]
    fn filter_overlay_replaces_apply_with_an_error_prompt_and_shows_the_message() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for _ in 0..3 {
            handle_key_event(&mut app, key_event(KeyCode::Tab)); // project -> min-tokens
        }
        handle_key_event(&mut app, key_event(KeyCode::Char('x')));
        let output = render_app(&mut app, 100, 20);
        assert!(output.contains("expected a whole number"), "got:\n{output}");
        assert!(output.contains("fix errors to apply"), "got:\n{output}");
        assert!(!output.contains("Enter apply"), "got:\n{output}");
    }

    #[test]
    fn filter_overlay_omits_the_clear_only_marker_when_the_session_field_is_empty() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        let output = render_app(&mut app, 100, 20);
        assert!(!output.contains("(clear-only)"), "got:\n{output}");
    }

    #[test]
    fn empty_state_omits_filters_that_cannot_constrain_the_view() {
        // `load_sessions` never reads `inputs_session_id`, so naming
        // `--session` here would blame a filter that provably excluded
        // nothing — and offer a remedy that changes nothing.
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        app.ctx.query = Query {
            inputs_session_id: Some("abc".to_string()),
            ..Default::default()
        };
        let output = render_app(&mut app, 100, 10);
        // The header still names it — that over-claim is deliberate,
        // and it renders dimmed with the tab context to explain it.
        // The empty state is what must not assert causation.
        let body: String = output.lines().skip(1).collect::<Vec<_>>().join("\n");
        assert!(!body.contains("--session abc"), "got: {body}");
        assert!(
            body.contains("No sessions found in"),
            "with no view-constraining filter left, this is the genuinely-empty case; got: {body}",
        );

        // The same component is load-bearing on Inputs.
        app.tab = Tab::Inputs;
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            vec![],
            fixture_coverage_stats(),
        )));
        let output = render_app(&mut app, 100, 10);
        assert!(
            output.contains("No files match --session abc"),
            "got: {output}"
        );
    }

    #[test]
    fn zero_row_show_explains_the_threshold_that_emptied_it() {
        // `load_show` applies `ctx.query.thresholds`, and `f` binds
        // from inside Show — so this state is reachable in one commit.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        if let View::Show { prepared, .. } = &mut app.view {
            prepared.clear();
        }
        app.ctx.query = Query {
            thresholds: ThresholdsFilter {
                min_tokens: Some(999_999_999),
                min_cost: None,
            },
            ..Default::default()
        };
        let output = render_app(&mut app, 100, 10);
        assert!(output.contains("No exchanges match"), "got: {output}");
        assert!(output.contains("--min-tokens 999999999"), "got: {output}");
        assert!(
            output.contains("Press f to change filters."),
            "got: {output}"
        );
        assert!(
            output.contains("Esc back"),
            "the way out must stay visible; got: {output}"
        );
    }

    #[test]
    fn zero_row_show_without_a_filter_names_the_session_not_the_directory() {
        // A session can hold no substantive exchanges with no filter
        // active. The `projects_dir` is not what emptied it, so the
        // explanation the other views share does not fit here.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        if let View::Show { prepared, .. } = &mut app.view {
            prepared.clear();
        }
        let output = render_app(&mut app, 100, 10);
        assert!(
            output.contains("No exchanges in this session."),
            "got: {output}"
        );
        assert!(
            !output.contains("Press f to change filters."),
            "no filter is active, so `f` is not the remedy; got: {output}",
        );
    }

    #[test]
    fn zero_row_show_never_blames_a_filter_it_does_not_load_under() {
        // `load_show` is handed one session id and applies thresholds
        // only. Naming `--project` here would assert a causation that
        // did not happen and offer a remedy that changes nothing.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        if let View::Show { prepared, .. } = &mut app.view {
            prepared.clear();
        }
        app.ctx.query = populated_query();
        let output = render_app(&mut app, 100, 12);
        assert!(output.contains("No exchanges match"), "got: {output}");
        assert!(output.contains("--min-tokens 50000"), "got: {output}");
        assert!(
            !output.contains("--project"),
            "scope filters never reach `load_show`; got: {output}",
        );
        assert!(
            !output.contains("--session"),
            "`--session` is read by the inputs loader alone; got: {output}",
        );
    }

    #[test]
    fn ctrl_chords_do_not_insert_literal_characters() {
        // Ctrl+W is a habitual line-kill in a text field; inserting a
        // literal `w` is worse than ignoring the chord.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        // The editor opens focused on `project`, the first field that
        // accepts text.
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for c in "alpha".chars() {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL),
        );
        assert_eq!(filter_editor(&app).fields[1].input, "alpha");
    }

    #[test]
    fn ctrl_u_clears_the_focused_field() {
        // The editor is append-only, so without this the only way back
        // from a long mistyped value is one backspace per character.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for c in "alpha".chars() {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
        );
        assert_eq!(filter_editor(&app).fields[1].input, "");
        // Clearing re-parses: an emptied field is an inactive filter,
        // not a parse error that blocks the commit.
        assert!(filter_editor(&app).is_committable());
    }

    #[test]
    fn short_terminal_keeps_the_focused_filter_field_visible() {
        // The popup is capped to the terminal height. Without a scroll
        // offset the focused field and its parse error are clipped,
        // and the footer's "fix errors to apply" points at nothing the
        // user can see.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for _ in 0..4 {
            handle_key_event(&mut app, key_event(KeyCode::Tab)); // project -> min-cost
        }
        for c in "nope".chars() {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        let output = render_app(&mut app, 100, 8);
        assert!(output.contains("min-cost"), "got: {output}");
        assert!(
            output.contains("nope"),
            "the typed value must stay visible; got: {output}",
        );
        assert!(output.contains("fix errors to apply"), "got: {output}");
    }

    #[test]
    fn min_cost_field_rejects_non_finite_values() {
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        let before = app.ctx_generation;
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for _ in 0..4 {
            handle_key_event(&mut app, key_event(KeyCode::Tab)); // project -> min-cost
        }
        for c in "nan".chars() {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        assert!(!filter_editor(&app).is_committable());
        handle_key_event(&mut app, key_event(KeyCode::Enter));
        // Committing NaN would empty every view with no error, since
        // `cost >= NaN` is false for every row.
        assert_eq!(app.ctx.query, Query::default());
        assert_eq!(app.ctx_generation, before);
    }

    #[test]
    fn empty_state_footers_keep_the_pricing_key() {
        // `p` is still bound from these states; only navigation and
        // open keys are meaningless with zero rows.
        let mut app = new_app(vec![], fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 100, 10);
        let last = output.lines().last().unwrap();
        assert!(last.contains("p pricing"), "got: {last}");
        assert!(last.contains("f filter"), "got: {last}");
        assert!(!last.contains("navigate"), "got: {last}");
    }

    // --- Status severity and the missing-model hint ---

    /// `fixture_sessions` with its precondition asserted rather than
    /// assumed: every session priced, so the derived hint stays silent
    /// unless a test deliberately unprices one.
    fn fixture_sessions_asserted_priced() -> Vec<Session> {
        let sessions = fixture_sessions();
        assert!(
            sessions.iter().all(|s| s.cost_breakdown.is_some()),
            "fixture must start fully priced",
        );
        sessions
    }

    fn unpriced_sessions() -> Vec<Session> {
        let mut sessions = fixture_sessions();
        sessions[1].cost_breakdown = None;
        sessions
    }

    fn footer_of(app: &mut App) -> String {
        render_app(app, 120, 10)
            .lines()
            .last()
            .unwrap()
            .trim_end()
            .to_string()
    }

    #[test]
    fn footer_reports_the_unpriced_condition_at_eighty_columns() {
        // The full remedy needs 114 columns beside the list hints. An
        // all-or-nothing drop made a hint this branch adds inert on
        // the most common terminal width.
        let mut app = new_app(unpriced_sessions(), fixture_pricing_data(), Tab::Sessions);
        let output = render_app(&mut app, 80, 10);
        let footer = output.lines().last().unwrap();
        assert!(footer.contains("unpriced rows"), "got: {footer}");
        assert!(
            !footer.contains("p then r"),
            "the remedy does not fit at 80; the condition still must show: {footer}",
        );
        assert!(
            footer.contains("q quit"),
            "the key hints keep their place: {footer}",
        );

        let wide = render_app(&mut app, 130, 10);
        let wide_footer = wide.lines().last().unwrap();
        assert!(wide_footer.contains("p then r"), "got: {wide_footer}");
    }

    #[test]
    fn elide_front_keeps_the_tail_and_marks_the_cut() {
        assert_eq!(elide_front("alpha", 10), "alpha");
        assert_eq!(elide_front("alpha", 5), "alpha");
        assert_eq!(elide_front("alphabet", 5), "…abet");
        assert_eq!(elide_front("alpha", 1), "…");
        assert_eq!(elide_front("alpha", 0), "");
        // Scalars, not bytes: each `é` is two bytes but one column.
        assert_eq!(elide_front(&"é".repeat(6), 4), "…ééé");
    }

    #[test]
    fn a_long_filter_value_cannot_push_its_parse_error_off_the_row() {
        // The overlay has no cursor and no horizontal scroll, so an
        // unbounded value would run past the border and take the error
        // message with it — while the footer still says "fix errors".
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Sessions);
        handle_key_event(&mut app, key_event(KeyCode::Char('f')));
        for _ in 0..4 {
            handle_key_event(&mut app, key_event(KeyCode::Tab)); // project -> min-cost
        }
        for c in "9".repeat(80).chars().chain("x".chars()) {
            handle_key_event(&mut app, key_event(KeyCode::Char(c)));
        }
        let output = render_app(&mut app, 100, 14);
        assert!(
            output.contains("expected a number"),
            "the parse error keeps its place: {output}",
        );
        assert!(
            output.contains('…'),
            "the value gives ground and marks the cut: {output}",
        );
        assert!(output.contains("fix errors to apply"), "got: {output}");
    }

    #[test]
    fn footer_precedence_orders_error_info_derived_then_hint() {
        let mut app = new_app(unpriced_sessions(), fixture_pricing_data(), Tab::Sessions);

        // 4. Nothing set — the static key hints.
        let mut priced = new_app(
            fixture_sessions_asserted_priced(),
            fixture_pricing_data(),
            Tab::Sessions,
        );
        let hint = footer_of(&mut priced);
        assert!(hint.contains("q quit"), "got: {hint}");
        assert!(!hint.contains("unpriced"), "got: {hint}");

        // 3. A derived hint shares the row with the key hints rather
        // than evicting them — it never clears, so taking the row
        // would cost the user `q quit` permanently.
        let derived = footer_of(&mut app);
        assert!(derived.contains("some rows unpriced"), "got: {derived}");
        assert!(derived.contains("q quit"), "got: {derived}");

        // 2. An Info status outranks the derived hint.
        app.status = Some(StatusMessage {
            text: "an informational note".to_string(),
            kind: StatusKind::Info,
            from_refresh_failure: false,
        });
        let info = footer_of(&mut app);
        assert!(info.contains("an informational note"), "got: {info}");
        assert!(!info.contains("unpriced"), "got: {info}");

        // 1. An error outranks everything — it is the newer,
        // user-triggered fact, while the derived condition will still
        // be true once it clears.
        app.status = Some(StatusMessage {
            text: "a hard failure".to_string(),
            kind: StatusKind::Error,
            from_refresh_failure: false,
        });
        let error = footer_of(&mut app);
        assert!(error.contains("a hard failure"), "got: {error}");
        assert!(!error.contains("unpriced"), "got: {error}");
    }

    #[test]
    fn list_footer_reports_unpriced_sessions_and_stays_silent_when_all_are_priced() {
        let mut unpriced = new_app(unpriced_sessions(), fixture_pricing_data(), Tab::Sessions);
        assert!(footer_of(&mut unpriced).contains("some rows unpriced"));

        let mut priced = new_app(
            fixture_sessions_asserted_priced(),
            fixture_pricing_data(),
            Tab::Sessions,
        );
        assert!(!footer_of(&mut priced).contains("some rows unpriced"));
    }

    #[test]
    fn inputs_footer_reports_unpriced_rows() {
        // The shipped fixture already carries one `attributed_cost:
        // None` row.
        let mut app = new_app(fixture_sessions(), fixture_pricing_data(), Tab::Inputs);
        app.inputs = Some(InputsData::Loaded(InputsState::new(
            fixture_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert!(footer_of(&mut app).contains("some rows unpriced"));

        app.inputs = Some(InputsData::Loaded(InputsState::new(
            priced_attribution_rows(),
            fixture_coverage_stats(),
        )));
        assert!(!footer_of(&mut app).contains("some rows unpriced"));
    }

    #[test]
    fn feedback_view_reports_no_missing_model_hint() {
        // True by construction — `render_feedback_content` passes
        // `None` because it has no rows to derive from.
        let mut app = new_app(unpriced_sessions(), fixture_pricing_data(), Tab::Sessions);
        app.sessions = SessionsData::Loading;
        assert!(!footer_of(&mut app).contains("unpriced"));
    }

    #[test]
    fn show_view_reports_no_missing_model_hint() {
        // A `PreparedExchange` is a per-turn exchange rather than a
        // priced session, so "unpriced" is undefined over Show's rows
        // even when the underlying sessions are unpriced.
        let mut app = new_app(unpriced_sessions(), fixture_pricing_data(), Tab::Sessions);
        set_show_view(&mut app);
        assert!(!footer_of(&mut app).contains("unpriced"));
    }

    #[test]
    fn narrow_footer_drops_the_derived_half_not_the_key_hints() {
        // The key hints are the only place `q quit` is advertised, so
        // when both cannot fit, the standing condition is what yields.
        let mut app = new_app(unpriced_sessions(), fixture_pricing_data(), Tab::Sessions);

        let wide = render_app(&mut app, 130, 10)
            .lines()
            .last()
            .unwrap()
            .to_string();
        assert!(wide.contains("some rows unpriced"), "got: {wide}");
        assert!(wide.contains("q quit"), "got: {wide}");

        let narrow = render_app(&mut app, 70, 10)
            .lines()
            .last()
            .unwrap()
            .to_string();
        assert!(!narrow.contains("unpriced"), "got: {narrow}");
        assert!(narrow.contains("q quit"), "got: {narrow}");
    }
}
