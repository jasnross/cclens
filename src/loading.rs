//! Data-loading pipeline for the TUI (and, via the same functions, the
//! plain/JSON CLI paths). Composes `discovery` → `parsing` →
//! `aggregation` → `attribution` → `pricing` into view-ready data.
//!
//! Every loader parses transcripts through one private `ProjectWalk`
//! per project, the only parse path in this module. The walk parses
//! every parent, orders sessions by `continued-in` lineage (falling
//! back to discovery's mtime order), then dedups each session's parent
//! turns and its subagent transcripts against one per-project
//! `SeenTurns`, omitting a subagent transcript dedup emptied. Content a
//! continuation replays or links therefore counts once, under the
//! session that produced it.
//!
//! Public API:
//! - `Query` — the mutable, user-facing slice of load configuration
//!   (`sessions: SessionFilter`, `thresholds: ThresholdsFilter`,
//!   `inputs_session_id: Option<String>`, `pinning: PinningFilter`).
//!   The sole source of filter semantics across all four loaders.
//! - `DataContext` — everything a load needs (`projects_dir`,
//!   `catalog`, `inventory`, `query`). Cloned into each dispatched
//!   load; cheap (a `PathBuf`, two `Arc` bumps, a small `Query`).
//! - `RefreshFingerprint` — file-size + mtime fingerprint for change
//!   detection, keyed by watched path.
//! - `PricingData` — owned pricing entries + cache staleness info,
//!   built once via `pricing_data` and shared by startup and refresh.
//! - `load_sessions(&DataContext) -> anyhow::Result<Vec<Session>>`
//! - `load_show(&DataContext, &str) -> anyhow::Result<Vec<PreparedExchange>>`
//! - `load_inputs(&DataContext) -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)>`
//! - `load_agents(&DataContext) -> anyhow::Result<Vec<AgentRow>>`
//! - `Query::describe_active` — every active filter as a flag-shaped
//!   `FilterComponent`, in display order (`--session`, scope,
//!   thresholds). One producer for the CLI hint and the TUI header.
//! - `build_fingerprint(&Path) -> anyhow::Result<RefreshFingerprint>`
//! - `pricing_data(&PricingCatalog) -> PricingData`
//! - `refresh_pricing() -> anyhow::Result<(Arc<PricingCatalog>, PricingData)>`

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use crate::agents::{
    AgentRow, Dispatch, PinningFilter, dispatch_from_turns, group_dispatches, sort_rows,
};
use crate::aggregation::{
    PreparedExchange, SeenTurns, aggregate, dedup_turns, derive_project_short_name,
    group_into_exchanges, prepare_exchanges,
};
use crate::attribution::{
    AttributionRow, CoverageStats, InputsFilter, SessionKind, SessionMeta, compute_coverage,
    compute_rows, extend_inventory_for_session, session_meta_from_turns,
};
use crate::discovery::{
    ProjectSessions, SessionPaths, SubagentMeta, SubagentPaths, discover, read_subagent_meta,
};
use crate::domain::{Session, Turn, TurnOrigin};
use crate::filter::{
    FilterComponent, HonoredBy, SessionFilter, ThresholdsFilter, quote_filter_value,
};
use crate::inventory::{
    AgentFrontmatter, ContextFileKind, InventoryConfig, discover_inventory, read_agent_frontmatter,
};
use crate::parsing::parse_jsonl;
use crate::pricing::{self, ClaudePricing, PricingCatalog};

/// The mutable, user-facing slice of load configuration. Shared by all
/// three loaders — `load_inputs` builds its own `InputsFilter` from
/// `sessions` + `inputs_session_id` rather than taking one pre-built,
/// so a single `Query` is the sole source of filter semantics.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Query {
    pub sessions: SessionFilter,
    pub thresholds: ThresholdsFilter,
    pub inputs_session_id: Option<String>,
    /// Which pinning classifications the agents view admits. Unlike
    /// the other three fields, its default narrows rather than
    /// admitting everything — see `PinningFilter::describe_active` for
    /// why that does not make every view look filtered.
    pub pinning: PinningFilter,
}

impl Query {
    /// Every active filter as a flag-shaped component, in display
    /// order: `--session`, then scope, then thresholds, then
    /// `--pinning`. An empty vector means no filter is active — which
    /// is how both the CLI hint and the TUI empty states distinguish
    /// "filtered to nothing" from "nothing to show", and why the
    /// pinning filter contributes nothing at its (narrowing) default.
    #[must_use]
    pub fn describe_active(&self) -> Vec<FilterComponent> {
        let mut components = Vec::new();
        if let Some(id) = &self.inputs_session_id {
            components.push(FilterComponent {
                text: format!("--session {}", quote_filter_value(id)),
                honored_by: HonoredBy::INPUTS_ONLY,
            });
        }
        components.extend(self.sessions.describe_active());
        components.extend(self.thresholds.describe_active());
        components.extend(self.pinning.describe_active());
        components
    }
}

/// Everything a load needs. Cloned into each dispatched load — a
/// `PathBuf`, an `Arc` bump, and a small `Query` — so the single-
/// threaded event loop can hand out a consistent snapshot per dispatch
/// without locks.
#[derive(Clone)]
pub struct DataContext {
    pub projects_dir: PathBuf,
    pub catalog: Arc<PricingCatalog>,
    /// Filesystem roots the context-file walk reads. Held here rather
    /// than defaulted inside each loader so a caller — a test above
    /// all — can point the walk somewhere without reaching for the
    /// process-global `CCLENS_CLAUDE_HOME`. `Arc` for the same reason
    /// `catalog` is one: every dispatched load clones the context.
    pub inventory: Arc<InventoryConfig>,
    pub query: Query,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RefreshFingerprint {
    pub entries: HashMap<PathBuf, (u64, SystemTime)>,
}

pub struct PricingData {
    pub entries: Vec<(String, ClaudePricing)>,
    pub cache_info: pricing::CacheInfo,
}

/// Build the `PricingData` snapshot from a catalog. Shared by startup
/// and by `refresh_pricing` so both build the snapshot identically.
#[must_use]
pub fn pricing_data(catalog: &PricingCatalog) -> PricingData {
    PricingData {
        entries: catalog
            .sorted_entries(false)
            .into_iter()
            .map(|(k, v)| (k.to_owned(), *v))
            .collect(),
        cache_info: pricing::cache_info(),
    }
}

/// Refresh the on-disk pricing cache and return the reloaded catalog
/// alongside its `PricingData` snapshot. Callers own applying the new
/// catalog to their `DataContext` and bumping any generation counter.
///
/// # Errors
/// Propagates `pricing::refresh_catalog`'s error (network, HTTP
/// status, parse failure, or cache-file write failure).
pub fn refresh_pricing() -> anyhow::Result<(Arc<PricingCatalog>, PricingData)> {
    pricing::refresh_catalog()?;
    let catalog = pricing::load_catalog();
    let data = pricing_data(&catalog);
    Ok((Arc::new(catalog), data))
}

/// Returns sessions sorted by `started_at` (ascending), honoring
/// `ctx.query.sessions` and `ctx.query.thresholds`.
///
/// # Errors
/// Propagates a failure to read `ctx.projects_dir` itself; per-project
/// and per-session read/parse failures are skipped rather than
/// propagated (see `discovery`'s degrade-per-entry contract).
pub fn load_sessions(ctx: &DataContext) -> anyhow::Result<Vec<Session>> {
    let project_entries = discover(&ctx.projects_dir)?;
    let mut sessions = Vec::new();
    for ProjectSessions {
        project_dir,
        sessions: session_paths,
    } in project_entries
    {
        for walked in ProjectWalk::new(&project_dir, session_paths) {
            // An unreadable parent skips the whole session, its readable
            // subagents included: they have already fed the walk's
            // seen-set, so they count nowhere here (the agents view
            // still shows them), and a later session's linked copy of
            // one is not counted either. Only outside interference
            // produces this shape — a dangling symlink or a permissions
            // change on the parent alone, or a disk read error.
            let Some(turns) = walked.parent else {
                continue;
            };
            let subagent_turn_lists: Vec<Vec<Turn>> =
                walked.subagents.into_iter().map(|s| s.turns).collect();
            if let Some(session) = aggregate(
                walked.parent_short_name,
                walked.session_id,
                turns,
                &subagent_turn_lists,
                &ctx.catalog,
            ) && ctx
                .query
                .sessions
                .accepts(&session.project_short_name, session.started_at)
                && ctx.query.thresholds.matches(
                    session.total_billable,
                    session.cost_breakdown.map(|b| b.total()),
                )
            {
                sessions.push(session);
            }
        }
    }
    sessions.sort_by_key(|s| s.started_at);
    Ok(sessions)
}

/// Build one session's prepared per-exchange rows, honoring
/// `ctx.query.thresholds`. Rendering is the caller's job — see
/// `rendering::render_session` and `tui::render_show_table`.
///
/// # Errors
/// Returns an error if no session matches `session_id`, or if more
/// than one project claims the same session id.
pub fn load_show(ctx: &DataContext, session_id: &str) -> anyhow::Result<Vec<PreparedExchange>> {
    let project_entries = discover(&ctx.projects_dir)?;

    let mut matched_project: Option<(PathBuf, Vec<SessionPaths>)> = None;
    for ProjectSessions {
        project_dir,
        sessions: session_paths,
    } in project_entries
    {
        if session_paths
            .iter()
            .any(|sp| stem_matches(&sp.jsonl, session_id))
        {
            if matched_project.is_some() {
                anyhow::bail!("multiple sessions match id {session_id}");
            }
            matched_project = Some((project_dir, session_paths));
        }
    }
    let Some((project_dir, session_paths)) = matched_project else {
        anyhow::bail!("no session matches id {session_id}");
    };

    // The walk is lazy past its ordering phase, so `find` stops before
    // reading the subagents of any session ordered after the target.
    let Some(WalkedSession {
        parent: Some(parent_turns),
        subagents,
        ..
    }) = ProjectWalk::new(&project_dir, session_paths).find(|s| s.session_id == session_id)
    else {
        anyhow::bail!("no session matches id {session_id}");
    };
    let mut all_exchanges = group_into_exchanges(&parent_turns);
    for sub in &subagents {
        all_exchanges.extend(group_into_exchanges(&sub.turns));
    }
    all_exchanges.sort_by_key(|ex| {
        ex.user
            .and_then(|u| u.timestamp)
            .or_else(|| ex.assistants.first().and_then(|a| a.timestamp))
            .unwrap_or(chrono::DateTime::<chrono::Utc>::UNIX_EPOCH)
    });
    Ok(prepare_exchanges(
        &all_exchanges,
        &ctx.catalog,
        ctx.query.thresholds,
    ))
}

/// Attribution rows + coverage for the `inputs` view, honoring
/// `ctx.query.sessions` / `ctx.query.inputs_session_id` (via an
/// internally-built `InputsFilter`) and `ctx.query.thresholds`.
///
/// # Errors
/// Propagates a failure to read `ctx.projects_dir` itself.
pub fn load_inputs(ctx: &DataContext) -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)> {
    let inputs_filter = InputsFilter {
        session_id: ctx.query.inputs_session_id.clone(),
        scope: ctx.query.sessions.clone(),
    };

    let inventory_config = ctx.inventory.as_ref();
    let mut inventory = discover_inventory(inventory_config);
    let mut seen_inventory_paths: HashSet<PathBuf> = HashSet::new();
    for file in &inventory {
        seen_inventory_paths.insert(file.path.clone());
    }

    let project_entries = discover(&ctx.projects_dir)?;
    let mut session_metas: Vec<SessionMeta> = Vec::new();
    for ProjectSessions {
        project_dir,
        sessions: session_paths,
    } in project_entries
    {
        // A session the filter rejects has already fed the walk's
        // seen-set, so a later session never re-counts its content.
        for walked in ProjectWalk::new(&project_dir, session_paths) {
            // As in `load_sessions`, an unreadable parent skips the
            // whole session, and its subagents, already fed to the
            // seen-set, count nowhere here.
            let Some(turns) = walked.parent else {
                continue;
            };
            let Some(parent_meta) = session_meta_from_turns(
                SessionKind::Parent,
                walked.session_id,
                walked.parent_cwd.as_deref(),
                &walked.parent_short_name,
                &turns,
            ) else {
                continue;
            };
            if !inputs_filter.accepts(&parent_meta) {
                continue;
            }
            if let Some(cwd) = &parent_meta.cwd {
                extend_inventory_for_session(
                    &mut inventory,
                    &mut seen_inventory_paths,
                    cwd,
                    inventory_config,
                );
            }
            session_metas.push(parent_meta);
            for sub in walked.subagents {
                let Some(sub_meta) = session_meta_from_turns(
                    SessionKind::Subagent {
                        agent_type: sub.meta.agent_type,
                    },
                    sub.id,
                    sub.cwd.as_deref(),
                    &sub.short_name,
                    &sub.turns,
                ) else {
                    continue;
                };
                if let Some(cwd) = &sub_meta.cwd {
                    extend_inventory_for_session(
                        &mut inventory,
                        &mut seen_inventory_paths,
                        cwd,
                        inventory_config,
                    );
                }
                session_metas.push(sub_meta);
            }
        }
    }

    let rows = compute_rows(inventory, &session_metas, &ctx.catalog);
    let coverage = compute_coverage(&session_metas, &rows);
    let visible: Vec<_> = rows
        .into_iter()
        .filter(|r| {
            ctx.query
                .thresholds
                .matches(r.estimated_tokens_billed, r.attributed_cost)
        })
        .collect();
    Ok((visible, coverage))
}

/// Agent rows for the `agents` view, honoring `ctx.query.sessions`
/// (per dispatch), `ctx.query.pinning` and `ctx.query.thresholds`
/// (per accumulated row).
///
/// # Errors
/// Propagates a failure to read `ctx.projects_dir` itself; per-project
/// and per-subagent read/parse failures are skipped rather than
/// propagated (see `discovery`'s degrade-per-entry contract).
pub fn load_agents(ctx: &DataContext) -> anyhow::Result<Vec<AgentRow>> {
    let inventory_config = ctx.inventory.as_ref();
    let mut inventory = discover_inventory(inventory_config);
    let mut seen_inventory_paths: HashSet<PathBuf> = HashSet::new();
    for file in &inventory {
        seen_inventory_paths.insert(file.path.clone());
    }

    let project_entries = discover(&ctx.projects_dir)?;
    let mut dispatches: Vec<Dispatch> = Vec::new();
    for ProjectSessions {
        project_dir,
        sessions: session_paths,
    } in project_entries
    {
        for walked in ProjectWalk::new(&project_dir, session_paths) {
            if walked.subagents.is_empty() {
                continue;
            }
            // The parent's own turns never enter a dispatch; through the
            // walk it supplies only the cwd and short name a subagent
            // that recorded neither falls back on to match a
            // project-local agent file.
            //
            // An unreadable parent degrades to no fallbacks rather
            // than skipping the session: most subagent transcripts
            // record their own cwd, and dropping them here would
            // delete readable, priced spend over a file this loader
            // only consults for a default.
            if let Some(cwd) = &walked.parent_cwd {
                extend_inventory_for_session(
                    &mut inventory,
                    &mut seen_inventory_paths,
                    cwd,
                    inventory_config,
                );
            }

            for sub in &walked.subagents {
                let Some(dispatch) = dispatch_from_turns(
                    &sub.meta,
                    sub.cwd.as_deref(),
                    &sub.short_name,
                    &sub.turns,
                    &ctx.catalog,
                ) else {
                    continue;
                };
                if let Some(cwd) = &dispatch.cwd {
                    extend_inventory_for_session(
                        &mut inventory,
                        &mut seen_inventory_paths,
                        cwd,
                        inventory_config,
                    );
                }
                if !ctx
                    .query
                    .sessions
                    .accepts(&dispatch.project_short_name, dispatch.started_at)
                {
                    continue;
                }
                dispatches.push(dispatch);
            }
        }
    }

    // Built only once the inventory is final — project-local agent
    // files enter it exclusively through the
    // `extend_inventory_for_session` calls above, so reading
    // frontmatter any earlier would miss them. Only agent kinds are
    // read, so no CLAUDE.md, rule, skill, or command is parsed as YAML.
    let frontmatter: HashMap<PathBuf, AgentFrontmatter> = inventory
        .iter()
        .filter(|f| {
            matches!(
                f.kind,
                ContextFileKind::UserAgent
                    | ContextFileKind::PluginAgent { .. }
                    | ContextFileKind::ProjectLocalAgent
            )
        })
        .map(|f| (f.path.clone(), read_agent_frontmatter(&f.path)))
        .collect();

    let mut rows = group_dispatches(dispatches, &inventory, &frontmatter);
    rows.retain(|row| {
        ctx.query.pinning.accepts(&row.pinning)
            && ctx
                .query
                .thresholds
                .matches(row.usage.billable(), row.cost.map(|c| c.total()))
    });
    sort_rows(&mut rows);
    Ok(rows)
}

/// Build a file-size + mtime fingerprint of every session transcript
/// and context file under `projects_dir` plus `~/.claude`. Used to
/// gate timer-driven refreshes: an unchanged fingerprint skips the
/// (expensive) reload.
///
/// # Errors
/// Propagates a failure to read `projects_dir` itself; individual
/// unreadable files are skipped.
pub fn build_fingerprint(projects_dir: &Path) -> anyhow::Result<RefreshFingerprint> {
    let project_entries = discover(projects_dir)?;
    let mut entries = HashMap::new();
    let mtime = |meta: &std::fs::Metadata| meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    for project in &project_entries {
        for session in &project.sessions {
            if let Ok(meta) = std::fs::metadata(&session.jsonl) {
                entries.insert(session.jsonl.clone(), (meta.len(), mtime(&meta)));
            }
            for sub in &session.subagents {
                if let Ok(meta) = std::fs::metadata(&sub.jsonl) {
                    entries.insert(sub.jsonl.clone(), (meta.len(), mtime(&meta)));
                }
            }
        }
    }
    if let Some(home) = dirs::home_dir() {
        let claude_dir = home.join(".claude");
        for name in ["CLAUDE.md", "rules", "skills", "agents"] {
            let path = claude_dir.join(name);
            if path.is_file() {
                if let Ok(meta) = std::fs::metadata(&path) {
                    entries.insert(path, (meta.len(), mtime(&meta)));
                }
            } else if path.is_dir()
                && let Ok(dir) = std::fs::read_dir(&path)
            {
                for entry in dir.flatten() {
                    if let Ok(meta) = entry.metadata() {
                        if meta.is_file() {
                            entries.insert(entry.path(), (meta.len(), mtime(&meta)));
                        } else if meta.is_dir()
                            && let Ok(sub) = std::fs::read_dir(entry.path())
                        {
                            for child in sub.flatten() {
                                if let Ok(cm) = child.metadata()
                                    && cm.is_file()
                                {
                                    entries.insert(child.path(), (cm.len(), mtime(&cm)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(RefreshFingerprint { entries })
}

// ---- the project walk ----

/// One session as the project walk yields it, already deduplicated
/// against every file the walk fed before it.
struct WalkedSession {
    session_id: String,
    /// `None` when the parent transcript is unreadable.
    parent: Option<Vec<Turn>>,
    /// The first `cwd` the parent transcript records and the project
    /// short name derived from it, both read before dedup so lines a
    /// continuation replayed still supply them. They are the session's
    /// own cwd and short name in every view — a continuation whose
    /// only cwd-bearing lines are replays still matches `--project` —
    /// and the fallbacks for a subagent transcript that records no cwd
    /// of its own.
    parent_cwd: Option<PathBuf>,
    parent_short_name: String,
    subagents: Vec<WalkedSubagent>,
}

/// One subagent transcript as the project walk yields it: its turns
/// tagged `TurnOrigin::Subagent` and deduplicated. The walk omits a
/// transcript dedup emptied, so every yielded one contributed content
/// no earlier file did — or had none to contribute.
struct WalkedSubagent {
    /// The transcript's file stem, e.g. `agent-1`.
    id: String,
    meta: SubagentMeta,
    /// The first `cwd` the transcript records and the project short
    /// name derived from it, read before dedup so lines a fork
    /// replayed still supply them — the rule `WalkedSession` applies
    /// to the parent. A transcript recording no cwd takes the
    /// parent's, so it can still match a project-local agent file.
    cwd: Option<PathBuf>,
    short_name: String,
    turns: Vec<Turn>,
}

/// A parent transcript parsed during the walk's ordering phase, with
/// its subagent transcripts still unread.
struct PendingSession {
    session_id: String,
    parent: Option<Vec<Turn>>,
    parent_cwd: Option<PathBuf>,
    parent_short_name: String,
    subagents: Vec<SubagentPaths>,
}

/// The single place the loaders parse transcripts: one project's
/// sessions, ordered by continuation lineage and deduplicated against
/// one shared `SeenTurns`.
///
/// It runs in two phases. `new` parses every parent transcript and
/// orders the sessions with `lineage_order`, which needs every parent
/// first because a `continued-in` record sits near the end of the
/// original's file. Iteration then dedups lazily: each `next` feeds
/// the session's parent turns, then each of its subagent transcripts,
/// into the seen-set before yielding. Subagents feed parent agent
/// first (by the sidecar's `parentAgentId`, falling back to discovery
/// order), because a nested fork replays the opening messages of the
/// agent that spawned it. So content a continuation replays or links
/// counts under the original that produced it, content a fork replays
/// counts under its parent agent, and a caller that stops early never
/// reads the subagents of later sessions.
///
/// An unreadable parent still feeds its subagents to the seen-set, so
/// a later session's linked copy of one is recognized as repeated.
struct ProjectWalk {
    project_dir: PathBuf,
    pending: std::vec::IntoIter<PendingSession>,
    seen: SeenTurns,
}

impl ProjectWalk {
    fn new(project_dir: &Path, sessions: Vec<SessionPaths>) -> Self {
        let mut session_ids = Vec::with_capacity(sessions.len());
        let mut continued_in = Vec::with_capacity(sessions.len());
        let mut pending: Vec<Option<PendingSession>> = Vec::with_capacity(sessions.len());
        for SessionPaths { jsonl, subagents } in sessions {
            let session_id = file_stem_string(&jsonl);
            let (parent, lineage) = match parse_jsonl(&jsonl) {
                Ok(parsed) => (Some(parsed.turns), parsed.continued_in),
                Err(_) => (None, Vec::new()),
            };
            let full_parent = parent.as_deref().unwrap_or_default();
            let parent_cwd = full_parent.iter().find_map(|t| t.cwd.clone());
            let parent_short_name = derive_project_short_name(project_dir, full_parent);
            session_ids.push(session_id.clone());
            continued_in.push(lineage);
            pending.push(Some(PendingSession {
                session_id,
                parent,
                parent_cwd,
                parent_short_name,
                subagents,
            }));
        }
        let ordered: Vec<PendingSession> = lineage_order(&session_ids, &continued_in)
            .into_iter()
            .filter_map(|i| pending.get_mut(i).and_then(Option::take))
            .collect();
        Self {
            project_dir: project_dir.to_path_buf(),
            pending: ordered.into_iter(),
            seen: SeenTurns::default(),
        }
    }
}

impl Iterator for ProjectWalk {
    type Item = WalkedSession;

    fn next(&mut self) -> Option<WalkedSession> {
        let PendingSession {
            session_id,
            parent,
            parent_cwd,
            parent_short_name,
            subagents: subagent_paths,
        } = self.pending.next()?;
        let parent = parent.map(|turns| dedup_turns(turns, &mut self.seen));
        let mut subagents = Vec::new();
        for (jsonl, meta) in parent_agent_first(subagent_paths) {
            let Ok(parsed) = parse_jsonl(&jsonl) else {
                continue;
            };
            let mut turns = parsed.turns;
            for turn in &mut turns {
                turn.origin = TurnOrigin::Subagent {
                    agent_type: meta.agent_type.clone(),
                    description: meta.description.clone(),
                };
            }
            let (cwd, short_name) = match turns.iter().find_map(|t| t.cwd.clone()) {
                Some(cwd) => (
                    Some(cwd),
                    derive_project_short_name(&self.project_dir, &turns),
                ),
                None => (parent_cwd.clone(), parent_short_name.clone()),
            };
            let before = turns.len();
            let turns = dedup_turns(turns, &mut self.seen);
            if before > 0 && turns.is_empty() {
                continue;
            }
            subagents.push(WalkedSubagent {
                id: file_stem_string(&jsonl),
                meta,
                cwd,
                short_name,
                turns,
            });
        }
        Some(WalkedSession {
            session_id,
            parent,
            parent_cwd,
            parent_short_name,
            subagents,
        })
    }
}

/// Read each subagent's sidecar, skipping a subagent whose sidecar is
/// absent or unreadable, and order the rest so an agent precedes the
/// nested dispatches it spawned (its `parentAgentId` children), with
/// discovery order between unrelated agents.
fn parent_agent_first(subagents: Vec<SubagentPaths>) -> Vec<(PathBuf, SubagentMeta)> {
    let read: Vec<(PathBuf, SubagentMeta)> = subagents
        .into_iter()
        .filter_map(|SubagentPaths { jsonl, meta }| {
            let meta = meta.as_deref().and_then(read_subagent_meta)?;
            Some((jsonl, meta))
        })
        .collect();
    let agent_ids: Vec<String> = read
        .iter()
        .map(|(jsonl, _)| {
            let stem = file_stem_string(jsonl);
            stem.strip_prefix("agent-")
                .map_or(stem.clone(), str::to_string)
        })
        .collect();
    let index_of: HashMap<&str, usize> = agent_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let mut spawned: Vec<Vec<String>> = vec![Vec::new(); agent_ids.len()];
    for (child, (_, meta)) in read.iter().enumerate() {
        if let Some(parent) = meta
            .parent_agent_id
            .as_deref()
            .and_then(|id| index_of.get(id))
            && let Some(children) = spawned.get_mut(*parent)
            && let Some(child_id) = agent_ids.get(child)
        {
            children.push(child_id.clone());
        }
    }
    let order = lineage_order(&agent_ids, &spawned);
    let mut slots: Vec<Option<(PathBuf, SubagentMeta)>> = read.into_iter().map(Some).collect();
    order
        .into_iter()
        .filter_map(|i| slots.get_mut(i).and_then(Option::take))
        .collect()
}

/// Order items so each precedes every item that continues it,
/// returning indices into `ids`. `successors[i]` names the ids that
/// continue item `i`: the sessions a session's `continued-in` records
/// name, or the nested dispatches a subagent spawned.
///
/// A stable topological sort over discovery's mtime order: each step
/// emits the lowest-index item whose predecessors have all been
/// emitted. Ties and items lineage does not relate resolve to
/// discovery's mtime order. A successor id naming no item, or the item
/// itself, is ignored, and a repeated one counts once.
///
/// A cycle (only malformed data produces one) leaves no item free; the
/// fallback then emits the lowest-index remaining item, which need not
/// lie on the cycle — an item that merely continues a cycle member can
/// be emitted ahead of it.
///
/// For sessions, the `continued-in` record sits near the end of the
/// original's file, which is why `ProjectWalk` must parse every parent
/// before it can yield the first.
fn lineage_order(ids: &[String], successors: &[Vec<String>]) -> Vec<usize> {
    let n = ids.len();
    let index_of: HashMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    // `continuations[i]` lists the items that continue `i`;
    // `unplaced_predecessors[j]` counts the items `j` still waits on.
    let mut continuations: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut unplaced_predecessors: Vec<usize> = vec![0; n];
    for (i, targets) in successors.iter().enumerate().take(n) {
        for target in targets {
            let Some(&j) = index_of.get(target.as_str()) else {
                continue;
            };
            if j == i || continuations[i].contains(&j) {
                continue;
            }
            continuations[i].push(j);
            unplaced_predecessors[j] += 1;
        }
    }
    // `free` holds the unplaced items with no unplaced predecessors;
    // ordered sets make "lowest index" a first-element lookup.
    let mut unplaced: BTreeSet<usize> = (0..n).collect();
    let mut free: BTreeSet<usize> = (0..n).filter(|&i| unplaced_predecessors[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = free
        .pop_first()
        // A cycle: no remaining item is free of unplaced
        // predecessors, so fall back to the earliest by mtime.
        .or_else(|| unplaced.first().copied())
    {
        unplaced.remove(&i);
        order.push(i);
        for &j in &continuations[i] {
            unplaced_predecessors[j] = unplaced_predecessors[j].saturating_sub(1);
            if unplaced_predecessors[j] == 0 && unplaced.contains(&j) {
                free.insert(j);
            }
        }
    }
    order
}

fn file_stem_string(path: &Path) -> String {
    path.file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn stem_matches(path: &Path, session_id: &str) -> bool {
    path.file_stem()
        .is_some_and(|s| s.to_string_lossy() == session_id)
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    /// Write one JSONL line for a user turn with a `cwd`, so
    /// `derive_project_short_name` can resolve a project name.
    fn user_line(ts: &str, cwd: &str, content: &str) -> String {
        format!(
            r#"{{"type":"user","timestamp":"{ts}","cwd":"{cwd}","message":{{"role":"user","content":"{content}"}}}}"#
        )
    }

    /// Write one JSONL line for an assistant turn carrying usage —
    /// the minimal shape `parse_jsonl` accepts and `aggregate` bills.
    ///
    /// Carries a small fixed `ephemeral_5m_input_tokens` (5) alongside
    /// `input_tokens`/`output_tokens` so `SessionMeta::observed_tier`
    /// returns `Some` for every fixture session — without it,
    /// `evidence_for_file_in_session` treats a session as having
    /// loaded nothing, and always-loaded inventory rows (CLAUDE.md,
    /// rules) never get credited regardless of session scoping. The
    /// constant is small enough not to perturb any threshold
    /// assertion in this module (thresholds here differ by orders of
    /// magnitude).
    fn assistant_line(
        ts: &str,
        message_id: &str,
        request_id: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> String {
        format!(
            r#"{{"type":"assistant","timestamp":"{ts}","message":{{"id":"{message_id}","role":"assistant","model":"{model}","content":[{{"type":"text","text":"hi"}}],"usage":{{"input_tokens":{input_tokens},"output_tokens":{output_tokens},"cache_creation":{{"ephemeral_5m_input_tokens":5,"ephemeral_1h_input_tokens":0}}}}}},"requestId":"{request_id}"}}"#
        )
    }

    /// Build `<tmp>/projects/<project>/<session_id>.jsonl` with one
    /// user turn and one assistant turn, returning the tempdir (kept
    /// alive by the caller) and the projects root inside it.
    fn write_session(
        tmp: &tempfile::TempDir,
        project: &str,
        session_id: &str,
        cwd: &str,
        started_at: &str,
        input_tokens: u64,
    ) -> PathBuf {
        let root = tmp.path().join("projects");
        let project_dir = root.join(project);
        std::fs::create_dir_all(&project_dir).expect("create project dir");
        let jsonl_path = project_dir.join(format!("{session_id}.jsonl"));
        let mut file = std::fs::File::create(&jsonl_path).expect("create jsonl");
        writeln!(file, "{}", user_line(started_at, cwd, "hello")).expect("write user line");
        writeln!(
            file,
            "{}",
            assistant_line(
                started_at,
                &format!("msg_{session_id}"),
                &format!("req_{session_id}"),
                "claude-sonnet-4-5",
                input_tokens,
                50,
            )
        )
        .expect("write assistant line");
        root
    }

    /// A context whose context-file walk is rooted in an empty
    /// tempdir rather than the developer's real `~/.claude`. Without
    /// it a user-global agent file sharing a test's agent type would
    /// reclassify its row and fail the test on that machine alone.
    /// An empty `~/.claude` stand-in inside `tmp`.
    fn empty_claude_home(tmp: &tempfile::TempDir) -> PathBuf {
        let home = tmp.path().join("claude-home");
        std::fs::create_dir_all(&home).expect("create claude-home");
        home
    }

    fn ctx_with_claude_home(projects_dir: PathBuf, claude_home: &Path) -> DataContext {
        DataContext {
            projects_dir,
            catalog: Arc::new(PricingCatalog::default()),
            inventory: Arc::new(InventoryConfig {
                claude_home: claude_home.to_path_buf(),
                installed_plugins_path: claude_home.join("plugins/installed_plugins.json"),
            }),
            query: Query::default(),
        }
    }

    #[test]
    fn load_sessions_respects_project_filter() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_session(
            &tmp,
            "beta",
            "s2",
            "/home/user/beta",
            "2026-04-02T10:00:00Z",
            1000,
        );

        let mut data_ctx = ctx_with_claude_home(projects_dir, &empty_claude_home(&tmp));
        data_ctx.query.sessions.project_name = Some("alpha".to_string());

        let sessions = load_sessions(&data_ctx).expect("load_sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].project_short_name, "alpha");
    }

    #[test]
    fn load_sessions_respects_thresholds() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            10,
        );
        write_session(
            &tmp,
            "alpha",
            "s2",
            "/home/user/alpha",
            "2026-04-02T10:00:00Z",
            100_000,
        );

        let mut data_ctx = ctx_with_claude_home(projects_dir, &empty_claude_home(&tmp));
        data_ctx.query.thresholds.min_tokens = Some(1000);

        let sessions = load_sessions(&data_ctx).expect("load_sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "s2");
    }

    #[test]
    fn load_show_applies_thresholds() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            10,
        );

        let claude_home = empty_claude_home(&tmp);
        let unfiltered = ctx_with_claude_home(projects_dir.clone(), &claude_home);
        let unfiltered_exchanges = load_show(&unfiltered, "s1").expect("load_show unfiltered");
        assert!(!unfiltered_exchanges.is_empty());

        let mut filtered = ctx_with_claude_home(projects_dir, &claude_home);
        filtered.query.thresholds.min_tokens = Some(1_000_000);
        let filtered_exchanges = load_show(&filtered, "s1").expect("load_show filtered");
        // `prepare_exchanges` omits a below-threshold exchange
        // entirely rather than emitting it with empty rows.
        assert!(
            filtered_exchanges.is_empty(),
            "high threshold should filter out the below-threshold exchange; got {} exchanges",
            filtered_exchanges.len(),
        );
    }

    #[test]
    fn load_show_keeps_leading_assistant_turns() {
        // A continuation can open mid tool loop: assistant turns before
        // any prompt form their own exchange instead of vanishing.
        let tmp = tempfile::tempdir().expect("tempdir");
        let project_dir = tmp.path().join("projects").join("alpha");
        std::fs::create_dir_all(&project_dir).expect("create project dir");
        let mut file = std::fs::File::create(project_dir.join("s1.jsonl")).expect("create jsonl");
        for line in [
            assistant_line("2026-04-01T09:00:00Z", "msg_0", "req_0", "m", 10, 5),
            user_line("2026-04-01T10:00:00Z", "/home/user/alpha", "next"),
            assistant_line("2026-04-01T10:00:00Z", "msg_1", "req_1", "m", 20, 7),
        ] {
            writeln!(file, "{line}").expect("write line");
        }
        drop(file);

        let ctx = ctx_with_claude_home(tmp.path().join("projects"), &empty_claude_home(&tmp));
        let exchanges = load_show(&ctx, "s1").expect("load_show");
        assert_eq!(exchanges.len(), 2);
        // Every assistant turn reaches the running total, so `show`
        // ends where `list` does.
        let sessions = load_sessions(&ctx).expect("load_sessions");
        let last = exchanges[1].rows.last().expect("has rows");
        assert_eq!(last.cumulative_tokens, sessions[0].total_billable);
        assert_eq!(exchanges[0].rows.len(), 1);
        assert!(matches!(
            exchanges[0].rows[0].role,
            crate::aggregation::PreparedRowRole::Continuation
        ));
        assert_eq!(exchanges[1].rows.len(), 2);
        assert!(matches!(
            exchanges[1].rows[0].role,
            crate::aggregation::PreparedRowRole::User
        ));
        assert!(matches!(
            exchanges[1].rows[1].role,
            crate::aggregation::PreparedRowRole::Assistant
        ));
    }

    #[test]
    fn load_inputs_applies_thresholds() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );

        let mut data_ctx = ctx_with_claude_home(projects_dir, &empty_claude_home(&tmp));
        data_ctx.query.thresholds.min_cost = Some(1_000_000.0);

        let (rows, _coverage) = load_inputs(&data_ctx).expect("load_inputs");
        assert!(
            rows.is_empty(),
            "an unreachable cost threshold should filter out every row"
        );
    }

    #[test]
    fn load_inputs_scopes_by_session_id() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_session(
            &tmp,
            "alpha",
            "s2",
            "/home/user/alpha",
            "2026-04-02T10:00:00Z",
            1000,
        );

        // `compute_rows` always emits one row per real `~/.claude`
        // inventory file regardless of `session_metas` (unmatched
        // files just report zero loads) — so row *count* can't
        // distinguish scoping on a machine with any global inventory.
        // What must differ is the *evidence*: every always-loaded
        // file's scope matches any cwd, so both synthetic sessions
        // credit it in the unscoped case; scoping to a session id
        // that matches nothing empties `session_metas` entirely, and
        // every row's loads/billed-tokens/coverage must collapse to
        // zero. The synthetic `~/.claude` below is what makes that
        // deterministic: the walk reads it rather than the developer's
        // real one, so the assertion does not depend on which files
        // happen to exist on this machine.
        let claude_home = empty_claude_home(&tmp);
        std::fs::write(
            claude_home.join("CLAUDE.md"),
            "# Global\n\nA global rule body.\n",
        )
        .expect("write CLAUDE.md");
        let all_ctx = ctx_with_claude_home(projects_dir.clone(), &claude_home);
        let (all_rows, all_coverage) = load_inputs(&all_ctx).expect("load_inputs all");
        let all_loads: u64 = all_rows.iter().map(|r| r.loads_1h + r.loads_5m).sum();
        assert!(
            all_loads > 0,
            "sanity: two sessions with a global-scope-matching cwd should credit \
             at least one always-loaded inventory row",
        );

        let mut unmatched_ctx = ctx_with_claude_home(projects_dir, &claude_home);
        unmatched_ctx.query.inputs_session_id = Some("does-not-exist".to_string());
        let (unmatched_rows, unmatched_coverage) =
            load_inputs(&unmatched_ctx).expect("load_inputs unmatched");

        assert_eq!(
            unmatched_rows.len(),
            all_rows.len(),
            "the inventory itself is scope-independent — only the evidence changes",
        );
        assert!(
            unmatched_rows
                .iter()
                .all(|r| r.loads_1h == 0 && r.loads_5m == 0 && r.estimated_tokens_billed == 0),
            "scoping to a nonexistent session id must zero every row's evidence",
        );
        // The fixture's synthetic assistant turns carry no
        // `cache_creation` usage, so `observed_*_tokens` (which sums
        // raw `cache_creation.ephemeral_*`, independent of the
        // always-loaded-file crediting above) is zero regardless of
        // scoping — asserted for both to document that this signal
        // isn't exercised by this fixture, not to prove scoping.
        assert_eq!(all_coverage.long_1h.observed_tokens, 0);
        assert_eq!(unmatched_coverage.long_1h.observed_tokens, 0);
        assert_eq!(unmatched_coverage.short_5m.observed_tokens, 0);
    }

    #[test]
    fn build_fingerprint_changes_when_file_grows() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let before = build_fingerprint(&projects_dir).expect("fingerprint before");

        let jsonl_path = projects_dir.join("alpha").join("s1.jsonl");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&jsonl_path)
            .expect("open for append");
        writeln!(
            file,
            "{}",
            assistant_line(
                "2026-04-01T10:01:00Z",
                "msg_2",
                "req_2",
                "claude-sonnet-4-5",
                20,
                10,
            )
        )
        .expect("append line");

        let after = build_fingerprint(&projects_dir).expect("fingerprint after");
        assert_ne!(before, after);
    }

    #[test]
    fn data_context_clone_observes_independent_catalog() {
        let ctx_a = DataContext {
            projects_dir: PathBuf::from("/nonexistent"),
            catalog: Arc::new(PricingCatalog::default()),
            inventory: Arc::new(InventoryConfig::default()),
            query: Query::default(),
        };
        let original_catalog_ptr = Arc::as_ptr(&ctx_a.catalog);
        let mut ctx_b = ctx_a.clone();

        // `.clone()` is cheap because it bumps the `Arc`'s refcount
        // rather than deep-copying the catalog — this is what makes
        // per-dispatch cloning in `spawn_load` affordable.
        assert!(
            Arc::ptr_eq(&ctx_a.catalog, &ctx_b.catalog),
            "a fresh clone must share the same catalog allocation",
        );

        // Reassigning the clone's `catalog` — the catalog-swap
        // dispatch site's exact operation — must not perturb the
        // original context still held elsewhere (e.g. `App.ctx`
        // before a pending load against the old generation finishes).
        ctx_b.catalog = Arc::new(PricingCatalog::default());
        assert_eq!(
            Arc::as_ptr(&ctx_a.catalog),
            original_catalog_ptr,
            "reassigning the clone's catalog must not affect the original",
        );
        assert!(!Arc::ptr_eq(&ctx_a.catalog, &ctx_b.catalog));
    }

    #[test]
    fn load_sessions_skips_malformed_jsonl_without_aborting() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let malformed_dir = projects_dir.join("beta");
        std::fs::create_dir_all(&malformed_dir).expect("create beta dir");
        std::fs::write(malformed_dir.join("s2.jsonl"), "not valid json\n")
            .expect("write malformed jsonl");

        let sessions = load_sessions(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "s1");
    }

    // ---- load_agents ----

    /// Write a subagent transcript plus its sidecar under
    /// `<projects_dir>/<project>/<session_id>/subagents/`.
    ///
    /// `lines` are written verbatim so a test can repeat a
    /// `(message_id, request_id)` pair, which is the shape the dedup
    /// assertions turn on.
    fn write_subagent(
        projects_dir: &Path,
        project: &str,
        session_id: &str,
        agent_file: &str,
        agent_type: &str,
        lines: &[String],
    ) {
        let dir = projects_dir
            .join(project)
            .join(session_id)
            .join("subagents");
        std::fs::create_dir_all(&dir).expect("create subagents dir");
        let mut file =
            std::fs::File::create(dir.join(format!("{agent_file}.jsonl"))).expect("create jsonl");
        for line in lines {
            writeln!(file, "{line}").expect("write line");
        }
        std::fs::write(
            dir.join(format!("{agent_file}.meta.json")),
            format!(r#"{{"agentType":"{agent_type}"}}"#),
        )
        .expect("write sidecar");
    }

    #[test]
    fn load_agents_deduplicates_within_one_transcript() {
        // The figure every other number in the view rests on: one
        // `(message_id, request_id)` pair repeated three times is
        // billed once.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let repeated = assistant_line("2026-04-01T10:05:00Z", "msg_x", "req_x", "m", 700, 0);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "dedup-agent",
            &[repeated.clone(), repeated.clone(), repeated],
        );

        let rows = load_agents(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_agents");
        assert_eq!(rows.len(), 1, "expected one row, got {rows:#?}");
        // 700 input + 5 ephemeral_5m, counted once rather than thrice.
        assert_eq!(rows[0].usage.input, 700);
        assert_eq!(rows[0].usage.cache_creation.ephemeral_5m, 5);
        assert_eq!(rows[0].dispatches, 1);
    }

    #[test]
    fn load_sessions_counts_a_split_subagent_message_once() {
        // One message written as three lines, each repeating the
        // input and carrying a growing output count: billed once, at
        // the last line's output.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let line =
            |output| assistant_line("2026-04-01T10:05:00Z", "msg_x", "req_x", "m", 700, output);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "split-agent",
            &[line(1), line(2), line(30)],
        );

        let sessions = load_sessions(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].total_billable, (1000 + 50 + 5) + (700 + 30 + 5));
    }

    #[test]
    fn load_agents_bills_a_split_message_at_its_last_output() {
        // Earlier lines of a split message hold partial output counts;
        // the dispatch must carry the final one.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let line =
            |output| assistant_line("2026-04-01T10:05:00Z", "msg_x", "req_x", "m", 700, output);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "split-agent",
            &[line(1), line(2), line(30)],
        );

        let rows = load_agents(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_agents");
        assert_eq!(rows.len(), 1, "expected one row, got {rows:#?}");
        assert_eq!(rows[0].usage.input, 700);
        assert_eq!(rows[0].usage.output, 30);
    }

    #[test]
    fn subagent_majority_model_counts_messages_not_lines() {
        // Three lines of one opus message must cast one vote, so the
        // two sonnet messages win the majority.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let opus = assistant_line(
            "2026-04-01T10:05:00Z",
            "msg_1",
            "req_1",
            "claude-opus-4-7",
            10,
            1,
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "vote-agent",
            &[
                opus.clone(),
                opus.clone(),
                opus,
                assistant_line(
                    "2026-04-01T10:06:00Z",
                    "msg_2",
                    "req_2",
                    "claude-sonnet-4-6",
                    10,
                    1,
                ),
                assistant_line(
                    "2026-04-01T10:07:00Z",
                    "msg_3",
                    "req_3",
                    "claude-sonnet-4-6",
                    10,
                    1,
                ),
            ],
        );

        let path = projects_dir.join("alpha/s1/subagents/agent-1.jsonl");
        let turns = parse_jsonl(&path).expect("parse").turns;
        let meta = session_meta_from_turns(
            SessionKind::Subagent {
                agent_type: "vote-agent".to_string(),
            },
            "agent-1".to_string(),
            None,
            "alpha",
            &turns,
        )
        .expect("meta");
        assert_eq!(meta.model.as_deref(), Some("claude-sonnet-4-6"));
    }

    #[test]
    fn load_agents_counts_a_shared_key_once_per_project() {
        // Two transcripts in one session sharing a key: the second is
        // a copy of content the first contributed, so dedup empties it
        // and the walk omits it.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let shared = assistant_line("2026-04-01T10:05:00Z", "msg_x", "req_x", "m", 700, 0);
        for agent_file in ["agent-1", "agent-2"] {
            write_subagent(
                &projects_dir,
                "alpha",
                "s1",
                agent_file,
                "shared-key-agent",
                std::slice::from_ref(&shared),
            );
        }

        let rows = load_agents(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_agents");
        assert_eq!(rows.len(), 1, "got {rows:#?}");
        assert_eq!(rows[0].dispatches, 1);
        assert_eq!(rows[0].usage.input, 700);
    }

    #[test]
    fn load_agents_counts_a_shared_key_once_in_each_project() {
        // The seen-set is per project: the same transcript in two
        // projects counts in both.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_session(
            &tmp,
            "beta",
            "s2",
            "/home/user/beta",
            "2026-04-02T10:00:00Z",
            1000,
        );
        let shared = assistant_line("2026-04-01T10:05:00Z", "msg_x", "req_x", "m", 700, 0);
        for (project, session) in [("alpha", "s1"), ("beta", "s2")] {
            write_subagent(
                &projects_dir,
                project,
                session,
                "agent-1",
                "shared-key-agent",
                std::slice::from_ref(&shared),
            );
        }

        let rows = load_agents(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_agents");
        assert_eq!(rows.len(), 1, "same agent, model, effort, pinning");
        assert_eq!(rows[0].dispatches, 2);
        assert_eq!(rows[0].usage.input, 1400);
    }

    #[test]
    fn load_agents_skips_a_subagent_with_no_sidecar() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let dir = projects_dir.join("alpha").join("s1").join("subagents");
        std::fs::create_dir_all(&dir).expect("create subagents dir");
        std::fs::write(
            dir.join("agent-1.jsonl"),
            format!(
                "{}\n",
                assistant_line("2026-04-01T10:05:00Z", "m1", "r1", "m", 700, 0)
            ),
        )
        .expect("write orphan transcript");

        let rows = load_agents(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_agents");
        assert!(
            rows.is_empty(),
            "a sidecar-less subagent is skipped: {rows:#?}"
        );
    }

    #[test]
    fn load_agents_respects_project_filter() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_session(
            &tmp,
            "beta",
            "s2",
            "/home/user/beta",
            "2026-04-02T10:00:00Z",
            1000,
        );
        let line = assistant_line("2026-04-01T10:05:00Z", "m1", "r1", "m", 700, 0);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "alpha-agent",
            std::slice::from_ref(&line),
        );
        write_subagent(
            &projects_dir,
            "beta",
            "s2",
            "agent-1",
            "beta-agent",
            std::slice::from_ref(&line),
        );

        let mut data_ctx = ctx_with_claude_home(projects_dir, &empty_claude_home(&tmp));
        data_ctx.query.sessions.project_name = Some("alpha".to_string());
        let rows = load_agents(&data_ctx).expect("load_agents");
        assert_eq!(rows.len(), 1, "got {rows:#?}");
        assert_eq!(rows[0].agent_type, "alpha-agent");
    }

    #[test]
    fn load_agents_respects_thresholds() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "small-agent",
            &[assistant_line(
                "2026-04-01T10:05:00Z",
                "m1",
                "r1",
                "m",
                10,
                0,
            )],
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-2",
            "big-agent",
            &[assistant_line(
                "2026-04-01T10:06:00Z",
                "m2",
                "r2",
                "m",
                5000,
                0,
            )],
        );

        let mut data_ctx = ctx_with_claude_home(projects_dir, &empty_claude_home(&tmp));
        data_ctx.query.thresholds.min_tokens = Some(1000);
        let rows = load_agents(&data_ctx).expect("load_agents");
        assert_eq!(rows.len(), 1, "got {rows:#?}");
        assert_eq!(rows[0].agent_type, "big-agent");
    }

    #[test]
    fn load_agents_default_pinning_excludes_pinned_rows() {
        // A project-local agent file naming a concrete model. It
        // enters the inventory through `walk_for_session` on the
        // dispatch's own cwd, so this stays hermetic without touching
        // `CCLENS_CLAUDE_HOME`.
        let tmp = tempfile::tempdir().expect("tempdir");
        let work = tmp.path().join("work");
        std::fs::create_dir_all(work.join(".claude/agents")).expect("create agent dir");
        std::fs::write(
            work.join(".claude/agents/pinned-agent.md"),
            "---\nmodel: claude-opus-5\n---\n\nBody.\n",
        )
        .expect("write agent file");

        let cwd = work.to_string_lossy().into_owned();
        let projects_dir = write_session(&tmp, "alpha", "s1", &cwd, "2026-04-01T10:00:00Z", 1000);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "pinned-agent",
            &[assistant_line(
                "2026-04-01T10:05:00Z",
                "m1",
                "r1",
                "m",
                700,
                0,
            )],
        );

        let mut data_ctx = ctx_with_claude_home(projects_dir, &empty_claude_home(&tmp));
        let rows = load_agents(&data_ctx).expect("load_agents");
        assert!(
            rows.is_empty(),
            "the default slice hides pinned rows, got {rows:#?}",
        );

        // Widening admits it, and it classifies as pinned — which is
        // what proves the default excluded it for the right reason.
        data_ctx.query.pinning = PinningFilter::everything();
        let rows = load_agents(&data_ctx).expect("load_agents");
        assert_eq!(rows.len(), 1, "got {rows:#?}");
        assert_eq!(
            rows[0].pinning,
            crate::agents::Pinning::Pinned {
                declared: "claude-opus-5".to_string()
            },
        );
    }

    #[test]
    fn query_describe_active_orders_session_scope_thresholds_then_pinning() {
        let query = Query {
            sessions: SessionFilter {
                project_name: Some("alpha".to_string()),
                // Through the parser: the instant that renders as a
                // bare date is the local day start.
                since: crate::filter::parse_filter_datetime("2026-04-10").ok(),
                until: None,
            },
            thresholds: ThresholdsFilter {
                min_tokens: Some(50_000),
                min_cost: None,
            },
            inputs_session_id: Some("abc".to_string()),
            // Non-default, or the component would (correctly) not
            // appear at all — see `PinningFilter::describe_active`.
            pinning: PinningFilter::new(&[crate::agents::PinningKind::Pinned]),
        };
        let components = query.describe_active();
        assert_eq!(
            components
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>(),
            vec![
                "--session abc",
                "--project alpha",
                "--since 2026-04-10",
                "--min-tokens 50000",
                "--pinning pinned",
            ],
        );
        // `--session` is read by the inputs loader alone, scope
        // filters never reach `load_show`, thresholds constrain every
        // loader, and `--pinning` reaches the agents loader alone.
        assert_eq!(
            components.iter().map(|c| c.honored_by).collect::<Vec<_>>(),
            vec![
                HonoredBy::INPUTS_ONLY,
                HonoredBy::SESSION_SCOPED,
                HonoredBy::SESSION_SCOPED,
                HonoredBy::EVERY_LOADER,
                HonoredBy::AGENTS_ONLY,
            ],
        );
    }

    /// The assertion that proves the pinning filter's *narrowing*
    /// default did not silently make every view look filtered.
    #[test]
    fn query_describe_active_is_empty_when_no_filter_is_set() {
        assert!(Query::default().describe_active().is_empty());
    }

    // ---- lineage_order ----

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn lineage_order_keeps_mtime_order_without_lineage() {
        assert_eq!(
            lineage_order(&ids(&["a", "b", "c"]), &[vec![], vec![], vec![]]),
            vec![0, 1, 2],
        );
    }

    #[test]
    fn lineage_order_moves_original_before_its_continuation() {
        assert_eq!(
            lineage_order(&ids(&["cont", "orig"]), &[vec![], ids(&["cont"])]),
            vec![1, 0],
        );
    }

    #[test]
    fn lineage_order_follows_a_chain() {
        // a → b → c, discovered newest first.
        assert_eq!(
            lineage_order(&ids(&["c", "b", "a"]), &[vec![], ids(&["c"]), ids(&["b"])]),
            vec![2, 1, 0],
        );
    }

    #[test]
    fn lineage_order_resolves_a_cycle_in_mtime_order() {
        assert_eq!(
            lineage_order(&ids(&["x", "y"]), &[ids(&["y"]), ids(&["x"])]),
            vec![0, 1],
        );
    }

    #[test]
    fn lineage_order_ignores_ids_outside_the_project() {
        assert_eq!(lineage_order(&ids(&["a"]), &[ids(&["zzz"])]), vec![0]);
    }

    #[test]
    fn lineage_order_counts_a_repeated_successor_once() {
        // `a` names `b` twice. Counted twice, `b` would still wait on
        // `a` after `a` is placed, letting `c` jump ahead of it.
        assert_eq!(
            lineage_order(&ids(&["b", "a", "c"]), &[vec![], ids(&["b", "b"]), vec![]]),
            vec![1, 0, 2],
        );
    }

    #[test]
    fn lineage_order_ignores_a_self_edge() {
        // Counted, the self-edge would leave `a` waiting on itself and
        // let `b` go first.
        assert_eq!(
            lineage_order(&ids(&["a", "b"]), &[ids(&["a"]), vec![]]),
            vec![0, 1],
        );
    }

    #[test]
    fn lineage_order_cycle_fallback_takes_the_earliest_remaining_item() {
        // `x` and `y` continue each other; `d` continues `x`. Nothing is
        // free, so the fallback emits `d` first even though it is not
        // on the cycle — the documented limit for malformed data.
        assert_eq!(
            lineage_order(
                &ids(&["d", "x", "y"]),
                &[vec![], ids(&["d", "y"]), ids(&["x"])],
            ),
            vec![0, 1, 2],
        );
    }

    /// One `continued-in` record naming `target`.
    fn continued_in_line(ts: &str, from: &str, target: &str) -> String {
        format!(
            r#"{{"type":"continued-in","timestamp":"{ts}","sessionId":"{from}","continuedInSessionId":"{target}"}}"#
        )
    }

    // ---- ProjectWalk ----

    /// A user line carrying a line `uuid`, which `user_line` omits.
    fn user_line_with_uuid(ts: &str, uuid: &str, content: &str) -> String {
        format!(
            r#"{{"type":"user","uuid":"{uuid}","timestamp":"{ts}","cwd":"/home/user/alpha","message":{{"role":"user","content":"{content}"}}}}"#
        )
    }

    fn append_lines(path: &Path, lines: &[String]) {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open for append");
        for line in lines {
            writeln!(file, "{line}").expect("append line");
        }
    }

    fn walk_project(root: &Path, project: &str) -> Vec<WalkedSession> {
        let wanted = root.join(project);
        let sessions = discover(root)
            .expect("discover")
            .into_iter()
            .find(|p| p.project_dir == wanted)
            .expect("project discovered")
            .sessions;
        ProjectWalk::new(&wanted, sessions).collect()
    }

    #[test]
    fn project_walk_drops_replayed_user_lines_by_uuid() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let project_dir = tmp.path().join("projects").join("alpha");
        std::fs::create_dir_all(&project_dir).expect("create project dir");
        let s1 = project_dir.join("s1.jsonl");
        let s2 = project_dir.join("s2.jsonl");
        let replayed = user_line_with_uuid("2026-04-01T10:00:00Z", "u-1", "original prompt");
        std::fs::write(&s2, format!("{replayed}\n")).expect("write s2");
        std::fs::write(
            &s1,
            format!(
                "{replayed}\n{}\n",
                continued_in_line("2026-04-01T10:05:00Z", "s1", "s2")
            ),
        )
        .expect("write s1");
        // `s1` is newer, so only lineage can put it first.
        filetime::set_file_mtime(&s2, filetime::FileTime::from_unix_time(1_000, 0))
            .expect("set s2 mtime");
        filetime::set_file_mtime(&s1, filetime::FileTime::from_unix_time(2_000, 0))
            .expect("set s1 mtime");

        let walked = walk_project(&tmp.path().join("projects"), "alpha");
        assert_eq!(
            walked
                .iter()
                .map(|w| w.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["s1", "s2"],
        );
        let has_prompt = |w: &WalkedSession| {
            w.parent
                .as_ref()
                .expect("readable")
                .iter()
                .any(|t| t.uuid.as_deref() == Some("u-1"))
        };
        assert!(has_prompt(&walked[0]));
        assert!(!has_prompt(&walked[1]));
    }

    #[test]
    fn project_walk_omits_a_fully_counted_subagent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_session(
            &tmp,
            "alpha",
            "s2",
            "/home/user/alpha",
            "2026-04-01T11:00:00Z",
            1000,
        );
        append_lines(
            &projects_dir.join("alpha/s1.jsonl"),
            &[continued_in_line("2026-04-01T10:30:00Z", "s1", "s2")],
        );
        let lines = [
            user_line_with_uuid("2026-04-01T10:01:00Z", "su-1", "probe"),
            assistant_line("2026-04-01T10:02:00Z", "msg_s", "req_s", "m", 50, 5),
        ];
        for session in ["s1", "s2"] {
            write_subagent(
                &projects_dir,
                "alpha",
                session,
                "agent-1",
                "linked-agent",
                &lines,
            );
        }

        let walked = walk_project(&projects_dir, "alpha");
        let by_id = |id: &str| {
            walked
                .iter()
                .find(|w| w.session_id == id)
                .expect("session walked")
        };
        assert_eq!(by_id("s1").subagents.len(), 1);
        assert!(by_id("s2").subagents.is_empty());
    }

    #[test]
    fn project_walk_credits_a_forks_replay_to_its_parent_session() {
        // A top-level fork replays the session's own prompt and reply.
        // The fork's file is older, so only the walk feeding the parent
        // before its subagents can make the parent own them.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = tmp.path().join("projects");
        let session_dir = projects_dir.join("alpha");
        std::fs::create_dir_all(&session_dir).expect("create project dir");
        let prompt = user_line_with_uuid("2026-04-01T10:00:00Z", "u-p", "go");
        let reply = assistant_line("2026-04-01T10:00:10Z", "msg_p", "req_p", "m", 50, 5);
        std::fs::write(session_dir.join("s1.jsonl"), format!("{prompt}\n{reply}\n"))
            .expect("write s1");
        let own = assistant_line("2026-04-01T10:01:00Z", "msg_f", "req_f", "m", 20, 2);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-f",
            "fork",
            &[prompt, reply, own],
        );
        let dir = projects_dir.join("alpha/s1/subagents");
        std::fs::write(
            dir.join("agent-f.meta.json"),
            r#"{"agentType":"fork","isFork":true}"#,
        )
        .expect("write fork sidecar");
        filetime::set_file_mtime(
            dir.join("agent-f.jsonl"),
            filetime::FileTime::from_unix_time(1_000, 0),
        )
        .expect("set fork mtime");
        filetime::set_file_mtime(
            session_dir.join("s1.jsonl"),
            filetime::FileTime::from_unix_time(2_000, 0),
        )
        .expect("set parent mtime");

        let walked = walk_project(&projects_dir, "alpha");
        let parent = walked[0].parent.as_ref().expect("parent readable");
        assert_eq!(parent.len(), 2);
        let fork = &walked[0].subagents[0].turns;
        let message_ids: Vec<_> = fork
            .iter()
            .filter_map(|t| t.message_id.as_deref())
            .collect();
        assert_eq!(message_ids, vec!["msg_f"]);
        assert_eq!(fork.len(), 1, "the replayed prompt must drop too");
    }

    #[test]
    fn project_walk_keeps_a_subagent_that_never_responded() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "silent-agent",
            &[user_line_with_uuid(
                "2026-04-01T10:01:00Z",
                "fresh-1",
                "probe",
            )],
        );

        let walked = walk_project(&projects_dir, "alpha");
        assert_eq!(walked[0].subagents.len(), 1);
        assert_eq!(walked[0].subagents[0].turns.len(), 1);
    }

    #[test]
    fn project_walk_unreadable_parent_still_feeds_its_subagents() {
        // Built by hand: `discover` skips anything that is not a file,
        // so it cannot hand the walk an unreadable parent.
        let tmp = tempfile::tempdir().expect("tempdir");
        let project_dir = tmp.path().join("alpha");
        std::fs::create_dir_all(&project_dir).expect("create project dir");
        let subagent = |session: &str| {
            let dir = project_dir.join(session).join("subagents");
            std::fs::create_dir_all(&dir).expect("create subagents dir");
            let jsonl = dir.join("agent-1.jsonl");
            let meta = dir.join("agent-1.meta.json");
            std::fs::write(
                &jsonl,
                format!(
                    "{}\n",
                    assistant_line("2026-04-01T10:02:00Z", "msg_s", "req_s", "m", 50, 5)
                ),
            )
            .expect("write transcript");
            std::fs::write(&meta, r#"{"agentType":"linked-agent"}"#).expect("write sidecar");
            SubagentPaths {
                jsonl,
                meta: Some(meta),
            }
        };
        let s2_jsonl = project_dir.join("s2.jsonl");
        std::fs::write(
            &s2_jsonl,
            format!(
                "{}\n",
                user_line("2026-04-01T11:00:00Z", "/home/user/alpha", "hello")
            ),
        )
        .expect("write s2");
        let sessions = vec![
            SessionPaths {
                // Never written, so `parse_jsonl` fails with NotFound.
                jsonl: project_dir.join("s1.jsonl"),
                subagents: vec![subagent("s1")],
            },
            SessionPaths {
                jsonl: s2_jsonl,
                subagents: vec![subagent("s2")],
            },
        ];

        let walked: Vec<WalkedSession> = ProjectWalk::new(&project_dir, sessions).collect();
        assert_eq!(walked[0].session_id, "s1");
        assert!(walked[0].parent.is_none());
        assert_eq!(walked[0].subagents.len(), 1);
        assert_eq!(walked[1].session_id, "s2");
        assert!(walked[1].subagents.is_empty());
    }

    #[test]
    fn project_walk_credits_a_forks_replayed_messages_to_its_parent_agent() {
        // A nested fork replays its parent agent's opening message. The
        // fork's file is older, so only `parentAgentId` lineage can make
        // the parent agent own that message.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let shared = assistant_line(
            "2026-04-01T10:01:00Z",
            "msg_shared",
            "req_shared",
            "m",
            50,
            5,
        );
        let own = assistant_line("2026-04-01T10:02:00Z", "msg_fork", "req_fork", "m", 20, 2);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-ap",
            "general-purpose",
            std::slice::from_ref(&shared),
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-af",
            "fork",
            &[shared.clone(), own],
        );
        let dir = projects_dir.join("alpha/s1/subagents");
        std::fs::write(
            dir.join("agent-af.meta.json"),
            r#"{"agentType":"fork","isFork":true,"parentAgentId":"ap"}"#,
        )
        .expect("write fork sidecar");
        filetime::set_file_mtime(
            dir.join("agent-af.jsonl"),
            filetime::FileTime::from_unix_time(1_000, 0),
        )
        .expect("set fork mtime");
        filetime::set_file_mtime(
            dir.join("agent-ap.jsonl"),
            filetime::FileTime::from_unix_time(2_000, 0),
        )
        .expect("set parent agent mtime");

        let walked = walk_project(&projects_dir, "alpha");
        let message_ids = |id: &str| -> Vec<String> {
            walked[0]
                .subagents
                .iter()
                .find(|s| s.id == id)
                .unwrap_or_else(|| panic!("{id} walked"))
                .turns
                .iter()
                .filter_map(|t| t.message_id.clone())
                .collect()
        };
        assert_eq!(message_ids("agent-ap"), vec!["msg_shared".to_string()]);
        assert_eq!(message_ids("agent-af"), vec!["msg_fork".to_string()]);
    }

    #[test]
    fn project_walk_reads_a_subagents_cwd_before_dedup() {
        // The fork's only cwd-bearing line replays its parent agent's,
        // so dedup drops it; the walk must still report the fork's own
        // cwd rather than the session's.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/other",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let shared = user_line_with_uuid("2026-04-01T10:01:00Z", "u-shared", "go");
        let own = assistant_line("2026-04-01T10:02:00Z", "msg_fork", "req_fork", "m", 20, 2);
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-ap",
            "general-purpose",
            std::slice::from_ref(&shared),
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-af",
            "fork",
            &[shared.clone(), own],
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-cwdless",
            "x",
            &[assistant_line(
                "2026-04-01T10:03:00Z",
                "msg_x",
                "req_x",
                "m",
                1,
                1,
            )],
        );
        std::fs::write(
            projects_dir.join("alpha/s1/subagents/agent-af.meta.json"),
            r#"{"agentType":"fork","isFork":true,"parentAgentId":"ap"}"#,
        )
        .expect("write fork sidecar");

        let walked = walk_project(&projects_dir, "alpha");
        let subagent = |id: &str| {
            walked[0]
                .subagents
                .iter()
                .find(|s| s.id == id)
                .unwrap_or_else(|| panic!("{id} walked"))
        };
        let fork = subagent("agent-af");
        assert!(fork.turns.iter().all(|t| t.cwd.is_none()));
        assert_eq!(fork.cwd.as_deref(), Some(Path::new("/home/user/alpha")));
        assert_eq!(fork.short_name, "alpha");
        // A transcript recording no cwd takes the parent's.
        let cwdless = subagent("agent-cwdless");
        assert_eq!(cwdless.cwd.as_deref(), Some(Path::new("/home/user/other")));
        assert_eq!(cwdless.short_name, "other");
    }

    /// A project `proj-dir` whose session `s1` continues in `s2`. The
    /// continuation's only cwd-bearing line (`/home/user/alpha`) is a
    /// replay, which dedup drops, and its one subagent records no cwd.
    /// Each assistant line carries 5 tokens of 5m cache creation.
    /// Returns the projects root.
    fn write_continuation_with_replayed_cwd(tmp: &tempfile::TempDir) -> PathBuf {
        let session_dir = tmp.path().join("projects").join("proj-dir");
        std::fs::create_dir_all(&session_dir).expect("create project dir");
        let replayed = user_line_with_uuid("2026-04-01T10:00:00Z", "u-1", "hello");
        std::fs::write(
            session_dir.join("s1.jsonl"),
            format!(
                "{replayed}\n{}\n{}\n",
                assistant_line("2026-04-01T10:00:10Z", "msg_1", "req_1", "m", 10, 1),
                continued_in_line("2026-04-01T10:05:00Z", "s1", "s2"),
            ),
        )
        .expect("write s1");
        std::fs::write(
            session_dir.join("s2.jsonl"),
            format!(
                "{replayed}\n{}\n",
                assistant_line("2026-04-01T11:00:00Z", "msg_2", "req_2", "m", 10, 1),
            ),
        )
        .expect("write s2");
        let projects_dir = tmp.path().join("projects");
        write_subagent(
            &projects_dir,
            "proj-dir",
            "s2",
            "agent-1",
            "cwdless-agent",
            &[assistant_line(
                "2026-04-01T11:01:00Z",
                "msg_s",
                "req_s",
                "m",
                700,
                0,
            )],
        );
        projects_dir
    }

    /// A context over `write_continuation_with_replayed_cwd`'s tree,
    /// scoped to `--project alpha`.
    fn alpha_scoped_ctx(tmp: &tempfile::TempDir) -> DataContext {
        let projects_dir = write_continuation_with_replayed_cwd(tmp);
        let mut ctx = ctx_with_claude_home(projects_dir, &empty_claude_home(tmp));
        ctx.query.sessions.project_name = Some("alpha".to_string());
        ctx
    }

    // The parent's cwd and short name must come from the transcript
    // before dedup in every view, so `--project` scopes them alike.

    #[test]
    fn load_sessions_short_name_survives_dedup() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let sessions = load_sessions(&alpha_scoped_ctx(&tmp)).expect("load_sessions");
        let ids: Vec<&str> = sessions.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["s1", "s2"]);
    }

    #[test]
    fn load_inputs_parent_fallbacks_survive_dedup() {
        // s1's assistant line, s2's, and the subagent's: 5 tokens each.
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_, coverage) = load_inputs(&alpha_scoped_ctx(&tmp)).expect("load_inputs");
        assert_eq!(coverage.short_5m.observed_tokens, 15);
    }

    #[test]
    fn load_agents_parent_fallbacks_survive_dedup() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let rows = load_agents(&alpha_scoped_ctx(&tmp)).expect("load_agents");
        assert_eq!(rows.len(), 1, "got {rows:#?}");
        assert_eq!(rows[0].agent_type, "cwdless-agent");
    }

    #[cfg(unix)]
    #[test]
    fn project_walk_omits_a_symlinked_subagent_copy() {
        // Claude Code links a continued session's subagents as symlinks
        // into the original's directory.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        write_session(
            &tmp,
            "alpha",
            "s2",
            "/home/user/alpha",
            "2026-04-01T11:00:00Z",
            1000,
        );
        append_lines(
            &projects_dir.join("alpha/s1.jsonl"),
            &[continued_in_line("2026-04-01T10:30:00Z", "s1", "s2")],
        );
        write_subagent(
            &projects_dir,
            "alpha",
            "s1",
            "agent-1",
            "linked-agent",
            &[assistant_line(
                "2026-04-01T10:02:00Z",
                "msg_s",
                "req_s",
                "m",
                50,
                5,
            )],
        );
        let original = projects_dir.join("alpha/s1/subagents");
        let linked = projects_dir.join("alpha/s2/subagents");
        std::fs::create_dir_all(&linked).expect("create linked dir");
        for name in ["agent-1.jsonl", "agent-1.meta.json"] {
            std::os::unix::fs::symlink(original.join(name), linked.join(name)).expect("symlink");
        }

        let walked = walk_project(&projects_dir, "alpha");
        assert_eq!(walked[0].session_id, "s1");
        assert_eq!(walked[0].subagents.len(), 1);
        assert!(walked[1].subagents.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn project_walk_skips_a_dangling_symlinked_subagent() {
        // When the original session is deleted, a continuation's linked
        // transcript dangles; the walk skips it and carries on.
        let tmp = tempfile::tempdir().expect("tempdir");
        let projects_dir = write_session(
            &tmp,
            "alpha",
            "s1",
            "/home/user/alpha",
            "2026-04-01T10:00:00Z",
            1000,
        );
        let dir = projects_dir.join("alpha/s1/subagents");
        std::fs::create_dir_all(&dir).expect("create subagents dir");
        std::os::unix::fs::symlink(tmp.path().join("gone.jsonl"), dir.join("agent-1.jsonl"))
            .expect("symlink");
        std::fs::write(
            dir.join("agent-1.meta.json"),
            r#"{"agentType":"gone-agent"}"#,
        )
        .expect("write sidecar");

        let walked = walk_project(&projects_dir, "alpha");
        assert_eq!(walked.len(), 1);
        assert!(walked[0].parent.is_some());
        assert!(walked[0].subagents.is_empty());
        let sessions = load_sessions(&ctx_with_claude_home(
            projects_dir,
            &empty_claude_home(&tmp),
        ))
        .expect("load_sessions");
        assert_eq!(sessions.len(), 1);
    }
}
