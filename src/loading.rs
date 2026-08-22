//! Data-loading pipeline for the TUI (and, via the same functions, the
//! plain/JSON CLI paths). Composes `discovery` → `parsing` →
//! `aggregation` → `attribution` → `pricing` into view-ready data.
//!
//! Public API:
//! - `Query` — the mutable, user-facing slice of load configuration
//!   (`sessions: SessionFilter`, `thresholds: ThresholdsFilter`,
//!   `inputs_session_id: Option<String>`). The sole source of filter
//!   semantics across all three loaders.
//! - `DataContext` — everything a load needs (`projects_dir`, `catalog`,
//!   `query`). Cloned into each dispatched load; cheap (a `PathBuf`, an
//!   `Arc` bump, a small `Query`).
//! - `RefreshFingerprint` — file-size + mtime fingerprint for change
//!   detection, keyed by watched path.
//! - `PricingData` — owned pricing entries + cache staleness info,
//!   built once via `pricing_data` and shared by startup and refresh.
//! - `load_sessions(&DataContext) -> anyhow::Result<Vec<Session>>`
//! - `load_show(&DataContext, &str) -> anyhow::Result<Vec<PreparedExchange>>`
//! - `load_inputs(&DataContext) -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)>`
//! - `Query::describe_active` — every active filter as a flag-shaped
//!   `FilterComponent`, in display order (`--session`, scope,
//!   thresholds). One producer for the CLI hint and the TUI header.
//! - `build_fingerprint(&Path) -> anyhow::Result<RefreshFingerprint>`
//! - `pricing_data(&PricingCatalog) -> PricingData`
//! - `refresh_pricing() -> anyhow::Result<(Arc<PricingCatalog>, PricingData)>`

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use crate::aggregation::{
    PreparedExchange, aggregate, dedup_assistant_turns, group_into_exchanges, prepare_exchanges,
};
use crate::attribution::{
    AttributionRow, CoverageStats, InputsFilter, SessionKind, SessionMeta, compute_coverage,
    compute_rows, extend_inventory_for_session, session_meta_from_turns,
};
use crate::discovery::{
    ProjectSessions, SessionPaths, SubagentPaths, discover, read_subagent_meta,
};
use crate::domain::{Session, Turn, TurnOrigin};
use crate::filter::{FilterComponent, QueryScope, SessionFilter, ThresholdsFilter};
use crate::inventory::{InventoryConfig, discover_inventory};
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
}

impl Query {
    /// Every active filter as a flag-shaped component, in display
    /// order: `--session`, then scope, then thresholds. An empty
    /// vector means no filter is active — which is how both the CLI
    /// hint and the TUI empty states distinguish "filtered to nothing"
    /// from "nothing to show".
    #[must_use]
    pub fn describe_active(&self) -> Vec<FilterComponent> {
        let mut components = Vec::new();
        if let Some(id) = &self.inputs_session_id {
            components.push(FilterComponent {
                text: format!("--session {id}"),
                scoped_to: Some(QueryScope::Inputs),
            });
        }
        components.extend(self.sessions.describe_active());
        components.extend(self.thresholds.describe_active());
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
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for SessionPaths { jsonl, subagents } in session_paths {
            let Ok(turns) = parse_jsonl(&jsonl) else {
                continue;
            };
            let turns = dedup_assistant_turns(turns, &mut seen);
            let session_id = jsonl
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            let subagent_turn_lists: Vec<Vec<Turn>> =
                subagents.iter().filter_map(build_subagent_turns).collect();
            if let Some(session) = aggregate(
                &project_dir,
                session_id,
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
    let Some((_project_dir, session_paths)) = matched_project else {
        anyhow::bail!("no session matches id {session_id}");
    };

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut target: Option<(Vec<Turn>, Vec<SubagentPaths>)> = None;
    for SessionPaths { jsonl, subagents } in session_paths {
        let Ok(turns) = parse_jsonl(&jsonl) else {
            continue;
        };
        let turns = dedup_assistant_turns(turns, &mut seen);
        if stem_matches(&jsonl, session_id) {
            target = Some((turns, subagents));
            break;
        }
    }
    let (parent_turns, subagent_paths) =
        target.ok_or_else(|| anyhow::anyhow!("no session matches id {session_id}"))?;
    let subagent_turn_lists: Vec<Vec<Turn>> = subagent_paths
        .iter()
        .filter_map(build_subagent_turns)
        .collect();
    let mut all_exchanges = group_into_exchanges(&parent_turns);
    for sub_turns in &subagent_turn_lists {
        all_exchanges.extend(group_into_exchanges(sub_turns));
    }
    all_exchanges.sort_by_key(|ex| {
        ex.user
            .timestamp
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

    let inventory_config = InventoryConfig::default();
    let mut inventory = discover_inventory(&inventory_config);
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
        let mut seen_turn_keys: HashSet<(String, String)> = HashSet::new();
        for SessionPaths { jsonl, subagents } in session_paths {
            let Ok(turns) = parse_jsonl(&jsonl) else {
                continue;
            };
            let turns = dedup_assistant_turns(turns, &mut seen_turn_keys);
            let session_id = jsonl
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            let Some(parent_meta) =
                session_meta_from_turns(SessionKind::Parent, session_id, &project_dir, &turns)
            else {
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
                    &inventory_config,
                );
            }
            if subagents.is_empty() {
                session_metas.push(parent_meta);
                continue;
            }
            let parent_cwd = parent_meta.cwd.clone();
            let parent_short_name = parent_meta.project_short_name.clone();
            session_metas.push(parent_meta);
            for subagent in &subagents {
                let Some(sub_meta) = build_subagent_meta(
                    subagent,
                    &project_dir,
                    parent_cwd.as_deref(),
                    &parent_short_name,
                ) else {
                    continue;
                };
                if let Some(cwd) = &sub_meta.cwd {
                    extend_inventory_for_session(
                        &mut inventory,
                        &mut seen_inventory_paths,
                        cwd,
                        &inventory_config,
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

/// Read a single subagent's transcript and return its parsed `Turn`s,
/// each tagged with its `TurnOrigin::Subagent { agent_type, description }`.
///
/// The list/show pipeline needs the raw turn list (so a subagent's
/// billable contribution and cost can fold into `Session.total_*`,
/// and so the show renderer can render subagent rows inline); the
/// inputs pipeline keeps using `build_subagent_meta` for its
/// `SessionMeta` shape. The two helpers parallel each other in two
/// ways. First, both gate on the `.meta.json` sidecar — an absent or
/// unreadable sidecar returns `None` so the subagent is skipped.
/// Second, both skip the cross-file dedup pass — subagent transcripts
/// are non-resumable single-file, so the `(message_id, request_id)`
/// key set never fires across them.
///
/// The per-turn `agent_type` / `description` clones are bounded by
/// subagent transcript length (typically tens of turns) times short-
/// string copies (tens of bytes per field). Acceptable; an interning
/// scheme would be over-engineering for these sizes.
fn build_subagent_turns(subagent: &SubagentPaths) -> Option<Vec<Turn>> {
    let meta_path = subagent.meta.as_ref()?;
    let meta = read_subagent_meta(meta_path)?;
    let mut turns = parse_jsonl(&subagent.jsonl).ok()?;
    for turn in &mut turns {
        turn.origin = TurnOrigin::Subagent {
            agent_type: meta.agent_type.clone(),
            description: meta.description.clone(),
        };
    }
    Some(turns)
}

/// Read a single subagent's transcript and return its `SessionMeta`.
///
/// An absent or unreadable `.meta.json` sidecar causes the subagent
/// to be skipped entirely (returns `None`); a future version could
/// fall back to correlating against the parent's
/// `tool_result.agentId` if the sidecar shape regresses or older
/// installs ship without one — out of scope for now.
///
/// Also returns `None` when the JSONL parse fails or when the turn
/// list has no timestamps (`session_meta_from_turns`'s contract).
/// Subagent JSONLs typically inherit the parent's cwd in real Claude
/// Code data, but the contract doesn't *require* it — the parent's
/// cwd / short-name are passed in as fallbacks for the rare subagent
/// that has no recorded `Turn.cwd`.
fn build_subagent_meta(
    subagent: &SubagentPaths,
    project_dir: &Path,
    parent_cwd: Option<&Path>,
    parent_short_name: &str,
) -> Option<SessionMeta> {
    let meta_path = subagent.meta.as_ref()?;
    let subagent_meta_json = read_subagent_meta(meta_path)?;
    // Subagent JSONLs are non-resumable single-file transcripts, so
    // the cross-file `(message_id, request_id)` dedup pass that the
    // parent walk runs would never fire here. Skipping it keeps the
    // call site honest about the single-file nature of this read.
    let sub_turns = parse_jsonl(&subagent.jsonl).ok()?;
    let sub_session_id = subagent
        .jsonl
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut sub_meta = session_meta_from_turns(
        SessionKind::Subagent {
            agent_type: subagent_meta_json.agent_type,
        },
        sub_session_id,
        project_dir,
        &sub_turns,
    )?;
    if sub_meta.cwd.is_none()
        && let Some(cwd) = parent_cwd
    {
        sub_meta.cwd = Some(cwd.to_path_buf());
        sub_meta.project_short_name = parent_short_name.to_string();
    }
    Some(sub_meta)
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

    fn ctx(projects_dir: PathBuf) -> DataContext {
        DataContext {
            projects_dir,
            catalog: Arc::new(PricingCatalog::default()),
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

        let mut data_ctx = ctx(projects_dir);
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

        let mut data_ctx = ctx(projects_dir);
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

        let unfiltered = ctx(projects_dir.clone());
        let unfiltered_exchanges = load_show(&unfiltered, "s1").expect("load_show unfiltered");
        assert!(!unfiltered_exchanges.is_empty());

        let mut filtered = ctx(projects_dir);
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

        let mut data_ctx = ctx(projects_dir);
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
        // zero. This is deterministic and environment-independent —
        // it holds regardless of which inventory files exist here.
        let all_ctx = ctx(projects_dir.clone());
        let (all_rows, all_coverage) = load_inputs(&all_ctx).expect("load_inputs all");
        let all_loads: u64 = all_rows.iter().map(|r| r.loads_1h + r.loads_5m).sum();
        assert!(
            all_loads > 0,
            "sanity: two sessions with a global-scope-matching cwd should credit \
             at least one always-loaded inventory row",
        );

        let mut unmatched_ctx = ctx(projects_dir);
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

        let sessions = load_sessions(&ctx(projects_dir)).expect("load_sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "s1");
    }

    #[test]
    fn query_describe_active_orders_session_scope_then_thresholds() {
        let query = Query {
            sessions: SessionFilter {
                project_name: Some("alpha".to_string()),
                since: Some(
                    chrono::DateTime::parse_from_rfc3339("2026-04-10T00:00:00Z")
                        .unwrap()
                        .with_timezone(&chrono::Utc),
                ),
                until: None,
            },
            thresholds: ThresholdsFilter {
                min_tokens: Some(50_000),
                min_cost: None,
            },
            inputs_session_id: Some("abc".to_string()),
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
            ],
        );
        // Only `--session` is view-scoped; every other component
        // constrains all three loaders.
        assert_eq!(
            components.iter().map(|c| c.scoped_to).collect::<Vec<_>>(),
            vec![Some(QueryScope::Inputs), None, None, None],
        );
    }

    #[test]
    fn query_describe_active_is_empty_when_no_filter_is_set() {
        assert!(Query::default().describe_active().is_empty());
    }
}
