mod cli;

use std::collections::{HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use cclens::aggregation::{
    SessionSummary, aggregate, dedup_assistant_turns, group_into_exchanges, prepare_exchanges,
};
use cclens::attribution::{
    AttributionRow, CoverageStats, InputsFilter, SessionKind, SessionMeta, compute_coverage,
    compute_rows, extend_inventory_for_session, session_meta_from_turns,
};
use cclens::discovery::{
    ProjectSessions, SessionPaths, SubagentPaths, discover, read_subagent_meta,
};
use cclens::domain::{Session, Turn, TurnOrigin};
use cclens::filter::{SessionFilter, ThresholdsFilter};
use cclens::inventory::{InventoryConfig, discover_inventory};
use cclens::parsing::parse_jsonl;
use cclens::pricing;
use cclens::rendering::{render_inputs, render_prices, render_session, render_table};
use cclens::tui::{PricingData, RefreshFingerprint, Tab, run_tui};
use clap::{CommandFactory, Parser};
use clap_complete::CompleteEnv;
use cli::{
    Cli, Command, InputsArgs, OutputFormat, PricingAction, SessionFilterArgs, ThresholdsFilterArgs,
    emit_empty_result_hint, emit_inputs_empty_hint,
};
use serde::Serialize;

/// Resolved output mode. Computed once from `OutputFormat` (CLI surface)
/// and TTY detection; `Tui` is the auto-detected case when no `--format`
/// is given and stdout is a terminal.
#[derive(Clone, Copy)]
enum RenderMode {
    Tui,
    Plain,
    Json,
}

#[derive(Serialize)]
struct ShowOutput<'a> {
    session_id: &'a str,
    exchanges: &'a [cclens::aggregation::PreparedExchange],
}

#[derive(Serialize)]
struct InputsOutput<'a> {
    rows: &'a [AttributionRow],
    coverage: &'a CoverageStats,
}

#[derive(Serialize)]
struct PricingListEntry<'a> {
    model: &'a str,
    #[serde(flatten)]
    rates: &'a cclens::pricing::ClaudePricing,
}

fn main() -> anyhow::Result<()> {
    CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();
    let mode = match cli.format {
        Some(OutputFormat::Json) => RenderMode::Json,
        None if std::io::stdout().is_terminal() => RenderMode::Tui,
        // Explicit --format plain, or no flag with non-TTY stdout.
        Some(OutputFormat::Plain) | None => RenderMode::Plain,
    };
    match cli.command.unwrap_or(Command::List {
        scope: SessionFilterArgs::default(),
        thresholds: ThresholdsFilterArgs::default(),
    }) {
        Command::List { scope, thresholds } => {
            run_list(mode, &cli.projects_dir, &scope, thresholds)
        }
        Command::Show {
            session_id,
            thresholds,
        } => run_show(mode, &cli.projects_dir, &session_id, thresholds),
        Command::Pricing { action } => run_pricing(mode, action),
        Command::Inputs {
            scope,
            inputs,
            thresholds,
        } => run_inputs(mode, &cli.projects_dir, &scope, &inputs, thresholds),
    }
}

/// Returns sessions sorted by `started_at` (ascending).
fn load_sessions_data(
    projects_dir: &Path,
    session_filter: &SessionFilter,
    thresholds_filter: &ThresholdsFilter,
    catalog: &pricing::PricingCatalog,
) -> anyhow::Result<Vec<Session>> {
    let project_entries = discover(projects_dir)?;
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
                catalog,
            ) && session_filter.accepts(&session.project_short_name, session.started_at)
                && thresholds_filter.matches(
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

fn build_fingerprint(projects_dir: &Path) -> anyhow::Result<RefreshFingerprint> {
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

fn run_list(
    mode: RenderMode,
    projects_dir: &Path,
    scope: &SessionFilterArgs,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let catalog = Arc::new(pricing::load_catalog());
    let session_filter = scope.session_filter();
    let thresholds_filter = thresholds.thresholds_filter();
    let sessions = load_sessions_data(projects_dir, &session_filter, &thresholds_filter, &catalog)?;
    match mode {
        RenderMode::Tui if !sessions.is_empty() => {
            let pricing_data = PricingData {
                entries: catalog
                    .sorted_entries(false)
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), *v))
                    .collect(),
                cache_info: pricing::cache_info(),
            };
            let initial_fingerprint = build_fingerprint(projects_dir).unwrap_or_default();
            let projects_dir_owned = projects_dir.to_path_buf();
            let sessions_loader = {
                let pd = projects_dir_owned.clone();
                let cat = Arc::clone(&catalog);
                let sf = session_filter.clone();
                let tf = thresholds_filter;
                move || load_sessions_data(&pd, &sf, &tf, &cat)
            };
            let fp_builder = {
                let pd = projects_dir_owned.clone();
                move || build_fingerprint(&pd)
            };
            let show_loader = {
                let pd = projects_dir_owned.clone();
                let cat = Arc::clone(&catalog);
                move |session_id: &str| {
                    load_show_detail(&pd, session_id, &cat, ThresholdsFilter::default())
                }
            };
            let inputs_filter = InputsFilter {
                session_id: None,
                scope: session_filter,
            };
            let inputs_loader = {
                let pd = projects_dir_owned;
                let cat = Arc::clone(&catalog);
                move || load_inputs_data(&pd, &inputs_filter, &cat)
            };
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let result = rt.block_on(run_tui(
                sessions,
                show_loader,
                inputs_loader,
                sessions_loader,
                fp_builder,
                initial_fingerprint,
                pricing_data,
                Tab::Sessions,
            ));
            rt.shutdown_timeout(std::time::Duration::from_millis(100));
            result?;
        }
        RenderMode::Json => {
            let summaries: Vec<SessionSummary> =
                sessions.iter().map(SessionSummary::from).collect();
            println!("{}", serde_json::to_string_pretty(&summaries)?);
        }
        RenderMode::Tui | RenderMode::Plain => {
            println!("{}", render_table(&sessions));
            if sessions.is_empty() {
                emit_empty_result_hint(scope, &thresholds);
            }
        }
    }
    Ok(())
}

fn load_inputs_data(
    projects_dir: &Path,
    inputs_filter: &InputsFilter,
    catalog: &pricing::PricingCatalog,
) -> anyhow::Result<(Vec<AttributionRow>, CoverageStats)> {
    let inventory_config = InventoryConfig::default();
    let mut inventory = discover_inventory(&inventory_config);
    let mut seen_inventory_paths: HashSet<PathBuf> = HashSet::new();
    for file in &inventory {
        seen_inventory_paths.insert(file.path.clone());
    }

    let project_entries = discover(projects_dir)?;
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

    let rows = compute_rows(inventory, &session_metas, catalog);
    let coverage = compute_coverage(&session_metas, &rows);
    Ok((rows, coverage))
}

fn run_inputs(
    mode: RenderMode,
    projects_dir: &Path,
    scope: &SessionFilterArgs,
    inputs: &InputsArgs,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let catalog = Arc::new(pricing::load_catalog());
    let inputs_filter = InputsFilter {
        session_id: inputs.session_id(),
        scope: scope.session_filter(),
    };
    let thresholds_filter = thresholds.thresholds_filter();
    match mode {
        RenderMode::Tui => {
            let session_filter = scope.session_filter();
            let sessions = load_sessions_data(
                projects_dir,
                &session_filter,
                &ThresholdsFilter::default(),
                &catalog,
            )?;
            let pricing_data = PricingData {
                entries: catalog
                    .sorted_entries(false)
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), *v))
                    .collect(),
                cache_info: pricing::cache_info(),
            };
            let initial_fingerprint = build_fingerprint(projects_dir).unwrap_or_default();
            let projects_dir_owned = projects_dir.to_path_buf();
            let sessions_loader = {
                let pd = projects_dir_owned.clone();
                let cat = Arc::clone(&catalog);
                let sf = session_filter;
                move || load_sessions_data(&pd, &sf, &ThresholdsFilter::default(), &cat)
            };
            let fp_builder = {
                let pd = projects_dir_owned.clone();
                move || build_fingerprint(&pd)
            };
            let show_loader = {
                let pd = projects_dir_owned.clone();
                let cat = Arc::clone(&catalog);
                move |session_id: &str| {
                    load_show_detail(&pd, session_id, &cat, ThresholdsFilter::default())
                }
            };
            let inputs_loader = {
                let pd = projects_dir_owned;
                let cat = Arc::clone(&catalog);
                move || {
                    let (rows, coverage) = load_inputs_data(&pd, &inputs_filter, &cat)?;
                    let visible: Vec<_> = rows
                        .into_iter()
                        .filter(|r| {
                            thresholds_filter.matches(r.estimated_tokens_billed, r.attributed_cost)
                        })
                        .collect();
                    Ok((visible, coverage))
                }
            };
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let result = rt.block_on(run_tui(
                sessions,
                show_loader,
                inputs_loader,
                sessions_loader,
                fp_builder,
                initial_fingerprint,
                pricing_data,
                Tab::Inputs,
            ));
            rt.shutdown_timeout(std::time::Duration::from_millis(100));
            result?;
        }
        RenderMode::Json => {
            let (rows, coverage) = load_inputs_data(projects_dir, &inputs_filter, &catalog)?;
            let visible_rows: Vec<_> = rows
                .into_iter()
                .filter(|r| thresholds_filter.matches(r.estimated_tokens_billed, r.attributed_cost))
                .collect();
            let output = InputsOutput {
                rows: &visible_rows,
                coverage: &coverage,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        RenderMode::Plain => {
            let (rows, coverage) = load_inputs_data(projects_dir, &inputs_filter, &catalog)?;
            let visible_rows: Vec<_> = rows
                .into_iter()
                .filter(|r| thresholds_filter.matches(r.estimated_tokens_billed, r.attributed_cost))
                .collect();
            println!("{}", render_inputs(&visible_rows, &coverage));
            if visible_rows.is_empty() {
                emit_inputs_empty_hint(scope, inputs, &thresholds);
            }
        }
    }
    Ok(())
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

fn run_pricing(mode: RenderMode, action: PricingAction) -> anyhow::Result<()> {
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
        PricingAction::List { all } => {
            let catalog = pricing::load_catalog();
            let entries = catalog.sorted_entries(all);
            match mode {
                RenderMode::Json => {
                    let json_entries: Vec<PricingListEntry<'_>> = entries
                        .iter()
                        .map(|(model, rates)| PricingListEntry { model, rates })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&json_entries)?);
                }
                RenderMode::Tui | RenderMode::Plain => {
                    if entries.is_empty() {
                        eprintln!("note: pricing catalog is empty — try `cclens pricing refresh`");
                    } else {
                        println!("{}", render_prices(&entries));
                    }
                }
            }
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

fn load_show_detail(
    projects_dir: &Path,
    session_id: &str,
    catalog: &pricing::PricingCatalog,
    thresholds: ThresholdsFilter,
) -> anyhow::Result<Vec<cclens::aggregation::PreparedExchange>> {
    let project_entries = discover(projects_dir)?;

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
    Ok(prepare_exchanges(&all_exchanges, catalog, thresholds))
}

/// Render one session's per-exchange table.
///
/// Walks the parent JSONL (with the same per-project cross-file dedup
/// pass `run_list` runs) plus every subagent transcript discovered
/// under `<stem>/subagents/`. Subagent transcripts are parsed via
/// `build_subagent_turns`, which tags each turn with
/// `TurnOrigin::Subagent`. The renderer dispatches on origin to render
/// subagent exchanges as single rows (role `subagent`) inline with the
/// parent's exchanges, sorted by user-turn timestamp. The body's
/// `cumulative` column at the bottom equals what `cclens list` reports
/// for the same session — list/show consistency by construction.
fn run_show(
    mode: RenderMode,
    projects_dir: &Path,
    session_id: &str,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        anyhow::bail!("session id must not be empty");
    }
    let catalog = pricing::load_catalog();
    let prepared = load_show_detail(
        projects_dir,
        session_id,
        &catalog,
        thresholds.thresholds_filter(),
    )?;
    match mode {
        RenderMode::Json => {
            let output = ShowOutput {
                session_id,
                exchanges: &prepared,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        RenderMode::Tui | RenderMode::Plain => {
            let (rendered, _rows_shown) = render_session(&prepared);
            println!("{rendered}");
            if prepared.is_empty() {
                emit_empty_result_hint(&SessionFilterArgs::default(), &thresholds);
            }
        }
    }
    Ok(())
}

fn stem_matches(path: &Path, session_id: &str) -> bool {
    path.file_stem()
        .is_some_and(|s| s.to_string_lossy() == session_id)
}
