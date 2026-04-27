mod cli;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cclens::aggregation::{aggregate, dedup_assistant_turns, group_into_exchanges};
use cclens::attribution::{
    SessionKind, SessionMeta, compute_coverage, compute_rows, extend_inventory_for_session,
    session_meta_from_turns,
};
use cclens::discovery::{
    ProjectSessions, SessionPaths, SubagentPaths, discover, read_subagent_meta,
};
use cclens::domain::Turn;
use cclens::inventory::{InventoryConfig, discover_inventory};
use cclens::parsing::parse_jsonl;
use cclens::pricing;
use cclens::rendering::{render_inputs, render_session, render_table};
use clap::{CommandFactory, Parser};
use clap_complete::CompleteEnv;
use cli::{
    Cli, Command, FilterArgs, InputsFilterArgs, PricingAction, emit_empty_result_hint,
    emit_inputs_empty_hint,
};

fn main() -> anyhow::Result<()> {
    CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::List {
        filters: FilterArgs::default(),
    }) {
        Command::List { filters } => run_list(&cli.projects_dir, filters),
        Command::Show {
            session_id,
            filters,
        } => run_show(&cli.projects_dir, &session_id, filters),
        Command::Pricing { action } => run_pricing(action),
        Command::Inputs {
            inputs_filters,
            filters,
        } => run_inputs(&cli.projects_dir, &inputs_filters, filters),
    }
}

fn run_list(projects_dir: &Path, filters: FilterArgs) -> anyhow::Result<()> {
    let catalog = pricing::load_catalog();
    let project_entries = discover(projects_dir)?;
    let mut sessions = Vec::new();
    for ProjectSessions {
        project_dir,
        sessions: session_paths,
    } in project_entries
    {
        // Cross-file dedup state, scoped to one project. Resumed
        // sessions replay prior assistant turns verbatim; this set
        // ensures any `(message.id, requestId)` pair contributes cost
        // exactly once across the project's `.jsonl` files. `discover`
        // already returned `sessions` in mtime-ascending order, so
        // the earliest file containing a given key wins. Subagent
        // JSONLs aren't part of the listing view — `list` is the
        // session browser and subagents are an internal partition of
        // a parent session, not a session in their own right.
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for SessionPaths { jsonl, .. } in session_paths {
            // A single unreadable file should not abort the whole listing.
            let Ok(turns) = parse_jsonl(&jsonl) else {
                continue;
            };
            let turns = dedup_assistant_turns(turns, &mut seen);
            let session_id = jsonl
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            // The new threshold filter sits *after* `aggregate`'s
            // existing zero-billable pre-pass, so it composes
            // additively rather than altering the "session is
            // meaningful" contract.
            if let Some(session) = aggregate(&project_dir, session_id, turns, &catalog)
                && filters
                    .thresholds()
                    .matches(session.total_billable, session.total_cost)
            {
                sessions.push(session);
            }
        }
    }
    sessions.sort_by_key(|s| s.started_at);
    println!("{}", render_table(&sessions));
    if sessions.is_empty() {
        emit_empty_result_hint(&filters);
    }
    Ok(())
}

fn run_inputs(
    projects_dir: &Path,
    inputs_filters: &InputsFilterArgs,
    filters: FilterArgs,
) -> anyhow::Result<()> {
    let catalog = pricing::load_catalog();
    let inventory_config = InventoryConfig::default();
    let mut inventory = discover_inventory(&inventory_config);
    let mut seen_inventory_paths: HashSet<PathBuf> = HashSet::new();
    for file in &inventory {
        seen_inventory_paths.insert(file.path.clone());
    }

    let project_entries = discover(projects_dir)?;
    let attribution_filter = inputs_filters.attribution_filter();
    let mut session_metas: Vec<SessionMeta> = Vec::new();
    for ProjectSessions {
        project_dir,
        sessions: session_paths,
    } in project_entries
    {
        // Same per-project cross-file dedup pattern as run_list/run_show.
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
            if !attribution_filter.accepts(&parent_meta) {
                // Whole session (parent + subagents) is filtered out
                // — `--session foo` and `--project bar` apply to the
                // parent session, and a subagent inherits its parent's
                // identity. Skipping here keeps subagent attribution
                // consistent with what the user asked for.
                continue;
            }
            // Extend the shared inventory with this session's
            // ancestor + project-local context files. Path-keyed
            // dedup means a shared CLAUDE.md across sibling cwds
            // appears in the inventory exactly once.
            if let Some(cwd) = &parent_meta.cwd {
                extend_inventory_for_session(
                    &mut inventory,
                    &mut seen_inventory_paths,
                    cwd,
                    &inventory_config,
                );
            }

            // Walk this parent's subagents via the per-subagent
            // helper, which encapsulates the parse → meta → cwd
            // fallback pipeline. v1 skips any subagent whose
            // `.meta.json` is absent or unreadable.
            //
            // The clone of cwd + short-name is gated on
            // `subagents.is_empty()` because the overwhelmingly
            // common case (a session that never spawned a subagent)
            // shouldn't pay for the parent-fallback machinery.
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
                // Extend the inventory for this subagent's cwd. Path-
                // keyed dedup makes same-cwd subagents (the common
                // case) free.
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

    let rows = compute_rows(inventory, &session_metas, &catalog);
    let coverage = compute_coverage(&session_metas, &rows);
    let thresholds = filters.thresholds();
    // Apply --min-tokens / --min-cost as a presentation-only row filter:
    // coverage stats reflect every session in scope, not just rows kept.
    let visible_rows: Vec<_> = rows
        .into_iter()
        .filter(|row| thresholds.matches(row.estimated_tokens_billed, row.attributed_cost))
        .collect();
    println!("{}", render_inputs(&visible_rows, &coverage));
    if visible_rows.is_empty() {
        emit_inputs_empty_hint(inputs_filters, &filters);
    }
    Ok(())
}

/// Read a single subagent's transcript and return its `SessionMeta`.
///
/// Phase 1 contract: an absent or unreadable `.meta.json` sidecar
/// causes the subagent to be skipped entirely (returns `None`). v2
/// could fall back to correlating against the parent's
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

fn run_pricing(action: PricingAction) -> anyhow::Result<()> {
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

fn run_show(projects_dir: &Path, session_id: &str, filters: FilterArgs) -> anyhow::Result<()> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        anyhow::bail!("session id must not be empty");
    }
    let project_entries = discover(projects_dir)?;

    // Locate the project that owns this session. A stem collision
    // *across* projects is unlikely (Claude Code uses fresh UUIDs)
    // but we keep the "multiple sessions match id …" error for
    // global ambiguity. Filenames within a single project directory
    // are unique, so per-project ambiguity isn't possible here.
    // Subagent JSONLs (under `<stem>/subagents/`) are intentionally
    // ignored — `show` is the per-session view, and subagents are
    // partitions of the parent session, not sessions in their own
    // right.
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

    // Walk the project's files in mtime-ascending order, threading the
    // same per-project dedup state used by `run_list`. Stop as soon as
    // we've processed the target file: files later in mtime order
    // can't affect the target's filtered turns.
    let catalog = pricing::load_catalog();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut target_turns: Option<Vec<Turn>> = None;
    for SessionPaths { jsonl, .. } in session_paths {
        let Ok(turns) = parse_jsonl(&jsonl) else {
            continue;
        };
        let turns = dedup_assistant_turns(turns, &mut seen);
        if stem_matches(&jsonl, session_id) {
            target_turns = Some(turns);
            break;
        }
    }
    let turns =
        target_turns.ok_or_else(|| anyhow::anyhow!("no session matches id {session_id}"))?;
    let exchanges = group_into_exchanges(&turns);
    let (rendered, rows_shown) = render_session(&exchanges, &catalog, filters.thresholds());
    println!("{rendered}");
    if rows_shown == 0 {
        emit_empty_result_hint(&filters);
    }
    Ok(())
}

fn stem_matches(path: &Path, session_id: &str) -> bool {
    path.file_stem()
        .is_some_and(|s| s.to_string_lossy() == session_id)
}
