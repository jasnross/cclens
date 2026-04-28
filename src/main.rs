mod cli;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cclens::aggregation::{aggregate, dedup_assistant_turns, group_into_exchanges};
use cclens::attribution::{
    InputsFilter, SessionKind, SessionMeta, compute_coverage, compute_rows,
    extend_inventory_for_session, session_meta_from_turns,
};
use cclens::discovery::{
    ProjectSessions, SessionPaths, SubagentPaths, discover, read_subagent_meta,
};
use cclens::domain::{Turn, TurnOrigin};
use cclens::inventory::{InventoryConfig, discover_inventory};
use cclens::parsing::parse_jsonl;
use cclens::pricing;
use cclens::rendering::{render_inputs, render_session, render_table};
use clap::{CommandFactory, Parser};
use clap_complete::CompleteEnv;
use cli::{
    Cli, Command, InputsArgs, PricingAction, SessionFilterArgs, ThresholdsFilterArgs,
    emit_empty_result_hint, emit_inputs_empty_hint,
};

fn main() -> anyhow::Result<()> {
    CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::List {
        scope: SessionFilterArgs::default(),
        thresholds: ThresholdsFilterArgs::default(),
    }) {
        Command::List { scope, thresholds } => run_list(&cli.projects_dir, &scope, thresholds),
        Command::Show {
            session_id,
            thresholds,
        } => run_show(&cli.projects_dir, &session_id, thresholds),
        Command::Pricing { action } => run_pricing(action),
        Command::Inputs {
            scope,
            inputs,
            thresholds,
        } => run_inputs(&cli.projects_dir, &scope, &inputs, thresholds),
    }
}

fn run_list(
    projects_dir: &Path,
    scope: &SessionFilterArgs,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let catalog = pricing::load_catalog();
    let project_entries = discover(projects_dir)?;
    let session_filter = scope.session_filter();
    let thresholds_filter = thresholds.thresholds_filter();
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
        // JSONLs are folded into each parent session's totals via
        // `build_subagent_turns`; they are not subject to cross-file
        // dedup (each subagent transcript is single-file and
        // non-resumable, matching `build_subagent_meta`'s contract).
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for SessionPaths { jsonl, subagents } in session_paths {
            // A single unreadable file should not abort the whole listing.
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
            // Both filters sit *after* `aggregate`'s existing
            // zero-billable pre-pass, so they compose additively
            // rather than altering the "session is meaningful"
            // contract. Scope check runs before thresholds for
            // readability — both are cheap once `aggregate` has
            // produced the session, and the ordering matches how
            // the flags appear left-to-right in the user-facing
            // CLI surface.
            if let Some(session) = aggregate(
                &project_dir,
                session_id,
                turns,
                &subagent_turn_lists,
                &catalog,
            ) && session_filter.accepts(&session.project_short_name, session.started_at)
                && thresholds_filter.matches(session.total_billable, session.total_cost)
            {
                sessions.push(session);
            }
        }
    }
    sessions.sort_by_key(|s| s.started_at);
    println!("{}", render_table(&sessions));
    if sessions.is_empty() {
        emit_empty_result_hint(scope, &thresholds);
    }
    Ok(())
}

fn run_inputs(
    projects_dir: &Path,
    scope: &SessionFilterArgs,
    inputs: &InputsArgs,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let catalog = pricing::load_catalog();
    let inventory_config = InventoryConfig::default();
    let mut inventory = discover_inventory(&inventory_config);
    let mut seen_inventory_paths: HashSet<PathBuf> = HashSet::new();
    for file in &inventory {
        seen_inventory_paths.insert(file.path.clone());
    }

    let project_entries = discover(projects_dir)?;
    let inputs_filter = InputsFilter {
        session_id: inputs.session_id(),
        scope: scope.session_filter(),
    };
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
            if !inputs_filter.accepts(&parent_meta) {
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
    let thresholds_filter = thresholds.thresholds_filter();
    // Apply --min-tokens / --min-cost as a presentation-only row filter:
    // coverage stats reflect every session in scope, not just rows kept.
    let visible_rows: Vec<_> = rows
        .into_iter()
        .filter(|row| thresholds_filter.matches(row.estimated_tokens_billed, row.attributed_cost))
        .collect();
    println!("{}", render_inputs(&visible_rows, &coverage));
    if visible_rows.is_empty() {
        emit_inputs_empty_hint(scope, inputs, &thresholds);
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
    projects_dir: &Path,
    session_id: &str,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
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
    // Bind the per-transcript `Vec<Turn>` lists to outer-scope locals
    // so every `Exchange<'a>` borrows from the same lifetime; the
    // merged exchange list then lives across all of them.
    let subagent_turn_lists: Vec<Vec<Turn>> = subagent_paths
        .iter()
        .filter_map(build_subagent_turns)
        .collect();
    let mut all_exchanges = group_into_exchanges(&parent_turns);
    for sub_turns in &subagent_turn_lists {
        all_exchanges.extend(group_into_exchanges(sub_turns));
    }
    // Sort exchanges by user-turn timestamp. `unwrap_or(UNIX_EPOCH)`
    // mirrors `discovery::sort_by_mtime_asc`'s defensive default — in
    // observed Claude Code data substantive user turns always carry a
    // timestamp, but the fallback keeps the sort total and keeps the
    // renderer free of `.unwrap()` (banned by the `unwrap_used` lint).
    all_exchanges.sort_by_key(|ex| {
        ex.user
            .timestamp
            .unwrap_or(chrono::DateTime::<chrono::Utc>::UNIX_EPOCH)
    });
    let (rendered, rows_shown) =
        render_session(&all_exchanges, &catalog, thresholds.thresholds_filter());
    println!("{rendered}");
    if rows_shown == 0 {
        // `cclens show` deliberately excludes scope flags (its required
        // <session-id> argument already pins a single session), so the
        // default `SessionFilterArgs` reports `any_active() == false`
        // and the hint suppression path is preserved exactly.
        emit_empty_result_hint(&SessionFilterArgs::default(), &thresholds);
    }
    Ok(())
}

fn stem_matches(path: &Path, session_id: &str) -> bool {
    path.file_stem()
        .is_some_and(|s| s.to_string_lossy() == session_id)
}
