mod cli;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cclens::aggregation::{aggregate, group_into_exchanges};
use cclens::discovery::discover;
use cclens::domain::{Role, Turn};
use cclens::parsing::parse_jsonl;
use cclens::pricing;
use cclens::rendering::{render_session, render_table};
use clap::Parser;
use cli::{Cli, Command, FilterArgs, PricingAction, emit_empty_result_hint};

fn main() -> anyhow::Result<()> {
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
    }
}

fn run_list(projects_dir: &Path, filters: FilterArgs) -> anyhow::Result<()> {
    let catalog = pricing::load_catalog();
    let project_entries = discover(projects_dir)?;
    let mut sessions = Vec::new();
    for (project_dir, jsonl_paths) in project_entries {
        // Cross-file dedup state, scoped to one project. Resumed
        // sessions replay prior assistant turns verbatim; this set
        // ensures any `(message.id, requestId)` pair contributes cost
        // exactly once across the project's `.jsonl` files. `discover`
        // already returned `jsonl_paths` in mtime-ascending order, so
        // the earliest file containing a given key wins.
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for jsonl_path in jsonl_paths {
            // A single unreadable file should not abort the whole listing.
            let Ok(turns) = parse_jsonl(&jsonl_path) else {
                continue;
            };
            let turns = dedup_assistant_turns(turns, &mut seen);
            let session_id = jsonl_path
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

/// Drop assistant turns whose `(message_id, request_id)` pair has
/// already been seen earlier in this project's file walk.
///
/// Mirrors ccusage's `createUniqueHash`: a turn missing either part of
/// the pair passes through (matches their `null`-on-partial-key rule).
/// Non-assistant turns (user, attachment, system, other) are unaffected
/// — they don't carry billable usage and are required for title
/// extraction and exchange grouping. Sidechain markers (`isSidechain`)
/// aren't visible at this layer (the parser doesn't surface the flag),
/// so every assistant turn is keyed on `(message_id, request_id)`
/// regardless of sidechain status.
fn dedup_assistant_turns(turns: Vec<Turn>, seen: &mut HashSet<(String, String)>) -> Vec<Turn> {
    turns
        .into_iter()
        .filter(|turn| match &turn.role {
            Role::Assistant => match (turn.message_id.as_ref(), turn.request_id.as_ref()) {
                (Some(mid), Some(rid)) => seen.insert((mid.clone(), rid.clone())),
                // Missing key — pass through unchanged (matches
                // ccusage's createUniqueHash returning null on
                // partial keys).
                _ => true,
            },
            Role::User | Role::Attachment | Role::System | Role::Other(_) => true,
        })
        .collect()
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
    let mut matched_project: Option<(PathBuf, Vec<PathBuf>)> = None;
    for (project_dir, jsonl_paths) in project_entries {
        if jsonl_paths.iter().any(|p| stem_matches(p, session_id)) {
            if matched_project.is_some() {
                anyhow::bail!("multiple sessions match id {session_id}");
            }
            matched_project = Some((project_dir, jsonl_paths));
        }
    }
    let Some((_project_dir, jsonl_paths)) = matched_project else {
        anyhow::bail!("no session matches id {session_id}");
    };

    // Walk the project's files in mtime-ascending order, threading the
    // same per-project dedup state used by `run_list`. Stop as soon as
    // we've processed the target file: files later in mtime order
    // can't affect the target's filtered turns.
    let catalog = pricing::load_catalog();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut target_turns: Option<Vec<Turn>> = None;
    for jsonl_path in jsonl_paths {
        let Ok(turns) = parse_jsonl(&jsonl_path) else {
            continue;
        };
        let turns = dedup_assistant_turns(turns, &mut seen);
        if stem_matches(&jsonl_path, session_id) {
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

#[cfg(test)]
mod tests {
    use cclens::domain::{CacheCreation, Usage};
    use serde_json::Value;

    use super::*;

    // --- test helpers ---

    fn user_string_turn(content: &str) -> Turn {
        Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::String(content.to_string())),
            cwd: None,
        }
    }

    fn assistant_turn_with_usage(input: u64, output: u64, cache_creation: u64) -> Turn {
        // Helper takes a flat u64 for backwards-compatible test
        // ergonomics; treated as 5m, matching the legacy wire scalar.
        Turn {
            timestamp: Some("2026-04-01T10:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input,
                output,
                cache_creation: CacheCreation {
                    ephemeral_5m: cache_creation,
                    ephemeral_1h: 0,
                },
                cache_read: 0,
            }),
            content: None,
            cwd: None,
        }
    }

    // --- dedup ---

    #[test]
    fn dedup_drops_duplicate_assistant_turn_within_run() {
        // Build two assistants with the same (mid, rid); a third with
        // a different pair; a fourth with a missing key (must pass);
        // a user turn (must pass regardless).
        let mut first_dup = assistant_turn_with_usage(1, 1, 0);
        first_dup.message_id = Some("m1".into());
        first_dup.request_id = Some("r1".into());
        let mut second_dup = assistant_turn_with_usage(2, 2, 0);
        second_dup.message_id = Some("m1".into());
        second_dup.request_id = Some("r1".into());
        let mut other = assistant_turn_with_usage(3, 3, 0);
        other.message_id = Some("m2".into());
        other.request_id = Some("r2".into());
        let mut partial = assistant_turn_with_usage(4, 4, 0);
        // Partial key — must pass through.
        partial.message_id = Some("m3".into());
        partial.request_id = None;
        let user = user_string_turn("hi");

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let kept =
            dedup_assistant_turns(vec![first_dup, second_dup, other, partial, user], &mut seen);

        // first_dup (kept), second_dup (dropped), other (kept),
        // partial (kept, missing key), user (kept)
        assert_eq!(kept.len(), 4);
        // Verify the key cache observed the two complete pairs.
        assert!(seen.contains(&("m1".to_string(), "r1".to_string())));
        assert!(seen.contains(&("m2".to_string(), "r2".to_string())));
        // Surviving assistant turns retain their hash keys.
        assert_eq!(kept[0].message_id.as_deref(), Some("m1"));
        assert_eq!(kept[0].request_id.as_deref(), Some("r1"));
    }

    #[test]
    fn dedup_drops_duplicate_assistant_across_separate_calls() {
        // The orchestration layer reuses one HashSet across multiple
        // parse_jsonl results — confirm the second call picks up the
        // first call's state.
        let mut first_call_turn = assistant_turn_with_usage(1, 1, 0);
        first_call_turn.message_id = Some("m1".into());
        first_call_turn.request_id = Some("r1".into());
        let mut second_call_turn = assistant_turn_with_usage(2, 2, 0);
        second_call_turn.message_id = Some("m1".into());
        second_call_turn.request_id = Some("r1".into());

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let first = dedup_assistant_turns(vec![first_call_turn], &mut seen);
        let second = dedup_assistant_turns(vec![second_call_turn], &mut seen);
        assert_eq!(first.len(), 1);
        assert!(second.is_empty(), "duplicate from second call must drop");
    }

    #[test]
    fn dedup_passes_through_sidechain_assistant_with_unique_key() {
        // Sidechain turns aren't special-cased — they're deduped on
        // their own keys like any other assistant turn. This guards
        // against a regression that filtered them by role flag.
        let mut sidechain = assistant_turn_with_usage(50, 0, 0);
        sidechain.message_id = Some("msc".into());
        sidechain.request_id = Some("rsc".into());

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let kept = dedup_assistant_turns(vec![sidechain], &mut seen);
        assert_eq!(kept.len(), 1);
    }
}
