//! Two-level walk of `~/.claude/projects/<dir>/<session>.jsonl`, with a
//! per-session three-level walk into `<session_stem>/subagents/` for the
//! subagent transcripts recent Claude Code versions emit.
//!
//! Public API:
//! - `discover(&Path) -> anyhow::Result<Vec<ProjectSessions>>` — returns
//!   one `ProjectSessions` per project directory; each carries the parent
//!   jsonl files plus their per-session subagent walks. JSONL paths are
//!   sorted by mtime ascending (both top-level and per-subagent) so the
//!   cross-file dedup pass is deterministic.
//! - `ProjectSessions`, `SessionPaths`, `SubagentPaths`, `SubagentMeta`
//!   — typed records the discoverer emits.
//! - `read_subagent_meta(&Path) -> Option<SubagentMeta>` — read a
//!   subagent's `.meta.json` sidecar; returns `None` when absent or
//!   malformed (graceful degradation, the caller decides whether to
//!   skip the subagent). The returned `SubagentMeta` carries both
//!   `agent_type` and an optional `description` (the per-invocation
//!   summary surfaced in the show view).
//!
//! Per-entry, per-subdir, and per-subagent-walk read errors are silently
//! skipped; only a failure to read the top-level `projects_dir` propagates.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Deserialize;

/// One project directory (e.g. `~/.claude/projects/-Users-x-foo/`) with
/// every parent session JSONL it contains, each annotated with its
/// per-session subagent transcripts.
#[derive(Debug)]
pub struct ProjectSessions {
    pub project_dir: PathBuf,
    pub sessions: Vec<SessionPaths>,
}

/// One parent session JSONL plus any subagent transcripts found in its
/// sibling `<session_stem>/subagents/` directory.
#[derive(Debug)]
pub struct SessionPaths {
    pub jsonl: PathBuf,
    pub subagents: Vec<SubagentPaths>,
}

/// One subagent transcript at `<session_stem>/subagents/agent-<id>.jsonl`,
/// with an optional sibling `.meta.json` sidecar. The presence/absence of
/// `meta` reflects what's on disk — it stays `Some(_)` even if the file
/// turns out to be malformed; gating on parse-success is the caller's
/// concern (see `read_subagent_meta`).
#[derive(Debug)]
pub struct SubagentPaths {
    pub jsonl: PathBuf,
    pub meta: Option<PathBuf>,
}

/// Parsed contents of a subagent `.meta.json` sidecar.
///
/// The on-disk schema observed in real Claude Code data is
/// `{"agentType": "<name>", "description": "<text>"}`. cclens exposes
/// both fields: `agent_type` is consumed by the inputs pipeline for
/// agent-row crediting, and `description` is consumed by the show
/// renderer for per-invocation labels. Older sidecars without a
/// `description` field deserialize cleanly (the field is `Option`).
#[derive(Debug)]
pub struct SubagentMeta {
    pub agent_type: String,
    pub description: Option<String>,
}

/// On-disk shape of a subagent `.meta.json` sidecar. Unknown fields
/// (and any future additions) deserialize without error thanks to
/// serde's default behavior; `description` is `Option` so older
/// sidecars without it still parse.
#[derive(Deserialize)]
struct RawSubagentMeta {
    #[serde(rename = "agentType")]
    agent_type: String,
    #[serde(default)]
    description: Option<String>,
}

/// # Errors
///
/// Returns an error only if the top-level `projects_dir` itself cannot
/// be read (a user-facing config error). Per-entry and per-subdir read
/// failures are silently skipped so one unreadable project doesn't
/// hide the rest. Per-session subagent-walk failures are likewise
/// absorbed — a session with an unreadable `<stem>/subagents/`
/// directory simply yields `subagents: vec![]`.
pub fn discover(projects_dir: &Path) -> anyhow::Result<Vec<ProjectSessions>> {
    let mut result = Vec::new();
    // Propagate only failure to read the top-level projects dir itself — that's
    // a user-facing config error. Per-entry and per-subdir errors are skipped
    // so one unreadable project doesn't hide the rest.
    for entry in fs::read_dir(projects_dir)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(inner) = fs::read_dir(&path) else {
            continue;
        };
        let mut jsonl_paths = Vec::new();
        for session_entry in inner {
            let Ok(session_entry) = session_entry else {
                continue;
            };
            let session_path = session_entry.path();
            // Skip every subdirectory at this level — including the
            // per-session `<stem>/` directories that hold subagent
            // transcripts. Those directories are walked separately
            // *per parent jsonl* below, so we don't need them in the
            // top-level enumeration.
            if session_path.is_dir() {
                continue;
            }
            if session_path.extension().is_some_and(|ext| ext == "jsonl") {
                jsonl_paths.push(session_path);
            }
        }
        // Resumed sessions inherit cost from the original; ordering by
        // mtime ascending makes "earliest file wins" deterministic and
        // semantically meaningful for the cross-file dedup pass.
        sort_by_mtime_asc(&mut jsonl_paths, PathBuf::as_path);
        let sessions: Vec<SessionPaths> = jsonl_paths
            .into_iter()
            .map(|jsonl| {
                let subagents = discover_subagents(&jsonl);
                SessionPaths { jsonl, subagents }
            })
            .collect();
        result.push(ProjectSessions {
            project_dir: path,
            sessions,
        });
    }
    Ok(result)
}

/// Sort items by their associated path's `(mtime, path)` ascending,
/// using `extract` to project each item to the path the sort key
/// reads. Items whose metadata or `modified()` call fails sort to the
/// start (`UNIX_EPOCH`); they are then attempted by the consuming
/// reader (`parse_jsonl`) like any other path and skipped via the
/// existing per-file `let Ok(...) = ... else { continue }` handler if
/// genuinely unreadable. The `path` secondary key makes the order
/// fully deterministic when two files share an mtime (possible on
/// low-resolution filesystems or after `cp -p`); without it, ties
/// would fall back to `read_dir` insertion order, which the OS does
/// not specify.
fn sort_by_mtime_asc<T, F>(items: &mut [T], extract: F)
where
    F: Fn(&T) -> &Path,
{
    items.sort_by_cached_key(|item| {
        let path = extract(item);
        let mtime = fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        (mtime, path.to_path_buf())
    });
}

/// Probe `<session_jsonl_parent>/<session_stem>/subagents/` for
/// `agent-*.jsonl` transcripts and their `.meta.json` sidecars.
///
/// Returns an empty `Vec` when the probe directory is missing,
/// unreadable, or contains no `agent-*.jsonl` files (the common case
/// for sessions that never spawned a subagent). Subdirectories under
/// `subagents/` are skipped: real Claude Code data is flat per
/// `claude-devtools` (`SubagentLocator.ts:86-115` does not recurse).
fn discover_subagents(session_jsonl: &Path) -> Vec<SubagentPaths> {
    let Some(stem) = session_jsonl.file_stem() else {
        return Vec::new();
    };
    let Some(parent) = session_jsonl.parent() else {
        return Vec::new();
    };
    let subagents_dir = parent.join(stem).join("subagents");
    let Ok(entries) = fs::read_dir(&subagents_dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // Skip nested directories — flat-only assumption (validated
        // against real data + the upstream `SubagentLocator` reference).
        if path.is_dir() {
            continue;
        }
        if path.extension().is_none_or(|ext| ext != "jsonl") {
            continue;
        }
        // Only `agent-*.jsonl` files; ignore stray jsonls that don't
        // match the convention.
        if path
            .file_name()
            .is_none_or(|n| !n.to_string_lossy().starts_with("agent-"))
        {
            continue;
        }
        // The sidecar replaces `.jsonl` with `.meta.json`. Its presence
        // here only reflects what's on disk — `read_subagent_meta` is
        // the gate for whether the contents are usable.
        let meta_path = path.with_extension("meta.json");
        let meta = if meta_path.is_file() {
            Some(meta_path)
        } else {
            None
        };
        out.push(SubagentPaths { jsonl: path, meta });
    }
    sort_by_mtime_asc(&mut out, |sp| sp.jsonl.as_path());
    out
}

/// Read a subagent's `.meta.json` sidecar. Returns `None` when the
/// file is missing, unreadable, or fails to parse — callers (typically
/// `run_inputs`) treat that as "skip this subagent" in v1.
#[must_use]
pub fn read_subagent_meta(path: &Path) -> Option<SubagentMeta> {
    let body = fs::read_to_string(path).ok()?;
    let raw: RawSubagentMeta = serde_json::from_str(&body).ok()?;
    Some(SubagentMeta {
        agent_type: raw.agent_type,
        description: raw.description,
    })
}

#[cfg(test)]
mod tests {
    use std::fs as stdfs;

    use super::*;

    #[test]
    fn discover_finds_jsonl_files_two_levels_deep() {
        let tmp = tempfile::tempdir().unwrap();
        let project_a = tmp.path().join("-Users-alpha");
        let project_b = tmp.path().join("-Users-beta");
        stdfs::create_dir(&project_a).unwrap();
        stdfs::create_dir(&project_b).unwrap();

        for dir in [&project_a, &project_b] {
            stdfs::File::create(dir.join("session1.jsonl")).unwrap();
            stdfs::File::create(dir.join("session2.jsonl")).unwrap();
            // Non-jsonl file — should be skipped.
            stdfs::File::create(dir.join("sessions-index.json")).unwrap();
            // Nested directory — should be skipped at the project-dir
            // level; per-session subagent walks happen separately below.
            stdfs::create_dir(dir.join("subagents")).unwrap();
        }

        let mut result = discover(tmp.path()).unwrap();
        result.sort_by(|a, b| a.project_dir.cmp(&b.project_dir));
        assert_eq!(result.len(), 2);
        for project in &result {
            assert_eq!(project.sessions.len(), 2);
            for session in &project.sessions {
                assert_eq!(session.jsonl.extension().unwrap(), "jsonl");
                // No subagents/ alongside any session_stem, so each
                // session's subagent vec is empty.
                assert!(session.subagents.is_empty());
            }
        }
    }

    #[test]
    fn discover_handles_empty_projects_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = discover(tmp.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn discover_handles_project_dir_with_no_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("-Users-only-index");
        stdfs::create_dir(&project).unwrap();
        stdfs::File::create(project.join("sessions-index.json")).unwrap();

        let result = discover(tmp.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].sessions.is_empty());
    }

    #[test]
    fn discover_walks_session_subagent_directory() {
        // For each session jsonl, look at <project>/<stem>/subagents/
        // and enumerate agent-*.jsonl files plus their .meta.json
        // sidecars.
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("-Users-x-proj");
        stdfs::create_dir(&project).unwrap();
        let session_stem = "abcd1234";
        stdfs::File::create(project.join(format!("{session_stem}.jsonl"))).unwrap();

        let subagents = project.join(session_stem).join("subagents");
        stdfs::create_dir_all(&subagents).unwrap();
        stdfs::File::create(subagents.join("agent-1.jsonl")).unwrap();
        stdfs::write(
            subagents.join("agent-1.meta.json"),
            r#"{"agentType":"foo"}"#,
        )
        .unwrap();
        stdfs::File::create(subagents.join("agent-2.jsonl")).unwrap();
        // agent-2 deliberately has no .meta.json — covers the absent
        // sidecar case.

        let result = discover(tmp.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].sessions.len(), 1);
        let session = &result[0].sessions[0];
        assert_eq!(session.subagents.len(), 2);
        let agent_1 = session
            .subagents
            .iter()
            .find(|s| s.jsonl.ends_with("agent-1.jsonl"))
            .expect("agent-1 missing");
        assert!(agent_1.meta.is_some());
        let agent_2 = session
            .subagents
            .iter()
            .find(|s| s.jsonl.ends_with("agent-2.jsonl"))
            .expect("agent-2 missing");
        assert!(agent_2.meta.is_none());
    }

    #[test]
    fn discover_subagents_returns_empty_when_directory_missing() {
        // Common case: most sessions never spawn a subagent, so the
        // probe directory simply isn't there.
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("-Users-x-proj");
        stdfs::create_dir(&project).unwrap();
        stdfs::File::create(project.join("session1.jsonl")).unwrap();

        let result = discover(tmp.path()).unwrap();
        assert!(result[0].sessions[0].subagents.is_empty());
    }

    #[test]
    fn discover_subagents_does_not_recurse_into_nested_directories() {
        // Validates the flat-only assumption: a directory inside
        // <stem>/subagents/ is ignored, not traversed.
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("-Users-x-proj");
        stdfs::create_dir(&project).unwrap();
        let session_stem = "abcd1234";
        stdfs::File::create(project.join(format!("{session_stem}.jsonl"))).unwrap();
        let subagents = project.join(session_stem).join("subagents");
        stdfs::create_dir_all(&subagents).unwrap();
        stdfs::File::create(subagents.join("agent-1.jsonl")).unwrap();
        // Nested directory with another agent-shaped file — must NOT
        // be recursed into.
        let nested = subagents.join("nested-dir");
        stdfs::create_dir(&nested).unwrap();
        stdfs::File::create(nested.join("agent-buried.jsonl")).unwrap();

        let result = discover(tmp.path()).unwrap();
        let session = &result[0].sessions[0];
        assert_eq!(session.subagents.len(), 1);
        assert!(session.subagents[0].jsonl.ends_with("agent-1.jsonl"));
    }

    #[test]
    fn discover_subagents_ignores_non_agent_jsonls() {
        // Stray .jsonl files in subagents/ that don't follow the
        // `agent-*` convention are skipped.
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("-Users-x-proj");
        stdfs::create_dir(&project).unwrap();
        let session_stem = "abcd1234";
        stdfs::File::create(project.join(format!("{session_stem}.jsonl"))).unwrap();
        let subagents = project.join(session_stem).join("subagents");
        stdfs::create_dir_all(&subagents).unwrap();
        stdfs::File::create(subagents.join("agent-1.jsonl")).unwrap();
        stdfs::File::create(subagents.join("notes.jsonl")).unwrap();
        stdfs::File::create(subagents.join("agent-1.meta.json")).unwrap();

        let result = discover(tmp.path()).unwrap();
        let session = &result[0].sessions[0];
        assert_eq!(session.subagents.len(), 1);
        assert!(session.subagents[0].jsonl.ends_with("agent-1.jsonl"));
    }

    #[test]
    fn read_subagent_meta_extracts_agent_type_and_description() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("agent-1.meta.json");
        stdfs::write(
            &path,
            r#"{"agentType":"tw-code-reviewer","description":"Review code"}"#,
        )
        .unwrap();
        let meta = read_subagent_meta(&path).expect("should parse");
        assert_eq!(meta.agent_type, "tw-code-reviewer");
        assert_eq!(meta.description.as_deref(), Some("Review code"));
    }

    #[test]
    fn read_subagent_meta_description_absent_yields_none() {
        // Older sidecars may carry only `agentType`; description must
        // gracefully degrade to None.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("agent-1.meta.json");
        stdfs::write(&path, r#"{"agentType":"tw-code-reviewer"}"#).unwrap();
        let meta = read_subagent_meta(&path).expect("should parse");
        assert_eq!(meta.agent_type, "tw-code-reviewer");
        assert!(meta.description.is_none());
    }

    #[test]
    fn read_subagent_meta_returns_none_for_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist.meta.json");
        assert!(read_subagent_meta(&missing).is_none());
    }

    #[test]
    fn read_subagent_meta_returns_none_for_malformed_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.meta.json");
        stdfs::write(&path, "{not valid json").unwrap();
        assert!(read_subagent_meta(&path).is_none());
    }

    #[test]
    fn read_subagent_meta_returns_none_when_agent_type_missing() {
        // Required field absent → parse fails → None. Graceful
        // degradation, no panic.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("incomplete.meta.json");
        stdfs::write(&path, r#"{"description":"no agentType field"}"#).unwrap();
        assert!(read_subagent_meta(&path).is_none());
    }
}
