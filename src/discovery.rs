//! Two-level walk of `~/.claude/projects/<dir>/<session>.jsonl`.
//!
//! Public API:
//! - `discover(&Path) -> anyhow::Result<Vec<(PathBuf, Vec<PathBuf>)>>`
//!   — returns one tuple per project directory, jsonl files sorted by
//!   mtime ascending so the cross-file dedup pass is deterministic.
//!
//! Per-entry and per-subdir read errors are silently skipped; only a
//! failure to read the top-level `projects_dir` propagates.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// # Errors
///
/// Returns an error only if the top-level `projects_dir` itself cannot
/// be read (a user-facing config error). Per-entry and per-subdir read
/// failures are silently skipped so one unreadable project doesn't
/// hide the rest.
pub fn discover(projects_dir: &Path) -> anyhow::Result<Vec<(PathBuf, Vec<PathBuf>)>> {
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
        sort_paths_by_mtime_asc(&mut jsonl_paths);
        result.push((path, jsonl_paths));
    }
    Ok(result)
}

/// Sort `.jsonl` paths by `(mtime, path)` ascending. Files whose
/// metadata or `modified()` call fails sort to the start
/// (`UNIX_EPOCH`); they are then attempted by `parse_jsonl` like any
/// other file and skipped via the existing per-file
/// `let Ok(turns) = parse_jsonl(...) else { continue }` handler if
/// genuinely unreadable. The `path` secondary key makes the order
/// fully deterministic when two files share an mtime (possible on
/// low-resolution filesystems or after `cp -p`); without it, ties
/// would fall back to `read_dir` insertion order, which the OS does
/// not specify.
fn sort_paths_by_mtime_asc(paths: &mut [PathBuf]) {
    paths.sort_by_cached_key(|p| {
        let mtime = fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        (mtime, p.clone())
    });
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
            // Nested directory — should be skipped at this level.
            stdfs::create_dir(dir.join("subagents")).unwrap();
        }

        let mut result = discover(tmp.path()).unwrap();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(result.len(), 2);
        for (_, jsonl_paths) in &result {
            assert_eq!(jsonl_paths.len(), 2);
            for p in jsonl_paths {
                assert_eq!(p.extension().unwrap(), "jsonl");
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
        assert!(result[0].1.is_empty());
    }
}
