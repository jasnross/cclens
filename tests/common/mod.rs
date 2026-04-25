// Shared helpers for integration tests. `tests/common/mod.rs` is the
// canonical Cargo pattern for sharing code across `tests/*.rs` integration
// test binaries — declaring `mod common;` from a sibling file pulls this
// in as a module rather than treating it as a separate test crate.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used, dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use assert_cmd::Command;
use filetime::{FileTime, set_file_mtime};

pub fn projects_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/projects")
}

/// Project directory for the cost-specific integration tests.
/// Isolated from `projects/` so adding fixture sessions for unknown
/// models / synthetic turns doesn't perturb the listing test
/// assertions.
pub fn cost_projects_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cost-projects")
}

/// Project directory for the cross-file dedup tests. Isolated from the
/// other fixture trees so sorting, mtime overrides, and resumed-session
/// shapes are kept out of the main listing assertions.
pub fn dedup_projects_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dedup-projects")
}

pub fn pricing_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pricing")
        .join(name)
}

pub fn pricing_fixture_url(name: &str) -> String {
    format!("file://{}", pricing_fixture(name).display())
}

pub fn snapshot_projects_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/snapshot-projects")
}

pub fn snapshot_pricing_url() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/snapshot-pricing/pricing-catalog.json");
    format!("file://{}", path.display())
}

/// Build a `cclens` command with hermetic env vars: pricing URL points
/// at a local fixture, cache directory points at the caller's tempdir.
/// Tests must use this rather than `Command::cargo_bin("cclens")` so
/// they don't pick up the user's real `~/Library/Caches/cclens/` cache.
pub fn cclens_command(cache_dir: &Path, pricing_url: &str) -> Command {
    let mut cmd = Command::cargo_bin("cclens").expect("cclens bin");
    cmd.env("CCLENS_PRICING_URL", pricing_url)
        .env("CCLENS_CACHE_DIR", cache_dir);
    cmd
}

/// Recursively copy `src` into a fresh tempdir and apply the given
/// mtime overrides (keyed by file name). Used by the dedup integration
/// tests to pin file ordering deterministically — the production
/// `discover` walker sorts `.jsonl` paths by mtime ascending, so the
/// test must control mtimes rather than relying on filesystem defaults.
pub fn copy_fixture_to_tempdir_with_mtimes(
    src: &Path,
    mtime_overrides: &[(&str, SystemTime)],
) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("tempdir");
    copy_dir_recursive(src, tmp.path());
    for (name, when) in mtime_overrides {
        let target = find_file_named(tmp.path(), name)
            .unwrap_or_else(|| panic!("override target {name} not found in copied fixture"));
        set_file_mtime(&target, FileTime::from_system_time(*when))
            .unwrap_or_else(|e| panic!("failed to set mtime on {}: {e}", target.display()));
    }
    tmp
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dst");
    for entry in fs::read_dir(src).expect("read src") {
        let entry = entry.expect("entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

fn find_file_named(root: &Path, name: &str) -> Option<PathBuf> {
    // Surface I/O failures rather than silently returning None — a
    // permission error on a subdir would otherwise show up as a
    // misleading "override target {name} not found" panic.
    for entry in fs::read_dir(root).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file_named(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
    }
    None
}
