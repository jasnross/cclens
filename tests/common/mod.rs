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

/// Build a `cclens` command for the `inputs` subcommand with a
/// hermetic `CCLENS_CLAUDE_HOME` override. Adds the third env var on
/// top of `cclens_command`'s pricing/cache isolation so tests don't
/// leak into the user's real `~/.claude/` directory.
pub fn cclens_inputs_command(cache_dir: &Path, pricing_url: &str, claude_home: &Path) -> Command {
    let mut cmd = cclens_command(cache_dir, pricing_url);
    cmd.env("CCLENS_CLAUDE_HOME", claude_home);
    cmd
}

pub fn inputs_projects_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inputs-projects")
}

/// Build a synthetic `~/.claude/` tree under `cache_dir` and return the
/// path. Contents:
/// - `CLAUDE.md` (global)
/// - `rules/test-rule.md`
/// - `skills/test-skill/SKILL.md`
/// - `agents/test-agent.md`
/// - `plugins/installed_plugins.json` referencing one plugin
/// - `plugins/cache/test-marketplace/test-plugin/unknown/skills/test/SKILL.md`
///   (uses the literal `unknown` version segment to exercise that case)
pub fn build_inputs_claude_home(cache_dir: &Path) -> PathBuf {
    let claude_home = cache_dir.join("claude-home");
    fs::create_dir_all(&claude_home).expect("create claude-home");

    fs::write(
        claude_home.join("CLAUDE.md"),
        "# Global rules\n\nThis is the user's global CLAUDE.md \
         file with some content for the inputs integration tests \
         to attribute against. The exact token count comes from \
         the tokenizer.\n",
    )
    .expect("write CLAUDE.md");

    fs::create_dir_all(claude_home.join("rules")).expect("create rules/");
    fs::write(
        claude_home.join("rules/test-rule.md"),
        "# Test Rule\n\nA short global rule body.\n",
    )
    .expect("write rule");

    fs::create_dir_all(claude_home.join("skills/test-skill")).expect("create skills/test-skill");
    fs::write(
        claude_home.join("skills/test-skill/SKILL.md"),
        "# Test Skill\n\nUse this when running the integration tests.\n",
    )
    .expect("write skill");

    fs::create_dir_all(claude_home.join("agents")).expect("create agents/");
    fs::write(
        claude_home.join("agents/test-agent.md"),
        "# Test Agent\n\nDispatched when the integration tests need an agent file.\n",
    )
    .expect("write agent");

    let plugin_path = claude_home.join("plugins/cache/test-marketplace/test-plugin/unknown");
    fs::create_dir_all(plugin_path.join("skills/test")).expect("create plugin skill dir");
    fs::write(
        plugin_path.join("skills/test/SKILL.md"),
        "# Plugin Skill\n\nFrom a plugin in the cache.\n",
    )
    .expect("write plugin skill");

    let plugins_dir = claude_home.join("plugins");
    let json = format!(
        r#"{{"plugins":{{"test-plugin@test-marketplace":[{{"installPath":{path:?},"version":"unknown"}}]}}}}"#,
        path = plugin_path.to_string_lossy(),
    );
    fs::write(plugins_dir.join("installed_plugins.json"), json)
        .expect("write installed_plugins.json");

    claude_home
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
