// Shared helpers for integration tests. `tests/common/mod.rs` is the
// canonical Cargo pattern for sharing code across `tests/*.rs` integration
// test binaries — declaring `mod common;` from a sibling file pulls this
// in as a module rather than treating it as a separate test crate.

#![allow(clippy::expect_used, clippy::unwrap_used, dead_code)]

use std::path::{Path, PathBuf};

use assert_cmd::Command;

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

pub fn pricing_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pricing")
        .join(name)
}

pub fn pricing_fixture_url(name: &str) -> String {
    format!("file://{}", pricing_fixture(name).display())
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
