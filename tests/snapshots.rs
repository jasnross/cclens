// Byte-for-byte snapshot tests for `list` and `show`. These act as the
// regression-detection contract for the modular refactor: any phase that
// causes rendering to drift will fail one of these assertions.
//
// `TZ=UTC` is set on the spawned child so `format_local`'s
// `chrono::Local` resolves deterministically. This is a Unix-only
// behavior (chrono on Windows reads the system API rather than the env
// var); cclens is currently Unix-targeted, so this is safe.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use common::{cclens_command, snapshot_pricing_url, snapshot_projects_dir};
use insta::assert_snapshot;

fn isolated_cache() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

#[test]
fn list_snapshot() {
    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &snapshot_pricing_url())
        .env("TZ", "UTC")
        .args(["--projects-dir"])
        .arg(snapshot_projects_dir())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).expect("stdout utf8");
    assert_snapshot!(stdout);
}

#[test]
fn show_snapshot() {
    // Session A — the short slash-command session — exercises a normal
    // user→assistant exchange rendering.
    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &snapshot_pricing_url())
        .env("TZ", "UTC")
        .args(["--projects-dir"])
        .arg(snapshot_projects_dir())
        .arg("show")
        .arg("aaaa0001-0001-0001-0001-000000000001")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).expect("stdout utf8");
    assert_snapshot!(stdout);
}
