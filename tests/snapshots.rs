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

use common::{
    build_inputs_claude_home, cclens_command, cclens_inputs_command, inputs_projects_fixture_dir,
    pricing_fixture_url, snapshot_pricing_url, snapshot_projects_dir,
};
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

#[test]
fn show_snapshot_with_subagents() {
    // Pin the byte-for-byte rendering of `cclens show` against the
    // delta session in the snapshot fixture, which carries two
    // tw-code-reviewer subagent invocations with distinct descriptions.
    // Locks the interleave + subagent-row content prefix shape.
    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &snapshot_pricing_url())
        .env("TZ", "UTC")
        .args(["--projects-dir"])
        .arg(snapshot_projects_dir())
        .arg("show")
        .arg("eeee0005-0005-0005-0005-000000000005")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).expect("stdout utf8");
    assert_snapshot!(stdout);
}

#[test]
fn inputs_snapshot() {
    // Snapshot strategy: set HOME to a per-test tempdir, build the
    // synthetic ~/.claude/ tree inside it, and rely on
    // `render_inputs::pretty_path` to substitute that prefix with
    // `~/...`. The result is byte-stable across machines because
    // every rendered path becomes `~/claude-home/...` regardless
    // of the actual tempdir location.
    let cache = isolated_cache();
    let home = isolated_cache();
    let claude_home = build_inputs_claude_home(home.path());
    let pricing = pricing_fixture_url("litellm-mini.json");

    let stdout_bytes = cclens_inputs_command(cache.path(), &pricing, &claude_home)
        .env("HOME", home.path())
        .env("TZ", "UTC")
        .args(["--projects-dir"])
        .arg(inputs_projects_fixture_dir())
        .arg("inputs")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).expect("stdout utf8");
    assert_snapshot!(stdout);
}
