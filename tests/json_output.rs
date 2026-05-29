// Integration tests for `--format json` output across all four data views.
//
// Each test validates that the JSON output is well-formed, deserializable,
// and contains the expected top-level field names. Tests are hermetic —
// pricing URL and cache dir are overridden per `common::cclens_command`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use common::{
    build_inputs_claude_home, cclens_command, cclens_inputs_command, inputs_projects_fixture_dir,
    pricing_fixture_url, projects_fixture_dir,
};

fn isolated_cache() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

#[test]
fn list_format_json_produces_valid_json_array() {
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("invalid JSON: {e}\nstdout was:\n{stdout}"));
    assert!(!parsed.is_empty(), "expected at least one session summary");
    let first = &parsed[0];
    for field in [
        "id",
        "project_short_name",
        "started_at",
        "total_billable",
        "cost_breakdown",
        "total_cost",
    ] {
        assert!(
            first.get(field).is_some(),
            "missing field {field} in session summary: {first}",
        );
    }
    assert!(
        first.get("turns").is_none(),
        "session summary must not contain turns: {first}",
    );
}

#[test]
fn list_format_json_empty_fixture_produces_empty_array() {
    let cache = isolated_cache();
    let empty = tempfile::tempdir().expect("empty tempdir");
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(empty.path())
        .args(["list", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&stdout).expect("should parse as JSON array");
    assert!(parsed.is_empty(), "expected empty array, got: {stdout}");
}

#[test]
fn show_format_json_produces_valid_json_object() {
    let cache = isolated_cache();
    let session_id = "eeee5555-5555-5555-5555-555555555555";
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["show", "--format", "json", session_id])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("should parse as JSON object");
    assert_eq!(
        parsed.get("session_id").and_then(|v| v.as_str()),
        Some(session_id),
    );
    assert!(
        parsed.get("exchanges").is_some(),
        "missing 'exchanges' field in show output",
    );
}

#[test]
fn inputs_format_json_produces_valid_json_object() {
    let cache = isolated_cache();
    let claude_home = build_inputs_claude_home(cache.path());
    let out = cclens_inputs_command(
        cache.path(),
        &pricing_fixture_url("litellm-mini.json"),
        &claude_home,
    )
    .args(["--projects-dir"])
    .arg(inputs_projects_fixture_dir())
    .args(["inputs", "--format", "json"])
    .assert()
    .success()
    .get_output()
    .stdout
    .clone();
    let stdout = String::from_utf8(out).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("should parse as JSON object");
    assert!(
        parsed.get("rows").is_some(),
        "missing 'rows' field in inputs output",
    );
    assert!(
        parsed.get("coverage").is_some(),
        "missing 'coverage' field in inputs output",
    );
}

#[test]
fn pricing_list_format_json_produces_valid_json_array() {
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["pricing", "list", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&stdout).expect("should parse as JSON array");
    assert!(!parsed.is_empty(), "expected at least one pricing entry");
    let first = &parsed[0];
    for field in ["model", "input", "output"] {
        assert!(
            first.get(field).is_some(),
            "missing field {field} in pricing entry: {first}",
        );
    }
}

#[test]
fn format_json_flag_works_before_subcommand() {
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--format", "json", "--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    let _: Vec<serde_json::Value> =
        serde_json::from_str(&stdout).expect("should parse when --format is before subcommand");
}
