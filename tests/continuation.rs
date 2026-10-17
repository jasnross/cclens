// Integration tests for continued sessions. A continuation's files
// replay its original's lines and link its original's subagent
// transcripts; every view must count that content once, under the
// original. Covers:
//   - `list` ownership independent of which file is newer
//   - `show` omitting replayed exchanges and linked subagents, and
//     opening a mid-turn continuation with a `continuation` row
//   - `inputs` crediting a replayed skill load once
//   - `agents` counting a linked subagent transcript once
//
// The fixture tree is `tests/fixtures/continuation-projects/`:
//   - `c1000001` → `c1000002`: a prompt-first continuation whose
//     original dispatched `cont-agent`, linked into the continuation.
//   - `c2000001` → `c2000002`: a continuation that resumes mid tool
//     loop, so its first new lines are assistant turns.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod common;

use std::time::{Duration, SystemTime};

use common::{
    build_agents_claude_home, build_inputs_claude_home, cclens_claude_home_command, cclens_command,
    continuation_projects_fixture_dir, copy_fixture_to_tempdir_with_mtimes, pricing_fixture_url,
};

const PRICING_FIXTURE: &str = "litellm-mini.json";

const C1_ORIGINAL: &str = "c1000001-0000-0000-0000-000000000001";
const C1_CONTINUATION: &str = "c1000002-0000-0000-0000-000000000002";
const C2_ORIGINAL: &str = "c2000001-0000-0000-0000-000000000001";
const C2_CONTINUATION: &str = "c2000002-0000-0000-0000-000000000002";

fn isolated_tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

/// Run a plain-output subcommand against the continuation fixture.
fn run(args: &[&str]) -> String {
    let cache = isolated_tempdir();
    let output = cclens_command(cache.path(), &pricing_fixture_url(PRICING_FIXTURE))
        .args(["--projects-dir"])
        .arg(continuation_projects_fixture_dir())
        .args(args)
        .output()
        .expect("run cclens");
    assert!(
        output.status.success(),
        "expected success; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout).unwrap()
}

/// The list row naming `id`, as `(tokens cell, whole line)`.
fn list_row<'a>(stdout: &'a str, id: &str) -> (&'a str, &'a str) {
    let line = stdout
        .lines()
        .find(|l| l.contains(id))
        .unwrap_or_else(|| panic!("no row for {id} in:\n{stdout}"));
    // The title may contain spaces, so find the tokens cell by shape.
    let tokens = line
        .split_whitespace()
        .find(|cell| {
            cell.strip_suffix('k')
                .is_some_and(|n| n.parse::<f64>().is_ok())
        })
        .unwrap_or_else(|| panic!("no tokens cell in row: {line}"));
    (tokens, line)
}

/// The data rows of a `show` table: every non-empty line after the
/// header.
fn show_rows(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .skip_while(|l| !l.contains(" role "))
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .collect()
}

#[test]
fn list_assigns_replayed_content_to_the_original() {
    let earlier = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let later = earlier + Duration::from_secs(3_600);
    let original = format!("{C1_ORIGINAL}.jsonl");
    let continuation = format!("{C1_CONTINUATION}.jsonl");
    // Ownership must follow lineage, whichever file is newer.
    for (original_mtime, continuation_mtime) in [(later, earlier), (earlier, later)] {
        let tmp = copy_fixture_to_tempdir_with_mtimes(
            &continuation_projects_fixture_dir(),
            &[
                (original.as_str(), original_mtime),
                (continuation.as_str(), continuation_mtime),
            ],
        );
        let cache = isolated_tempdir();
        let output = cclens_command(cache.path(), &pricing_fixture_url(PRICING_FIXTURE))
            .args(["--projects-dir"])
            .arg(tmp.path())
            .args(["list", "--format", "plain"])
            .output()
            .expect("run cclens");
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();

        // msg_P1 100+40, msg_P2 50+10, subagent msg_S1 20+30.
        assert_eq!(list_row(&stdout, C1_ORIGINAL).0, "0.25k", "{stdout}");
        let (tokens, line) = list_row(&stdout, C1_CONTINUATION);
        assert_eq!(tokens, "0.09k", "{stdout}");
        assert!(line.contains("Now summarize"), "{line}");
        assert_eq!(list_row(&stdout, C2_ORIGINAL).0, "0.25k", "{stdout}");
        let (tokens, line) = list_row(&stdout, C2_CONTINUATION);
        assert_eq!(tokens, "0.98k", "{stdout}");
        assert!(line.contains("Run the tests"), "{line}");
    }
}

#[test]
fn show_continuation_omits_replayed_and_linked_content() {
    let continuation = run(&["show", C1_CONTINUATION]);
    assert!(continuation.contains("Now summarize"), "{continuation}");
    assert!(continuation.contains("Summary written"), "{continuation}");
    assert!(
        !continuation.contains("Investigate the cache"),
        "replayed exchanges belong to the original:\n{continuation}",
    );
    assert!(
        !continuation.contains("cont-agent"),
        "the linked subagent belongs to the original:\n{continuation}",
    );

    let original = run(&["show", C1_ORIGINAL]);
    assert!(
        original.contains("Looking into it +2 tool uses"),
        "{original}"
    );
    assert!(
        original.contains("(cont-agent · \"Probe cache\") Cache is warm"),
        "{original}",
    );
}

#[test]
fn show_mid_turn_continuation_opens_with_a_continuation_row() {
    let stdout = run(&["show", C2_CONTINUATION]);
    let rows = show_rows(&stdout);
    let first = rows.first().expect("has rows");
    assert!(first.contains(" continuation "), "{stdout}");
    assert!(first.contains("0.65k"), "{first}");
    assert!(first.contains("Resuming the edit +1 tool uses"), "{first}");
    let last = rows.last().expect("has rows");
    assert!(last.contains("0.98k"), "final cumulative: {last}");
}

#[test]
fn show_json_serializes_continuation_role() {
    let stdout = run(&["show", C2_CONTINUATION, "--format", "json"]);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["exchanges"][0]["rows"][0]["role"], "continuation");
}

#[test]
fn inputs_counts_a_replayed_skill_load_once() {
    let cache = isolated_tempdir();
    let home = isolated_tempdir();
    let claude_home = build_inputs_claude_home(home.path());
    let output = cclens_claude_home_command(
        cache.path(),
        &pricing_fixture_url(PRICING_FIXTURE),
        &claude_home,
    )
    .env("HOME", home.path())
    .args(["--projects-dir"])
    .arg(continuation_projects_fixture_dir())
    .args(["inputs", "--format", "json"])
    .output()
    .expect("run cclens");
    assert!(output.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    let row = parsed["rows"]
        .as_array()
        .expect("rows array")
        .iter()
        .find(|r| {
            r["file"]["path"]
                .as_str()
                .is_some_and(|p| p.ends_with("skills/test-skill/SKILL.md"))
        })
        .unwrap_or_else(|| panic!("no test-skill row in:\n{parsed:#}"));
    let loads = row["loads_1h"].as_u64().unwrap() + row["loads_5m"].as_u64().unwrap();
    assert_eq!(loads, 1, "{row:#}");
}

#[test]
fn agents_counts_a_linked_subagent_once() {
    let cache = isolated_tempdir();
    let claude_home_owner = isolated_tempdir();
    let claude_home = build_agents_claude_home(claude_home_owner.path());
    let output = cclens_claude_home_command(
        cache.path(),
        &pricing_fixture_url(PRICING_FIXTURE),
        &claude_home,
    )
    .args(["--projects-dir"])
    .arg(continuation_projects_fixture_dir())
    .args(["agents", "--format", "plain"])
    .output()
    .expect("run cclens");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // Columns after split_whitespace (no fixture cell contains a space):
    //   0: agent  1: model  2: effort  3: declared_effort  4: pinning
    //   5: dispatches  6: tokens  7: cost
    let row = stdout
        .lines()
        .find(|l| l.split_whitespace().next() == Some("cont-agent"))
        .unwrap_or_else(|| panic!("no cont-agent row in:\n{stdout}"));
    let cols: Vec<&str> = row.split_whitespace().collect();
    assert_eq!(
        cols[5], "1",
        "the linked copy is not a second dispatch: {row}"
    );
    assert_eq!(cols[6], "0.05k", "{row}");
}
