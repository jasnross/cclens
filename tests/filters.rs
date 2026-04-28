// Integration tests for `--min-tokens` / `--min-cost` filter flags on
// `list` and `show`. These tests are hermetic — they pin
// `CCLENS_PRICING_URL` to a local fixture and `CCLENS_CACHE_DIR` to a
// fresh tempdir so they never touch the user's real cache.
//
// The token / cost expectations below are derived from the
// `litellm-mini.json` fixture's claude-opus-4-7 rates and are
// hand-computed (the same way `tests/listing.rs` derives its
// expectations). Any change to the pricing fixture's claude-opus-4-7
// rates will require recomputing the thresholds here.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use common::{
    cclens_command, cost_projects_fixture_dir, pricing_fixture_url, projects_fixture_dir,
};
use predicates::prelude::PredicateBooleanExt;

fn isolated_cache() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

const SHOW_SESSION_ID: &str = "eeee5555-5555-5555-5555-555555555555";
const ALPHA_UUID: &str = "aaaa1111-1111-1111-1111-111111111111";
const BETA_PROSE_UUID: &str = "bbbb2222-2222-2222-2222-222222222222";
const BETA_LATE_UUID: &str = "dddd4444-4444-4444-4444-444444444444";
const UNKNOWN_MODEL_UUID: &str = "ffff6666-6666-6666-6666-666666666666";
const RUNNING_TOTALS_UUID: &str = "bbbb8888-8888-8888-8888-888888888888";

// ------------------------------ list filters ------------------------------

#[test]
fn list_no_filters_matches_baseline() {
    // Sanity: with no filter args, all three priced sessions in the
    // baseline fixture are present (matches the row set
    // `tests/listing.rs::list_renders_sessions_oldest_first…` covers).
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(stdout.contains(ALPHA_UUID), "alpha missing:\n{stdout}");
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose missing:\n{stdout}",
    );
    assert!(
        stdout.contains(BETA_LATE_UUID),
        "beta-late missing:\n{stdout}",
    );
}

#[test]
fn list_min_tokens_only() {
    // alpha=650, beta-prose=100, beta-after-malformed=10.
    // Threshold 100 keeps alpha (650) and beta-prose (100, boundary),
    // drops beta-after-malformed (10 < 100).
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--min-tokens", "100"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(ALPHA_UUID),
        "alpha should remain:\n{stdout}"
    );
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose at boundary should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-after-malformed should be dropped:\n{stdout}",
    );
}

#[test]
fn list_min_cost_only() {
    // alpha=$0.0379, beta-prose=$0.0069, beta-after-malformed=$0.0004.
    // Threshold $0.01 keeps only alpha; the other two fail on cost.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--min-cost", "0.01"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(ALPHA_UUID),
        "alpha should remain:\n{stdout}"
    );
    assert!(
        !stdout.contains(BETA_PROSE_UUID),
        "beta-prose ($0.0069 < $0.01) should be dropped:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-after-malformed ($0.0004 < $0.01) should be dropped:\n{stdout}",
    );
}

#[test]
fn list_both_filters_logical_and() {
    // Two invocations exercise distinct AND-rejection axes.
    //
    // 1) --min-tokens 200 --min-cost 0.001
    //    alpha (650, $0.0379) clears both; beta-prose (100 < 200)
    //    fails tokens; beta-after-malformed (10 < 200) fails tokens.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--min-tokens", "200", "--min-cost", "0.001"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(ALPHA_UUID),
        "alpha should remain:\n{stdout}"
    );
    assert!(
        !stdout.contains(BETA_PROSE_UUID),
        "beta-prose (100 < 200) should be dropped:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-after-malformed (10 < 200) should be dropped:\n{stdout}",
    );

    // 2) --min-tokens 50 --min-cost 0.01
    //    alpha (650, $0.0379) clears both; beta-prose ($0.0069 <
    //    $0.01) fails on cost; beta-after-malformed (10 < 50) fails
    //    on tokens. Different rows fail different axes.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--min-tokens", "50", "--min-cost", "0.01"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(ALPHA_UUID),
        "alpha should remain:\n{stdout}"
    );
    assert!(
        !stdout.contains(BETA_PROSE_UUID),
        "beta-prose ($0.0069 < $0.01) should be dropped on cost:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-after-malformed (10 < 50) should be dropped on tokens:\n{stdout}",
    );
}

#[test]
fn list_min_cost_excludes_unknown_model_session() {
    // The unknown-model session in cost-projects has cost=`—` (None).
    // Any active --min-cost (even one that any priced row clears) must
    // exclude it.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(cost_projects_fixture_dir())
        .args(["list", "--min-cost", "0.0001"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        !stdout.contains(UNKNOWN_MODEL_UUID),
        "unknown-model session should be excluded by any --min-cost:\n{stdout}",
    );
}

#[test]
fn list_empty_result_emits_stderr_hint() {
    // A token threshold larger than every priced row drops them all.
    // stdout still has the header (`render_table` always prints it);
    // stderr gains the one-line hint; exit is 0.
    let cache = isolated_cache();
    let assertion = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--min-tokens", "999999999"])
        .assert()
        .success()
        .stderr(predicates::str::contains(
            "note: no rows matched --min-tokens 999999999",
        ));
    let output = assertion.get_output();
    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    assert!(
        stdout.contains("datetime"),
        "header should still print:\n{stdout}"
    );
    let data_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.contains("datetime"))
        .collect();
    assert!(
        data_lines.is_empty(),
        "no data rows should remain; got:\n{stdout}",
    );
}

#[test]
fn list_no_filters_no_stderr_hint_on_empty_projects_dir() {
    // Pre-existing contract: empty `projects_dir` with no filter flags
    // produces no stderr hint. Asserting empty stderr explicitly via
    // `predicates::str::is_empty()` rather than relying on `.success()`
    // alone.
    let cache = isolated_cache();
    let tmp = tempfile::tempdir().unwrap();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(tmp.path())
        .arg("list")
        .assert()
        .success()
        .stderr(predicates::str::is_empty());
}

// ------------------------------ show filters ------------------------------

#[test]
fn show_no_filters_matches_baseline() {
    // Sanity: all three exchanges are present without filters.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg(SHOW_SESSION_ID)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(stdout.contains("/test-cmd demo"), "ex1 missing:\n{stdout}");
    assert!(
        stdout.contains("follow-up question"),
        "ex2 missing:\n{stdout}",
    );
    assert!(
        stdout.contains("third question with no response"),
        "ex3 (orphan) missing:\n{stdout}",
    );
}

#[test]
fn show_min_tokens_excludes_orphan() {
    // The orphan exchange has tokens=0; --min-tokens 1 drops it.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg(SHOW_SESSION_ID)
        .args(["--min-tokens", "1"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains("/test-cmd demo"),
        "ex1 should remain:\n{stdout}"
    );
    assert!(
        stdout.contains("follow-up question"),
        "ex2 should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains("third question with no response"),
        "orphan should be dropped:\n{stdout}",
    );
}

#[test]
fn show_min_tokens_excludes_below_threshold() {
    // Exchange 1: 595, exchange 2: 180, orphan: 0.
    // --min-tokens 500 keeps only exchange 1.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg(SHOW_SESSION_ID)
        .args(["--min-tokens", "500"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains("/test-cmd demo"),
        "ex1 should remain:\n{stdout}"
    );
    assert!(
        !stdout.contains("follow-up question"),
        "ex2 (180 < 500) should be dropped:\n{stdout}",
    );
    assert!(
        !stdout.contains("third question with no response"),
        "orphan should be dropped:\n{stdout}",
    );
}

#[test]
fn show_min_cost_excludes_orphan_and_below_threshold() {
    // Exchange 1: $0.0266, exchange 2: $0.0067, orphan: None.
    // --min-cost 0.01 keeps only exchange 1.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg(SHOW_SESSION_ID)
        .args(["--min-cost", "0.01"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains("/test-cmd demo"),
        "ex1 should remain:\n{stdout}"
    );
    assert!(
        !stdout.contains("follow-up question"),
        "ex2 ($0.0067 < $0.01) should be dropped:\n{stdout}",
    );
    assert!(
        !stdout.contains("third question with no response"),
        "orphan (cost=None) should be dropped:\n{stdout}",
    );
}

#[test]
fn show_min_cost_excludes_unknown_model_exchange() {
    // The unknown-model fixture has one assistant turn priced
    // `claude-fake-9-9` — its exchange-level cost is None, so any
    // active --min-cost excludes it. The session has only one
    // exchange, so the result is empty (and gets the stderr hint).
    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(cost_projects_fixture_dir())
        .arg("show")
        .arg(UNKNOWN_MODEL_UUID)
        .args(["--min-cost", "0.0001"])
        .assert()
        .success()
        .stderr(predicates::str::contains(
            "note: no rows matched --min-cost",
        ));
}

#[test]
fn show_both_filters_logical_and() {
    // Exchange 1 (595, $0.0266) clears both; exchange 2 (180,
    // $0.0067) fails tokens (180 < 200); orphan fails both.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg(SHOW_SESSION_ID)
        .args(["--min-tokens", "200", "--min-cost", "0.005"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains("/test-cmd demo"),
        "ex1 should remain:\n{stdout}"
    );
    assert!(
        !stdout.contains("follow-up question"),
        "ex2 (180 < 200) should be dropped:\n{stdout}",
    );
}

#[test]
fn show_running_totals_span_hidden_exchanges() {
    // Critical correctness test: `cumulative` and `cum_cost` must
    // continue to fold over **every** exchange so that filtering only
    // hides rows — running totals on visible rows still match the
    // full session.
    //
    // Per-exchange totals on the bbbb8888 fixture (claude-opus-4-7
    // rates from litellm-mini.json):
    //   ex1 `first question`: tokens=50,  cost=$0.0032
    //   ex2 `big question`:   tokens=500, cost=$0.0315
    //   ex3 `third question`: tokens=50,  cost=$0.0032
    //
    // With --min-tokens 200, only exchange 2 survives. Its rendered
    // rows must show:
    //   ex2 user row: cumulative=150, cum_cost=$0.0047
    //   ex2 asst row: cumulative=550, cum_cost=$0.0347
    // A regression that pre-pruned exchanges before rendering would
    // produce cumulative=100/500 and cum_cost=$0.0015/$0.0315 instead.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(cost_projects_fixture_dir())
        .arg("show")
        .arg(RUNNING_TOTALS_UUID)
        .args(["--min-tokens", "200"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();

    // Hidden exchanges must be absent.
    assert!(
        !stdout.contains("first question"),
        "ex1 (50 < 200) should be hidden:\n{stdout}",
    );
    assert!(
        !stdout.contains("third question"),
        "ex3 (50 < 200) should be hidden:\n{stdout}",
    );

    // Visible row markers (from the assistant cluster previews).
    let user_row = stdout
        .lines()
        .find(|l| l.contains("big question"))
        .unwrap_or_else(|| panic!("ex2 user row missing:\n{stdout}"));
    let asst_row = stdout
        .lines()
        .find(|l| l.contains("big answer"))
        .unwrap_or_else(|| panic!("ex2 assistant row missing:\n{stdout}"));

    // Cumulative on user row = 50 (ex1 billable) + 100 (ex2 user-row
    // tokens) = 150. cum_cost ≈ $0.0046 — accumulated f64s render
    // slightly low due to representation + banker's rounding via
    // `{:.4}` (matches the same effect documented in
    // `tests/listing.rs::show_renders_per_exchange_table_…`).
    assert!(
        user_row.contains("0.15k"),
        "ex2 user row cumulative should be 0.15k (folds in hidden ex1); got: {user_row}",
    );
    assert!(
        user_row.contains("$0.0046"),
        "ex2 user row cum_cost should be $0.0046 (folds in hidden ex1); got: {user_row}",
    );

    // Cumulative on asst row = 150 + 400 (ex2 asst output) = 550.
    // cum_cost = $0.0347 (ex1 + ex2 user-row + ex2 asst-row).
    assert!(
        asst_row.contains("0.55k"),
        "ex2 asst row cumulative should be 0.55k; got: {asst_row}",
    );
    assert!(
        asst_row.contains("$0.0347"),
        "ex2 asst row cum_cost should be $0.0347; got: {asst_row}",
    );
}

#[test]
fn show_empty_result_emits_stderr_hint() {
    // Threshold above every exchange's tokens drops them all. stdout
    // still prints the header; stderr gets the hint; exit is 0.
    let cache = isolated_cache();
    let assertion = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg(SHOW_SESSION_ID)
        .args(["--min-tokens", "999999999"])
        .assert()
        .success()
        .stderr(predicates::str::contains(
            "note: no rows matched --min-tokens 999999999",
        ));
    let output = assertion.get_output();
    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    assert!(
        stdout.contains("datetime"),
        "header should still print:\n{stdout}"
    );
    let data_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.contains("datetime"))
        .collect();
    assert!(
        data_lines.is_empty(),
        "no data rows should remain; got:\n{stdout}",
    );
}

// ------------------------- list scope filters --------------------------

// Baseline fixture timestamps (verified against
// tests/fixtures/projects/...):
//   alpha       (ALPHA_UUID)      2026-04-01T10:00:00Z
//   beta-prose  (BETA_PROSE_UUID) 2026-04-15T14:33:00Z
//   beta-late   (BETA_LATE_UUID)  2026-04-20T09:00:00Z
// The fixture has additional sessions (show-fixture, etc.) whose
// presence is ignored by these tests' positive/negative-only
// assertions — adding new fixture sessions never breaks them.

#[test]
fn list_project_only_keeps_matching_short_name() {
    // `--project beta` keeps both beta sessions (project_short_name
    // == "beta") and drops alpha.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--project", "beta"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose should remain:\n{stdout}",
    );
    assert!(
        stdout.contains(BETA_LATE_UUID),
        "beta-late should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(ALPHA_UUID),
        "alpha (project=alpha) should be dropped:\n{stdout}",
    );
}

#[test]
fn list_project_no_match_emits_empty_hint() {
    // `--project zzz` matches nothing; stdout still shows the header,
    // stderr gets the hint, exit is 0.
    let cache = isolated_cache();
    let assertion = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--project", "zzz"])
        .assert()
        .success()
        .stderr(predicates::str::contains(
            "note: no rows matched --project zzz",
        ));
    let output = assertion.get_output();
    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    assert!(
        stdout.contains("datetime"),
        "header should still print:\n{stdout}",
    );
    assert!(
        !stdout.contains(ALPHA_UUID),
        "no priced rows should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_PROSE_UUID),
        "no priced rows should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "no priced rows should remain:\n{stdout}",
    );
}

#[test]
fn list_since_inclusive_at_boundary() {
    // `--since 2026-04-15T14:33:00Z` (beta-prose's exact start) keeps
    // beta-prose (boundary) and beta-late (after); drops alpha
    // (before).
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--since", "2026-04-15T14:33:00Z"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose at boundary should remain:\n{stdout}",
    );
    assert!(
        stdout.contains(BETA_LATE_UUID),
        "beta-late (after boundary) should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(ALPHA_UUID),
        "alpha (before boundary) should be dropped:\n{stdout}",
    );
}

#[test]
fn list_since_excludes_one_second_before() {
    // `--since 2026-04-15T14:33:01Z` (one second past beta-prose)
    // drops beta-prose; keeps beta-late.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--since", "2026-04-15T14:33:01Z"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        !stdout.contains(BETA_PROSE_UUID),
        "beta-prose (one second before --since) should be dropped:\n{stdout}",
    );
    assert!(
        stdout.contains(BETA_LATE_UUID),
        "beta-late should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(ALPHA_UUID),
        "alpha should be dropped:\n{stdout}",
    );
}

#[test]
fn list_until_inclusive_at_boundary() {
    // `--until 2026-04-15T14:33:00Z` (beta-prose's exact start) keeps
    // alpha and beta-prose (boundary); drops beta-late.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--until", "2026-04-15T14:33:00Z"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(ALPHA_UUID),
        "alpha (before boundary) should remain:\n{stdout}",
    );
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose at boundary should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-late (after boundary) should be dropped:\n{stdout}",
    );
}

#[test]
fn list_since_until_brackets_window() {
    // `--since 2026-04-10T00:00:00Z --until 2026-04-18T00:00:00Z`:
    // alpha (2026-04-01) before window; beta-prose (2026-04-15) in
    // window; beta-late (2026-04-20) after window.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args([
            "list",
            "--since",
            "2026-04-10T00:00:00Z",
            "--until",
            "2026-04-18T00:00:00Z",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose (in window) should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(ALPHA_UUID),
        "alpha (before window) should be dropped:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-late (after window) should be dropped:\n{stdout}",
    );
}

#[test]
fn list_project_combined_with_min_tokens_logical_and() {
    // `--project beta --min-tokens 100` keeps only beta-prose
    // (boundary at 100 tokens). beta-late has 10 tokens so fails the
    // threshold; alpha is dropped on project. Exercises both axes
    // composing as logical AND.
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--project", "beta", "--min-tokens", "100"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose (project + tokens both clear) should remain:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-late (10 < 100) should be dropped:\n{stdout}",
    );
    assert!(
        !stdout.contains(ALPHA_UUID),
        "alpha (project mismatch) should be dropped:\n{stdout}",
    );
}

#[test]
fn list_invalid_date_format_clap_error() {
    // Bare `YYYY-MM-DD` (no time, no offset) is rejected by chrono's
    // RFC 3339 parser at clap parse time. The flag name appears in
    // stderr (anchoring on the offending flag is more stable across
    // clap/chrono error-template versions).
    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--since", "2026-04-15"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--since"));
}

#[test]
fn list_combined_empty_hint_describes_all_active_flags() {
    // `--project alpha --min-cost 99999`: scope keeps alpha (the
    // sole alpha session); threshold rejects it (alpha's cost is
    // ~$0.0379 ≪ $99999). Both filters causally contribute to the
    // empty result, so both flag descriptions must appear in the
    // hint. Using a large min-cost rather than `--project zzz`
    // ensures the threshold check actually fires (a non-existent
    // project would short-circuit before threshold matters).
    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args(["list", "--project", "alpha", "--min-cost", "99999"])
        .assert()
        .success()
        .stderr(predicates::str::contains("--project alpha"))
        .stderr(predicates::str::contains("--min-cost 99999"));
}

#[test]
fn list_full_stack_combined_filters() {
    // End-to-end: every layer (clap parsing →
    // SessionFilterArgs::session_filter() → SessionFilter::accepts →
    // ThresholdsFilter::matches → render).
    //
    // beta-prose:  beta + 2026-04-15T14:33:00Z + 100 tokens — clears
    //              project, since-boundary, until, min-tokens 50 → kept
    // beta-late:   beta + 2026-04-20T09:00:00Z + 10 tokens — clears
    //              project, since, until-boundary; fails min-tokens 50
    // alpha:       alpha (project mismatch)
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .args([
            "list",
            "--project",
            "beta",
            "--since",
            "2026-04-15T14:33:00Z",
            "--until",
            "2026-04-20T09:00:00Z",
            "--min-tokens",
            "50",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    assert!(
        stdout.contains(BETA_PROSE_UUID),
        "beta-prose clears all four filters:\n{stdout}",
    );
    assert!(
        !stdout.contains(BETA_LATE_UUID),
        "beta-late fails on tokens (10 < 50):\n{stdout}",
    );
    assert!(
        !stdout.contains(ALPHA_UUID),
        "alpha fails on project:\n{stdout}",
    );
}

#[test]
fn list_inputs_project_filter_parity() {
    // The same `--project alpha` filter against the same fixture must
    // keep and drop the same project sessions on `list` and `inputs`.
    // Divergence here would indicate a derivation drift between
    // `Session.project_short_name` (used by `list`) and
    // `SessionMeta.project_short_name` (used by `inputs`). We can't
    // compare full row sets (the two views render different things),
    // but we can assert that the `list` row count is ≥ 1 (alpha is
    // present) and that `inputs` produces a non-empty rendered table
    // for the same project — i.e. neither view incorrectly filters
    // alpha out.
    let cache_list = isolated_cache();
    let list_stdout = String::from_utf8(
        cclens_command(cache_list.path(), &pricing_fixture_url("litellm-mini.json"))
            .args(["--projects-dir"])
            .arg(projects_fixture_dir())
            .args(["list", "--project", "alpha"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone(),
    )
    .unwrap();
    assert!(
        list_stdout.contains(ALPHA_UUID),
        "list --project alpha should keep alpha:\n{list_stdout}",
    );

    let cache_inputs = isolated_cache();
    cclens_command(
        cache_inputs.path(),
        &pricing_fixture_url("litellm-mini.json"),
    )
    .args(["--projects-dir"])
    .arg(projects_fixture_dir())
    .args(["inputs", "--project", "alpha"])
    .assert()
    .success()
    // Negative assertion: no empty-result hint means inputs found
    // at least one session in scope.
    .stderr(predicates::str::contains("no rows matched").not());
}

// ----------------------------- filter scope ------------------------------

#[test]
fn pricing_subcommand_rejects_filter_flags() {
    // Confirms `ThresholdsFilterArgs` is NOT flattened into
    // `Command::Pricing`: passing `--min-tokens` or `--min-cost` to
    // `pricing refresh` must be a clap parse error (non-zero exit,
    // "unexpected argument").
    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["pricing", "refresh", "--min-tokens", "1"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--min-tokens"));

    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["pricing", "refresh", "--min-cost", "0.50"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--min-cost"));
}

#[test]
fn pricing_subcommand_rejects_scope_flags() {
    // Sibling of `pricing_subcommand_rejects_filter_flags`: confirms
    // `SessionFilterArgs` is NOT flattened into `Command::Pricing`.
    // Each scope flag must produce a clap parse error.
    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["pricing", "refresh", "--project", "foo"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--project"));

    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["pricing", "refresh", "--since", "2026-01-01T00:00:00Z"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--since"));

    let cache = isolated_cache();
    cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["pricing", "refresh", "--until", "2026-01-01T00:00:00Z"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--until"));
}
