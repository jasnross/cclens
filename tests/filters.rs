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
        user_row.contains("150"),
        "ex2 user row cumulative should be 150 (folds in hidden ex1); got: {user_row}",
    );
    assert!(
        user_row.contains("$0.0046"),
        "ex2 user row cum_cost should be $0.0046 (folds in hidden ex1); got: {user_row}",
    );

    // Cumulative on asst row = 150 + 400 (ex2 asst output) = 550.
    // cum_cost = $0.0347 (ex1 + ex2 user-row + ex2 asst-row).
    assert!(
        asst_row.contains("550"),
        "ex2 asst row cumulative should be 550; got: {asst_row}",
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

// ----------------------------- filter scope ------------------------------

#[test]
fn pricing_subcommand_rejects_filter_flags() {
    // Confirms `FilterArgs` is NOT flattened into `Command::Pricing`:
    // passing `--min-tokens` or `--min-cost` to `pricing refresh`
    // must be a clap parse error (non-zero exit, "unexpected argument").
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
