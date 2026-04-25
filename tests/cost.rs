// Cost-specific integration tests. Covers behaviors that are not
// already verified by `tests/listing.rs`:
//   - unknown-model rendering and cum_cost latching in `show`
//   - zero-usage (synthetic) turns rendering as $0.0000 (not `—`)
//   - graceful degradation when the pricing fetch fails on `list`
//
// Fixtures for these tests live under `tests/fixtures/cost-projects/`
// so they don't perturb the listing-test assertions.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use common::{cclens_command, cost_projects_fixture_dir, pricing_fixture_url};

const COST_FIXTURE_URL_NAME: &str = "litellm-mini.json";

fn isolated_cache() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

#[test]
fn list_renders_dash_for_unknown_model_session() {
    // The cost-projects fixture tree contains a session that uses
    // model `claude-fake-9-9` (not in the pricing fixture). Its row
    // must still render — but with `—` in the cost cell.
    let cache = isolated_cache();
    let stdout = cclens_command(cache.path(), &pricing_fixture_url(COST_FIXTURE_URL_NAME))
        .args(["--projects-dir"])
        .arg(cost_projects_fixture_dir())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let out = String::from_utf8(stdout).unwrap();
    let unknown_session_uuid = "ffff6666-6666-6666-6666-666666666666";
    let row = out
        .lines()
        .find(|l| l.contains(unknown_session_uuid))
        .unwrap_or_else(|| panic!("unknown-model session row missing:\n{out}"));

    // Strip the trailing UUID; what's left should end with the
    // cost cell. Unknown model → cost cell is `—` (not `$X.XXXX`).
    let pre_uuid = row
        .trim_end()
        .strip_suffix(unknown_session_uuid)
        .expect("uuid suffix")
        .trim_end();
    assert!(
        pre_uuid.ends_with('—'),
        "unknown-model session must show `—` in cost cell; got: {row}",
    );
    assert!(
        !pre_uuid.contains('$'),
        "unknown-model session must not show a `$` cost; got: {row}",
    );
}

#[test]
fn show_renders_dash_for_unknown_model_row_and_latches_cum_cost() {
    // The unknown-model session has one assistant turn priced
    // `claude-fake-9-9`. Both the user and assistant rows must show
    // `—` in `cost`, AND `cum_cost` must latch to `—` from the
    // first unknown row onwards.
    let cache = isolated_cache();
    let stdout = cclens_command(cache.path(), &pricing_fixture_url(COST_FIXTURE_URL_NAME))
        .args(["--projects-dir"])
        .arg(cost_projects_fixture_dir())
        .arg("show")
        .arg("ffff6666-6666-6666-6666-666666666666")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let out = String::from_utf8(stdout).unwrap();
    let lines: Vec<&str> = out.lines().collect();

    // Both the user row ("hello") and the assistant row ("reply
    // from a model the catalog doesn't know") must show `—`. The
    // tokens column has its own `—`-or-number value, so we count
    // total `—` glyphs to verify each row contributes at least one
    // dash beyond what the orphan-tokens contract would.
    let user_row = lines
        .iter()
        .find(|l| l.contains("hello"))
        .expect("user row missing");
    let assistant_row = lines
        .iter()
        .find(|l| l.contains("reply from a model the catalog doesn't know"))
        .expect("assistant row missing");

    // user row: cost cell `—`, cum_cost cell `—`
    // (tokens cell shows the cluster's input+cc total, so it's a
    //  number, not `—`)
    assert!(
        user_row.matches('—').count() >= 2,
        "user row should show `—` in both cost and cum_cost; got: {user_row}",
    );

    // assistant row: cost `—`, cum_cost `—`
    // (tokens cell shows output, a number)
    assert!(
        assistant_row.matches('—').count() >= 2,
        "assistant row should show `—` in both cost and cum_cost; got: {assistant_row}",
    );

    // cum_cost MUST NOT be `$0.0000` once the unknown row hits.
    assert!(
        !user_row.contains("$0.0000"),
        "unknown-model row should not show $0.0000 cum_cost; got: {user_row}",
    );
    assert!(
        !assistant_row.contains("$0.0000"),
        "unknown-model row should not show $0.0000 cum_cost; got: {assistant_row}",
    );
}

#[test]
fn show_zero_usage_turn_renders_zero_cost() {
    // The aaaa7777 fixture starts with a `<synthetic>` user message
    // followed by an assistant turn whose usage is all zeros. That
    // turn must show `$0.0000` in the cost cell — not `—` — because
    // `cost_for_components` short-circuits on all-zero usage to
    // `Some(0.0)`.
    let cache = isolated_cache();
    let stdout = cclens_command(cache.path(), &pricing_fixture_url(COST_FIXTURE_URL_NAME))
        .args(["--projects-dir"])
        .arg(cost_projects_fixture_dir())
        .arg("show")
        .arg("aaaa7777-7777-7777-7777-777777777777")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let out = String::from_utf8(stdout).unwrap();
    let lines: Vec<&str> = out.lines().collect();

    let synthetic_assistant = lines
        .iter()
        .find(|l| l.contains("synthetic-only assistant turn"))
        .expect("synthetic assistant row missing");
    // Zero-usage assistant: cost cell is `$0.0000`.
    assert!(
        synthetic_assistant.contains("$0.0000"),
        "synthetic assistant row should show $0.0000 cost; got: {synthetic_assistant}",
    );
    assert!(
        !synthetic_assistant.contains('—'),
        "synthetic assistant row should not show `—` (model is known + zero usage); got: {synthetic_assistant}",
    );

    // The corresponding user row (`<synthetic>`) also has zero
    // tokens (input + cache_creation = 0 across the cluster), so its
    // cost cell must also be $0.0000 — same short-circuit. The user
    // row is the one whose content cell starts with `<synthetic>`.
    let synthetic_user = lines
        .iter()
        .find(|l| l.contains("<synthetic>"))
        .expect("synthetic user row missing");
    assert!(
        synthetic_user.contains("$0.0000"),
        "synthetic user row should show $0.0000 cost; got: {synthetic_user}",
    );
}

#[test]
fn list_prices_1h_cache_creation_at_1h_rate() {
    // The eeee9999 fixture has one assistant turn with
    // ephemeral_1h_input_tokens: 130_259 (and zero tokens elsewhere).
    // Against the test fixture's rates (1h: 1e-5, 5m: 1.875e-5):
    //   - 1h rate applied (correct):   130_259 * 1e-5    = $1.3026
    //   - 5m rate applied (regression): 130_259 * 1.875e-5 = $2.4424
    let cache = isolated_cache();
    let stdout = cclens_command(cache.path(), &pricing_fixture_url(COST_FIXTURE_URL_NAME))
        .args(["--projects-dir"])
        .arg(cost_projects_fixture_dir())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let out = String::from_utf8(stdout).unwrap();
    let session_uuid = "eeee9999-9999-9999-9999-999999999999";
    let row = out
        .lines()
        .find(|l| l.contains(session_uuid))
        .unwrap_or_else(|| panic!("1h cache-creation row missing:\n{out}"));

    assert!(
        row.contains("$1.3026"),
        "1h cache_creation tokens must be priced at the 1h rate \
         ($1.3026); got: {row}",
    );
    assert!(
        !row.contains("$2.4424"),
        "1h tokens were priced at the 5m rate — rate-split regression. \
         got: {row}",
    );
}

#[test]
fn list_degrades_gracefully_on_fetch_failure() {
    // Pointing CCLENS_PRICING_URL at a nonexistent file simulates a
    // first-run fetch failure. Behavior contract:
    //   - exit status 0 (list does not abort on cost-load failure)
    //   - every row's cost cell is `—`
    //   - other columns render normally
    //   - stderr contains a one-time "fetch" warning
    let cache = isolated_cache();
    let bogus_url = "file:///tmp/cclens-this-fixture-must-not-exist.json";
    let assertion = cclens_command(cache.path(), bogus_url)
        .args(["--projects-dir"])
        .arg(common::projects_fixture_dir())
        .arg("list")
        .assert()
        .success();
    let output = assertion.get_output();
    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    let stderr = String::from_utf8(output.stderr.clone()).unwrap();

    // Header is present; data rows are present; cost cells are all `—`.
    assert!(stdout.contains("datetime"), "header missing:\n{stdout}");
    assert!(
        stdout.contains("aaaa1111-1111-1111-1111-111111111111"),
        "alpha session row missing:\n{stdout}",
    );

    // No `$` should appear in any data line — because every cost
    // cell renders `—` when the catalog is empty. (Header has no `$`
    // either, so any `$` indicates a regression.)
    assert!(
        !stdout.contains('$'),
        "no row should show a `$` cost when catalog load fails; got:\n{stdout}",
    );

    // Stderr must mention the fetch failure exactly once.
    let fetch_warnings: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("failed to fetch"))
        .collect();
    assert_eq!(
        fetch_warnings.len(),
        1,
        "expected exactly one fetch-failure warning; got: {fetch_warnings:?}",
    );
}
