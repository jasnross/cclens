// Integration tests for `cclens agents`. Covers:
//   - cost-descending ranking across two projects
//   - per-transcript deduplication, asserted against a hand-computed sum
//   - the narrowing `--pinning` default and widening past it
//   - fork rows, and the em dash for a transcript recording no effort
//   - `--compare-model`: columns, footer labels, and the two ways a
//     bad target is rejected
//   - the empty-result hint naming only agents-scoped flags
//   - JSON output carrying rows plus the repricing summary
//
// Hermetic: every test uses isolated tempdirs for the pricing cache
// and the synthetic `~/.claude/` tree (via CCLENS_CLAUDE_HOME).

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod common;

use common::{
    agents_projects_fixture_dir, build_agents_claude_home, cclens_claude_home_command,
    pricing_fixture_url,
};

const PRICING_FIXTURE: &str = "litellm-mini.json";

fn isolated_tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

/// Run `cclens agents` against the fixture tree, returning
/// `(stdout, stderr)`. Panics unless the command succeeds.
fn run_agents(extra_args: &[&str]) -> (String, String) {
    let (stdout, stderr, success) = try_run_agents(extra_args);
    assert!(success, "expected success; stderr:\n{stderr}");
    (stdout, stderr)
}

/// Run `cclens agents` without asserting the exit status, so a test can
/// check a rejection.
fn try_run_agents(extra_args: &[&str]) -> (String, String, bool) {
    let cache = isolated_tempdir();
    let claude_home_owner = isolated_tempdir();
    let claude_home = build_agents_claude_home(claude_home_owner.path());
    let mut cmd = cclens_claude_home_command(
        cache.path(),
        &pricing_fixture_url(PRICING_FIXTURE),
        &claude_home,
    );
    cmd.args(["--projects-dir"])
        .arg(agents_projects_fixture_dir())
        .args(["agents", "--format", "plain"]);
    for a in extra_args {
        cmd.arg(a);
    }
    let output = cmd.output().expect("run cclens");
    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
        output.status.success(),
    )
}

/// The data rows of a rendered table: everything between the header
/// and the `total:` footer.
fn data_rows(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .skip_while(|l| !l.contains(" agent "))
        .skip(1)
        .take_while(|l| !l.starts_with("total:"))
        .filter(|l| !l.trim().is_empty())
        .collect()
}

/// The row whose `agent` and `pinning` cells match, panicking with the
/// whole table when none does.
fn row_for<'a>(stdout: &'a str, agent: &str, pinning: &str) -> &'a str {
    data_rows(stdout)
        .into_iter()
        .find(|l| {
            let cols: Vec<&str> = l.split_whitespace().collect();
            // Index 4, not `contains`: a model or effort cell equal
            // to a pinning label would otherwise match the wrong row.
            cols.first() == Some(&agent) && cols.get(4) == Some(&pinning)
        })
        .unwrap_or_else(|| panic!("no `{agent}` row with pinning `{pinning}` in:\n{stdout}"))
}

const EVERY_KIND: &str = "pinned,inherit,unpinned,no-agent-file,fork";

#[test]
fn agents_ranks_rows_by_cost_descending() {
    let (stdout, _) = run_agents(&["--pinning", EVERY_KIND]);
    // Columns after split_whitespace (no fixture cell contains a space):
    //   0: agent  1: model  2: effort  3: declared_effort  4: pinning
    //   5: dispatches  6: tokens  7: cost
    let costs: Vec<f64> = data_rows(&stdout)
        .iter()
        .map(|l| {
            let cols: Vec<&str> = l.split_whitespace().collect();
            cols[7]
                .trim_start_matches('$')
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("unparseable cost cell in row: {l}"))
        })
        .collect();
    assert!(costs.len() >= 5, "expected several rows, got {costs:?}");
    assert!(
        costs.windows(2).all(|w| w[0] >= w[1]),
        "rows must be ranked by cost descending, got {costs:?}\n{stdout}",
    );
}

#[test]
fn agents_deduplicates_repeated_assistant_turns() {
    // `agent-1.jsonl` repeats one `(message_id, request_id)` pair three
    // times at 900 input tokens. The row must report 900, not 2700 —
    // this is the figure every other number in the view rests on.
    let (stdout, _) = run_agents(&["--pinning", EVERY_KIND]);
    let row = row_for(&stdout, "pinned-agent", "pinned");
    let cols: Vec<&str> = row.split_whitespace().collect();
    assert_eq!(cols[5], "1", "one dispatch, not three rows: {row}");
    assert_eq!(cols[6], "0.90k", "hand-computed deduplicated sum: {row}");
}

#[test]
fn agents_default_slice_excludes_pinned_rows() {
    let (stdout, _) = run_agents(&[]);
    assert!(
        !stdout.contains(" pinned "),
        "the default slice must hide pinned rows:\n{stdout}",
    );
    assert!(
        !stdout.contains(" fork "),
        "the default slice must hide fork rows:\n{stdout}",
    );
    // It does admit the other three, and says so unconditionally —
    // no filter component exists at the default to say it otherwise.
    assert!(
        stdout.contains("pinning: inherit,unpinned,no-agent-file"),
        "the footer must name the default slice:\n{stdout}",
    );
}

#[test]
fn agents_pinning_flag_widens_to_every_row() {
    let (narrow, _) = run_agents(&[]);
    let (wide, _) = run_agents(&["--pinning", EVERY_KIND]);
    assert!(
        data_rows(&wide).len() > data_rows(&narrow).len(),
        "widening the slice must reveal rows\nnarrow:\n{narrow}\nwide:\n{wide}",
    );
    // Spaced, so the assertion is not satisfied by `unpinned`.
    assert!(wide.contains(" pinned "), "{wide}");
}

#[test]
fn agents_marks_a_fork_row() {
    // `agent-2.meta.json` carries `isFork: true` while naming an agent
    // type whose file is pinned. The structural discriminator wins, so
    // the row reads `fork` and declares no effort.
    let (wide, _) = run_agents(&["--pinning", EVERY_KIND]);
    let row = row_for(&wide, "pinned-agent", "fork");
    let cols: Vec<&str> = row.split_whitespace().collect();
    assert_eq!(
        cols[3], "—",
        "a fork has no agent definition, so it declares no effort: {row}",
    );

    // And it is outside the default slice.
    let (narrow, _) = run_agents(&[]);
    assert!(!narrow.contains(" fork "), "{narrow}");
}

#[test]
fn agents_renders_em_dash_for_a_transcript_without_effort() {
    // `agent-3.jsonl` records no top-level `effort` key, so the cell is
    // a visible gap rather than a level the run did not use.
    let (stdout, _) = run_agents(&[]);
    let row = row_for(&stdout, "unpinned-agent", "unpinned");
    let cols: Vec<&str> = row.split_whitespace().collect();
    assert_eq!(cols[2], "—", "effort cell should be an em dash: {row}");
}

#[test]
fn agents_compare_model_adds_repriced_columns_and_footer() {
    let (stdout, _) = run_agents(&["--compare-model", "claude-sonnet-4-6"]);
    assert!(stdout.contains("repriced"), "{stdout}");
    assert!(stdout.contains("delta"), "{stdout}");
    assert!(stdout.contains("vs claude-sonnet-4-6"), "{stdout}");
    assert!(
        stdout.contains("upper bound"),
        "the footer must label the figure a bound:\n{stdout}",
    );
}

#[test]
fn agents_compare_model_reports_the_rows_it_excluded() {
    // With forks admitted, the repricing footer must say it skipped
    // one — a delta summed over a shrunken subset otherwise passes as
    // a complete one.
    let (stdout, _) = run_agents(&[
        "--pinning",
        EVERY_KIND,
        "--compare-model",
        "claude-sonnet-4-6",
    ]);
    assert!(
        stdout.contains("excluded 1 fork row"),
        "the footer must count the fork rows it dropped:\n{stdout}",
    );
    // And the per-row column must agree with the footer it sits above:
    // a priced fork cell would make the visible column sum differ.
    let fork_row = row_for(&stdout, "pinned-agent", "fork");
    let cols: Vec<&str> = fork_row.split_whitespace().collect();
    assert_eq!(cols[8], "—", "repriced cell: {fork_row}");
    assert_eq!(cols[9], "—", "delta cell: {fork_row}");
}

#[test]
fn agents_compare_model_rejects_an_unknown_model() {
    let (_, stderr, success) = try_run_agents(&["--compare-model", "no-such-model"]);
    assert!(!success, "expected a nonzero exit");
    assert!(
        stderr.contains("no-such-model"),
        "the error must name the value:\n{stderr}",
    );
}

#[test]
fn agents_compare_model_rejects_an_alias_lookup_would_have_resolved() {
    // `sonnet-4-6` is a spelling `PricingCatalog::lookup` resolves via
    // its `claude-` prefix stage. `--compare-model` must refuse it
    // rather than reprice against a model nobody named.
    let (_, stderr, success) = try_run_agents(&["--compare-model", "sonnet-4-6"]);
    assert!(
        !success,
        "an alias the fallback chain resolves must still be rejected:\n{stderr}",
    );
    assert!(stderr.contains("sonnet-4-6"), "{stderr}");
}

#[test]
fn agents_empty_result_hint_names_only_agents_scoped_flags() {
    let (stdout, stderr) = run_agents(&["--project", "no-such-project", "--min-tokens", "999999"]);
    assert!(data_rows(&stdout).is_empty(), "{stdout}");
    assert!(
        stderr.contains("no rows matched"),
        "expected an empty-result hint:\n{stderr}",
    );
    assert!(stderr.contains("--project no-such-project"), "{stderr}");
    assert!(stderr.contains("--min-tokens 999999"), "{stderr}");
    // `--session` is inputs-only; the agents loader never applies it,
    // so it has no place in a hint that asserts causation.
    assert!(!stderr.contains("--session"), "{stderr}");
}

#[test]
fn agents_json_output_carries_rows_and_repriced_summary() {
    let cache = isolated_tempdir();
    let claude_home_owner = isolated_tempdir();
    let claude_home = build_agents_claude_home(claude_home_owner.path());
    let mut cmd = cclens_claude_home_command(
        cache.path(),
        &pricing_fixture_url(PRICING_FIXTURE),
        &claude_home,
    );
    cmd.args(["--projects-dir"])
        .arg(agents_projects_fixture_dir())
        .args([
            "agents",
            "--format",
            "json",
            "--compare-model",
            "claude-sonnet-4-6",
        ]);
    let stdout = String::from_utf8(cmd.assert().success().get_output().stdout.clone()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");

    let rows = parsed["rows"].as_array().expect("rows array");
    assert!(!rows.is_empty(), "{stdout}");
    let first = &rows[0];
    assert!(first["agent_type"].is_string(), "{stdout}");
    for key in [
        "agent_type",
        "model",
        "effort",
        "declared_effort",
        "pinning",
        "dispatches",
        "usage",
        "cost",
    ] {
        assert!(first.get(key).is_some(), "missing key {key} in:\n{stdout}");
    }

    let repriced = &parsed["repriced"];
    for key in ["delta", "rows_counted", "rows_unpriced", "rows_forked"] {
        assert!(
            repriced.get(key).is_some(),
            "missing repriced.{key} in:\n{stdout}",
        );
    }
}

#[test]
fn agents_json_omits_the_repriced_summary_without_a_target() {
    let cache = isolated_tempdir();
    let claude_home_owner = isolated_tempdir();
    let claude_home = build_agents_claude_home(claude_home_owner.path());
    let mut cmd = cclens_claude_home_command(
        cache.path(),
        &pricing_fixture_url(PRICING_FIXTURE),
        &claude_home,
    );
    cmd.args(["--projects-dir"])
        .arg(agents_projects_fixture_dir())
        .args(["agents", "--format", "json"]);
    let stdout = String::from_utf8(cmd.assert().success().get_output().stdout.clone()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert!(parsed["repriced"].is_null(), "{stdout}");
}

#[test]
fn agents_skips_a_subagent_with_no_sidecar() {
    // `agent-4.jsonl` carries 99999 input tokens and no `.meta.json`.
    // If it were counted, it would dominate every ranking.
    let (stdout, _) = run_agents(&["--pinning", EVERY_KIND]);
    assert!(
        !stdout.contains("99.99k"),
        "a sidecar-less subagent must be skipped:\n{stdout}",
    );
}
