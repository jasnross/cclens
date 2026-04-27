// Integration tests for `cclens inputs`. Covers:
//   - empty `--projects-dir` rendering
//   - global-scope attribution against synthetic JSONL events
//   - plugin walker handling the literal `unknown` version segment
//   - filter behavior: `--session`, `--since` / `--until`,
//     `--min-cost` (presentation-only — coverage line stays
//     byte-identical between filtered/unfiltered runs)
//   - coverage line: both-tier ratios + n/a when a tier has no events
//   - unknown-model em-dash on 5m-tier rows
//   - the "observed = independent JSONL sum" fixed-point invariant
//     end-to-end, with and without filters
//
// Hermetic: every test uses isolated tempdirs for the pricing cache
// and the synthetic `~/.claude/` tree (via CCLENS_CLAUDE_HOME).

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use common::{
    build_inputs_claude_home, cclens_command, cclens_inputs_command, copy_dir_recursive,
    inputs_projects_fixture_dir, pricing_fixture_url,
};
use regex::Regex;

const PRICING_FIXTURE: &str = "litellm-mini.json";

fn isolated_tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn run_inputs(extra_args: &[&str]) -> (String, String) {
    // Tempdirs live for the function's scope — they're dropped (and
    // their directories removed) when this function returns, after
    // `cmd.assert()` has already completed and stdout/stderr have
    // been copied into owned `String`s.
    let cache = isolated_tempdir();
    let claude_home_owner = isolated_tempdir();
    let claude_home = build_inputs_claude_home(claude_home_owner.path());
    let mut cmd = cclens_inputs_command(
        cache.path(),
        &pricing_fixture_url(PRICING_FIXTURE),
        &claude_home,
    );
    cmd.args(["--projects-dir"])
        .arg(inputs_projects_fixture_dir())
        .arg("inputs");
    for a in extra_args {
        cmd.arg(a);
    }
    let output = cmd.assert().success();
    let raw = output.get_output();
    let stdout = String::from_utf8(raw.stdout.clone()).unwrap();
    let stderr = String::from_utf8(raw.stderr.clone()).unwrap();
    (stdout, stderr)
}

/// `(attributed, observed)` pair parsed from one half of the
/// coverage line. `None` when the half rendered as `n/a`.
type TierFigures = Option<(u64, u64)>;

/// Parse the rendered coverage line into its observed / attributed
/// figures per tier. Returns `(long_1h, short_5m)` — each half is
/// `None` when that tier rendered as `n/a`.
///
/// Sample input:
///   `coverage: 1h: 36.6% (183 / 500 1h-tokens) | 5m: 46.0% (23 / 50 5m-tokens)`
fn parse_coverage(line: &str) -> (TierFigures, TierFigures) {
    static ONE_H_RE: OnceLock<Regex> = OnceLock::new();
    static FIVE_M_RE: OnceLock<Regex> = OnceLock::new();
    let one_h_re = ONE_H_RE.get_or_init(|| Regex::new(r"\((\d+) / (\d+) 1h-tokens\)").unwrap());
    let five_m_re = FIVE_M_RE.get_or_init(|| Regex::new(r"\((\d+) / (\d+) 5m-tokens\)").unwrap());
    let one_h = one_h_re.captures(line).map(|c| {
        let attr = c[1].parse::<u64>().unwrap();
        let obs = c[2].parse::<u64>().unwrap();
        (attr, obs)
    });
    let five_m = five_m_re.captures(line).map(|c| {
        let attr = c[1].parse::<u64>().unwrap();
        let obs = c[2].parse::<u64>().unwrap();
        (attr, obs)
    });
    (one_h, five_m)
}

fn coverage_line(stdout: &str) -> &str {
    stdout
        .lines()
        .find(|l| l.starts_with("coverage:"))
        .expect("coverage line present")
}

#[test]
fn inputs_renders_empty_when_no_sessions() {
    // Empty projects-dir: no sessions, so coverage is n/a on both
    // tiers (no observed tokens).
    let cache = isolated_tempdir();
    let claude_home_owner = isolated_tempdir();
    let claude_home = build_inputs_claude_home(claude_home_owner.path());
    let empty_projects = isolated_tempdir();
    let stdout = cclens_inputs_command(
        cache.path(),
        &pricing_fixture_url(PRICING_FIXTURE),
        &claude_home,
    )
    .args(["--projects-dir"])
    .arg(empty_projects.path())
    .arg("inputs")
    .assert()
    .success()
    .get_output()
    .stdout
    .clone();
    let out = String::from_utf8(stdout).unwrap();
    let line = coverage_line(&out);
    assert!(line.contains("1h: n/a"), "got: {line}");
    assert!(line.contains("5m: n/a"), "got: {line}");
}

#[test]
fn inputs_attributes_global_claude_md_to_all_sessions() {
    // Three sessions in the fixture: A (events_1h=2), B (events_1h=1),
    // C (events_5m=1, unknown model). Global CLAUDE.md is in scope for
    // all three; the events column should reflect events_1h sum (2+1=3).
    // Session C contributes 0 to events_1h, so its unknown-model
    // status doesn't collapse the 1h-tier rows' cost.
    let (stdout, _) = run_inputs(&[]);
    // Locate by the `global` kind label rather than the path
    // (paths truncate past INPUTS_PATH_MAX_CHARS).
    let global_row = stdout
        .lines()
        .find(|l| l.contains(" global "))
        .unwrap_or_else(|| panic!("global CLAUDE.md row missing:\n{stdout}"));
    // Columns after split_whitespace (each cell is a single token in
    // our fixture rows since paths/kinds/numbers contain no spaces):
    //   0: file  1: kind  2: tier  3: tokens  4: events  5: billed  6: cost
    let cols: Vec<&str> = global_row.split_whitespace().collect();
    assert_eq!(
        cols.get(4),
        Some(&"3"),
        "events column (idx 4) should be 3 (events_1h sum across sessions A+B+C); \
         row: {global_row}\n(cols: {cols:?})",
    );
}

#[test]
fn inputs_handles_plugin_with_unknown_version_segment() {
    // The fixture's installed_plugins.json points at
    // .../test-plugin/unknown/skills/test/SKILL.md. Walker must
    // handle the literal `unknown` version segment and surface the
    // plugin's skill in the table.
    let (stdout, _) = run_inputs(&[]);
    assert!(
        stdout.contains("plugin:test-plugin:skill"),
        "expected plugin skill row; got:\n{stdout}",
    );
}

#[test]
fn inputs_filter_session_isolates_one_session() {
    // --session aaaa1111-... should restrict events_1h to session A
    // (events_1h=2). Global CLAUDE.md row's events column should be
    // 2, not 3.
    let (stdout, _) = run_inputs(&["--session", "aaaa1111-1111-1111-1111-111111111111"]);
    let cov = coverage_line(&stdout);
    let (one_h, _five_m) = parse_coverage(cov);
    let (_attr, obs) = one_h.expect("1h tier should have data");
    assert_eq!(
        obs, 200,
        "session A has events_1h=2 each carrying 100 tokens, \
         observed_1h should be 200; got {obs} (line: {cov})",
    );
}

#[test]
fn inputs_filter_project_isolates_one_project() {
    // --project proj-b should restrict to session B's contributions
    // (cwd /proj-b → project_short_name "proj-b"). Observed_1h
    // should equal session B's 300 tokens.
    let (stdout, _) = run_inputs(&["--project", "proj-b"]);
    let cov = coverage_line(&stdout);
    let (one_h, _) = parse_coverage(cov);
    let (_attr, obs) = one_h.expect("1h tier should have data");
    assert_eq!(
        obs, 300,
        "--project proj-b should isolate session B (observed_1h=300); \
         got {obs} (line: {cov})",
    );
}

#[test]
fn inputs_filter_since_until_excludes_outside_range() {
    // --since 2026-04-18 --until 2026-04-22T00:00:00Z brackets only
    // session B (started 2026-04-20). Observed_1h should be 300.
    let (stdout, _) = run_inputs(&[
        "--since",
        "2026-04-18T00:00:00Z",
        "--until",
        "2026-04-22T00:00:00Z",
    ]);
    let cov = coverage_line(&stdout);
    let (one_h, _) = parse_coverage(cov);
    let (_attr, obs) = one_h.expect("1h tier should have data");
    assert_eq!(
        obs, 300,
        "expected observed_1h=300 (session B only); got {obs}"
    );
}

#[test]
fn inputs_filter_min_cost_drops_below_threshold_rows_but_preserves_coverage() {
    // Critical regression guard: --min-cost is a presentation-only
    // row filter. The coverage line must be byte-identical between
    // unfiltered and filtered runs.
    let (unfiltered, _) = run_inputs(&[]);
    let (filtered, _) = run_inputs(&["--min-cost", "0.001"]);

    let unfiltered_cov = coverage_line(&unfiltered);
    let filtered_cov = coverage_line(&filtered);
    assert_eq!(
        unfiltered_cov, filtered_cov,
        "coverage line must be byte-identical between filtered and \
         unfiltered runs; --min-cost wired into compute_coverage \
         is a regression",
    );

    // Sanity: the filter actually dropped rows. Count data lines
    // (everything between the header and the coverage footer).
    let count = |s: &str| {
        s.lines()
            .filter(|l| l.contains("CLAUDE.md") || l.contains("SKILL.md") || l.contains("agent"))
            .count()
    };
    let uf = count(&unfiltered);
    let f = count(&filtered);
    assert!(
        f < uf,
        "expected filtered run ({f} rows) to drop rows from unfiltered ({uf} rows); \
         output:\n--unfiltered--\n{unfiltered}\n--filtered--\n{filtered}",
    );
}

#[test]
fn inputs_coverage_indicator_renders_both_tier_ratios() {
    // Both tiers have data in this fixture (1h: 500, 5m: 50 — though
    // 5m is unknown-model so attributed=0 there). Both halves of the
    // coverage line must show a percent (or n/a if attributed=0,
    // observed>0 still renders as a real ratio).
    let (stdout, _) = run_inputs(&[]);
    let cov = coverage_line(&stdout);
    // 1h half: opus sessions A+B contribute 200+300 = 500 observed.
    assert!(cov.contains("/ 500 1h-tokens"), "got: {cov}");
    // 5m half: session C contributes 50 observed.
    assert!(cov.contains("/ 50 5m-tokens"), "got: {cov}");
}

#[test]
fn inputs_handles_unknown_model_session_with_em_dash() {
    // Per-load math: session C (claude-fake-9-9) attributes its
    // CLAUDE.md/rule loads at the 5m tier (its primary_tier) because
    // its only cache_creation event is 5m. The unknown model
    // collapses cost on every row session C touches — so the global
    // CLAUDE.md and global rule rows render `—` in the cost column
    // (they're loaded by session C *and* the priced sessions A/B).
    let (stdout, _) = run_inputs(&[]);
    let global_row = stdout
        .lines()
        .find(|l| l.contains(" global "))
        .unwrap_or_else(|| panic!("global CLAUDE.md row missing:\n{stdout}"));
    assert!(
        global_row.contains('—'),
        "global CLAUDE.md (loaded by unknown-model session C) should show em-dash cost; got: {global_row}",
    );
}

#[test]
#[allow(clippy::similar_names)]
fn inputs_observed_tokens_match_independent_jsonl_sums() {
    // Fixed-point invariant: cclens's reported observed_*_tokens
    // must equal an independent walker's sums of the same JSONL data.
    //
    // Per the fixture: A has events of (5m=0,1h=100) twice, B has one
    // (5m=0,1h=300) event, C has one (5m=50,1h=0). Independent sums:
    //   1h: 100 + 100 + 300 = 500
    //   5m: 50 = 50
    let (stdout, _) = run_inputs(&[]);
    let cov = coverage_line(&stdout);
    let (one_h, five_m) = parse_coverage(cov);
    let (_, obs_1h) = one_h.expect("1h tier present");
    let (_, obs_5m) = five_m.expect("5m tier present");
    assert_eq!(obs_1h, 500, "observed_1h must equal independent sum");
    assert_eq!(obs_5m, 50, "observed_5m must equal independent sum");
}

#[test]
fn inputs_observed_1h_matches_jsonl_sum_under_filters() {
    // Same invariant under --since: only session B's 300 should
    // remain in the observed total. A regression that excluded B
    // from the row list but forgot to exclude it from coverage
    // (or vice versa) would diverge here.
    let (stdout, _) = run_inputs(&[
        "--since",
        "2026-04-18T00:00:00Z",
        "--until",
        "2026-04-22T00:00:00Z",
    ]);
    let cov = coverage_line(&stdout);
    let (one_h, _) = parse_coverage(cov);
    let (_, obs_1h) = one_h.expect("1h tier present after filter");
    assert_eq!(
        obs_1h, 300,
        "observed_1h under filter must equal session B's 300; got {obs_1h}",
    );
}

#[test]
fn inputs_subagent_credits_matching_agent_and_extra_load_for_always_loaded_files() {
    // Phase 2 contract: a subagent transcript credits the matching
    // agent file (via `agentType` in `.meta.json`) and contributes
    // one additional load to every always-loaded file in scope
    // (CLAUDE.md, rules). It also enters the 5m observed-tokens sum.
    //
    // We assert this by comparing two runs: vanilla fixture vs.
    // fixture with one planted subagent. Specific deltas:
    //   - test-agent row: 0 loads → 1 load.
    //   - global CLAUDE.md: loads += 1 (now 4 instead of 3).
    //   - 5m observed tokens: += 50 (subagent's ephemeral_5m).
    //   - cclens list output stays identical (subagents aren't
    //     standalone sessions for the listing browser).
    let cache = isolated_tempdir();
    let claude_home_owner = isolated_tempdir();
    let claude_home = build_inputs_claude_home(claude_home_owner.path());

    // Run #1: vanilla fixture, no subagents/ on disk.
    let projects_owner_a = isolated_tempdir();
    let projects_a = projects_owner_a.path().join("inputs-projects");
    copy_dir_recursive(&inputs_projects_fixture_dir(), &projects_a);
    let baseline = run_inputs_against(cache.path(), &claude_home, &projects_a);

    // Run #2: same fixture with a subagent JSONL + meta.json planted
    // under one session's <stem>/subagents/ directory. The subagent
    // references `agentType: test-agent` (matches the inventory's
    // agents/test-agent.md → identifier "test-agent").
    let projects_owner_b = isolated_tempdir();
    let projects_b = projects_owner_b.path().join("inputs-projects");
    copy_dir_recursive(&inputs_projects_fixture_dir(), &projects_b);
    let session_stem = "aaaa1111-1111-1111-1111-111111111111";
    let subagents_dir = projects_b
        .join("-test-inputs")
        .join(session_stem)
        .join("subagents");
    fs::create_dir_all(&subagents_dir).expect("create subagents dir");
    fs::write(
        subagents_dir.join("agent-1.jsonl"),
        r#"{"type":"user","timestamp":"2026-04-15T10:02:00Z","cwd":"/proj-a","message":{"content":"sub"}}
{"type":"assistant","timestamp":"2026-04-15T10:02:05Z","cwd":"/proj-a","message":{"id":"msg_sub1","model":"claude-opus-4-7","usage":{"input_tokens":0,"output_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":50,"ephemeral_1h_input_tokens":0},"cache_read_input_tokens":0},"content":[{"type":"text","text":"sub-reply"}]},"requestId":"req_sub1"}
"#,
    )
    .expect("write subagent jsonl");
    fs::write(
        subagents_dir.join("agent-1.meta.json"),
        r#"{"agentType":"test-agent","description":"test agent body"}"#,
    )
    .expect("write subagent meta");
    let with_subagents = run_inputs_against(cache.path(), &claude_home, &projects_b);

    // Cell extraction: rows are space-separated; cell at index 4 is
    // `loads`. Locate by kind label since paths truncate.
    let loads_for_kind = |stdout: &str, kind: &str| -> u64 {
        let row = stdout
            .lines()
            .find(|l| l.contains(&format!(" {kind} ")))
            .unwrap_or_else(|| panic!("row for kind `{kind}` missing in:\n{stdout}"));
        row.split_whitespace()
            .nth(4)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("could not parse loads cell from row: {row}"))
    };

    let baseline_global_loads = loads_for_kind(&baseline, "global");
    let with_global_loads = loads_for_kind(&with_subagents, "global");
    assert_eq!(
        with_global_loads,
        baseline_global_loads + 1,
        "subagent should add one load to the global CLAUDE.md row\n\nbaseline:\n{baseline}\n\nwith subagents:\n{with_subagents}",
    );

    let baseline_agent_loads = loads_for_kind(&baseline, "agent");
    let with_agent_loads = loads_for_kind(&with_subagents, "agent");
    assert_eq!(
        baseline_agent_loads, 0,
        "no subagent → no agent load attributed",
    );
    assert_eq!(
        with_agent_loads, 1,
        "matching subagent should credit the agent file once\n\n{with_subagents}",
    );

    // 5m observed tokens delta: subagent contributed 50 ephemeral_5m
    // tokens. Parse the coverage line to verify it lands in the
    // denominator.
    let baseline_cov = coverage_line(&baseline);
    let with_cov = coverage_line(&with_subagents);
    let (_, baseline_5m) = parse_coverage(baseline_cov);
    let (_, with_5m) = parse_coverage(with_cov);
    let (_, baseline_5m_obs) = baseline_5m.expect("5m observed in baseline");
    let (_, with_5m_obs) = with_5m.expect("5m observed with subagent");
    assert_eq!(
        with_5m_obs,
        baseline_5m_obs + 50,
        "subagent's 50 ephemeral_5m tokens must enter the 5m denominator",
    );

    // cclens list view stays byte-identical — subagents aren't
    // standalone sessions for the listing browser.
    let baseline_list = run_list_against(cache.path(), &projects_a);
    let with_subagents_list = run_list_against(cache.path(), &projects_b);
    assert_eq!(
        baseline_list, with_subagents_list,
        "cclens list output must stay byte-identical when only subagents are added",
    );
}

fn run_inputs_against(cache: &Path, claude_home: &Path, projects_dir: &Path) -> String {
    let mut cmd = cclens_inputs_command(cache, &pricing_fixture_url(PRICING_FIXTURE), claude_home);
    cmd.args(["--projects-dir"]).arg(projects_dir).arg("inputs");
    let raw = cmd.assert().success().get_output().stdout.clone();
    String::from_utf8(raw).unwrap()
}

fn run_list_against(cache: &Path, projects_dir: &Path) -> String {
    let mut cmd = cclens_command(cache, &pricing_fixture_url(PRICING_FIXTURE));
    cmd.args(["--projects-dir"]).arg(projects_dir).arg("list");
    let raw = cmd.assert().success().get_output().stdout.clone();
    String::from_utf8(raw).unwrap()
}
