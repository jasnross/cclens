// Listing/show integration tests must be hermetic — they all set
// `CCLENS_PRICING_URL` to a local fixture and `CCLENS_CACHE_DIR` to a
// fresh tempdir so they never read or write the user's real cache.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::time::{Duration, SystemTime};

use common::{
    cclens_command, copy_fixture_to_tempdir_with_mtimes, dedup_projects_fixture_dir,
    pricing_fixture_url, projects_fixture_dir,
};

fn isolated_cache() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

#[test]
fn list_renders_sessions_oldest_first_with_correct_totals() {
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
    let lines: Vec<&str> = stdout.lines().collect();

    let header = lines
        .iter()
        .find(|l| l.contains("datetime"))
        .expect("header row missing");
    for col in ["project", "title", "tokens", "cost", "id"] {
        assert!(header.contains(col), "header missing {col}: {header}");
    }

    // gamma has no assistant turns, so the zero-billable AND zero-cost
    // filter drops it. Scope the check to non-header lines.
    assert!(
        lines
            .iter()
            .filter(|l| !l.contains("datetime"))
            .all(|l| !l.contains("gamma")),
        "expected no row to reference project 'gamma' (zero-billable filter); stdout was:\n{stdout}",
    );

    // Column order is now: datetime project title tokens cost id.
    // Strip the trailing UUID, then strip the cost cell (always
    // present since fixtures use `claude-opus-4-7`), to put tokens
    // back at the line end for the per-row token assertions.
    let strip_uuid_and_cost = |l: &str, uuid: &str| -> (String, String) {
        let after_uuid = l
            .trim_end()
            .strip_suffix(uuid)
            .unwrap_or_else(|| panic!("row missing expected trailing UUID {uuid}:\n{l}"))
            .trim_end();
        // The cost cell is `$X.XXXX` for any priced session.
        let dollar_idx = after_uuid
            .rfind('$')
            .unwrap_or_else(|| panic!("expected `$` cost cell in row:\n{l}"));
        let cost_cell = after_uuid[dollar_idx..].to_string();
        let pre_cost = after_uuid[..dollar_idx].trim_end().to_string();
        (pre_cost, cost_cell)
    };

    // Hand-computed costs from the pricing fixture's claude-opus-4-7
    // rates (input=15e-6, output=75e-6, cache_creation=18.75e-6,
    // cache_read=1.5e-6) applied to each session's assistant turns.
    //
    // alpha: turn1 (100,200,300,9999) + turn2 (50,0,0,0)
    //   = 0.0015 + 0.015 + 0.005625 + 0.0149985  + 0.00075
    //   = 0.0378735 → "$0.0379"
    //
    // beta-prose: turn (10,90,0,0)
    //   = 0.00015 + 0.00675
    //   = 0.0069 → "$0.0069"
    //
    // beta-after-malformed: turn (5,5,0,0)
    //   = 0.000075 + 0.000375
    //   = 0.00045 → "$0.0004" (Rust's `{:.4}` banker's-rounds the
    //   tie-on-an-odd-digit, plus the f64 representation of 0.00045
    //   is slightly less than the rational value, so it rounds down)
    let alpha_uuid = "aaaa1111-1111-1111-1111-111111111111";
    let alpha = lines
        .iter()
        .find(|l| l.contains("/test-cmd hello world"))
        .expect("alpha row missing");
    assert!(alpha.contains("alpha"));
    assert!(alpha.contains(alpha_uuid));
    let (alpha_pre_cost, alpha_cost) = strip_uuid_and_cost(alpha, alpha_uuid);
    assert!(
        alpha_pre_cost.ends_with(" 0.65k"),
        "alpha tokens column: {alpha_pre_cost}",
    );
    assert_eq!(alpha_cost, "$0.0379", "alpha cost cell");

    let beta_prose_uuid = "bbbb2222-2222-2222-2222-222222222222";
    let beta_prose = lines
        .iter()
        .find(|l| l.contains("How do I configure neovim folds?"))
        .expect("beta-prose row missing");
    assert!(beta_prose.contains("beta"));
    assert!(beta_prose.contains(beta_prose_uuid));
    let (beta_prose_pre, beta_prose_cost) = strip_uuid_and_cost(beta_prose, beta_prose_uuid);
    assert!(
        beta_prose_pre.ends_with(" 0.10k"),
        "beta-prose tokens: {beta_prose_pre}",
    );
    assert_eq!(beta_prose_cost, "$0.0069", "beta-prose cost cell");

    let beta_late_uuid = "dddd4444-4444-4444-4444-444444444444";
    let beta_late = lines
        .iter()
        .find(|l| l.contains("after malformed"))
        .expect("after-malformed row missing");
    assert!(beta_late.contains("beta"));
    assert!(beta_late.contains(beta_late_uuid));
    let (beta_late_pre, beta_late_cost) = strip_uuid_and_cost(beta_late, beta_late_uuid);
    assert!(
        beta_late_pre.ends_with(" 0.01k"),
        "beta-late tokens: {beta_late_pre}",
    );
    assert_eq!(beta_late_cost, "$0.0004", "beta-late cost cell");

    // Ordering: alpha (2026-04-01) < beta-prose (2026-04-15) <
    // beta-after-malformed (2026-04-20).
    let alpha_idx = lines
        .iter()
        .position(|l| l.contains("/test-cmd hello world"))
        .unwrap();
    let beta_prose_idx = lines
        .iter()
        .position(|l| l.contains("How do I configure neovim folds?"))
        .unwrap();
    let beta_late_idx = lines
        .iter()
        .position(|l| l.contains("after malformed"))
        .unwrap();
    assert!(alpha_idx < beta_prose_idx);
    assert!(beta_prose_idx < beta_late_idx);
}

#[test]
fn list_handles_empty_projects_dir() {
    let cache = isolated_cache();
    let tmp = tempfile::tempdir().unwrap();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(tmp.path())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    // With no sessions, the only non-empty lines should be the header.
    let data_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.contains("datetime"))
        .collect();
    assert!(
        data_lines.is_empty(),
        "empty projects_dir should produce no data rows; got:\n{stdout}",
    );
}

#[test]
fn show_renders_per_exchange_table_with_tool_loop_collapse_and_orphan_user() {
    let cache = isolated_cache();
    let out = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg("eeee5555-5555-5555-5555-555555555555")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();

    let header = lines
        .iter()
        .find(|l| l.contains("datetime"))
        .expect("header row missing");
    for col in [
        "role",
        "tokens",
        "cost",
        "cumulative",
        "cum_cost",
        "content",
    ] {
        assert!(header.contains(col), "header missing {col}: {header}");
    }

    // (content-marker, tokens, cost, cumulative, cum_cost) for each
    // visible row.
    //
    // Costs are computed from the fixture's claude-opus-4-7 rates
    // applied to the per-row token decomposition (user rows get
    // input + cache_creation + cache_read costs across the assistant
    // cluster; assistant rows get output costs across the cluster).
    //
    // Exchange 1 (/test-cmd demo) has 3 assistant turns:
    //   T3: in=100 out=50 cc=200 cr=50
    //   T5: in=10  out=150 cc=0 cr=0
    //   T7: in=5   out=80  cc=0 cr=0
    //   user-row cost = (115*15 + 200*18.75 + 50*1.5) * 1e-6
    //                 = 0.001725 + 0.00375 + 0.000075
    //                 = 0.00555 → "$0.0056"
    //   asst-row cost = (50+150+80) * 75e-6 = 280 * 0.000075
    //                 = 0.021 → "$0.0210"
    //   running cum_cost (with f64 accumulation):
    //     0.00555  (user row)              → "$0.0056"
    //     0.02655  (asst row)              → "$0.0265"
    //     0.028725 (exchange 2 user row)   → "$0.0287"
    //     0.033225 (exchange 2 asst row)   → "$0.0332"
    //     0.033225 (orphan row, +Some(0))  → "$0.0332"
    //
    // Some `:.4` outputs end in *4 / *5 rather than the rational
    // half-even result because of f64 representation: e.g. 0.02655
    // is stored as 0.026549999..., which rounds to 0.0265.
    //
    // Exchange 2 (follow-up question) has 1 assistant turn:
    //   T9: in=20 out=60 cc=100 cr=0
    //   user-row cost = (20*15 + 100*18.75 + 0) * 1e-6
    //                 = 0.0003 + 0.001875 = 0.002175 → "$0.0022"
    //   asst-row cost = 60*75e-6 = 0.0045 → "$0.0045"
    //
    // Exchange 3 (third question, orphan): tokens=`—` cost=`—`
    //   user_cost_delta = Some(0.0) so cum_cost stays at 0.033225.
    let expected = [
        ("/test-cmd demo", " 0.32k ", "$0.0056", "0.32k", "$0.0056"),
        (
            "reading the file +2 tool uses",
            " 0.28k ",
            "$0.0210",
            "0.59k",
            "$0.0265",
        ),
        (
            "follow-up question",
            " 0.12k ",
            "$0.0022",
            "0.71k",
            "$0.0287",
        ),
        (" answer ", " 0.06k ", "$0.0045", "0.78k", "$0.0332"),
        (
            "third question with no response",
            "—",
            "—",
            "0.78k",
            "$0.0332",
        ),
    ];
    let mut positions: Vec<usize> = Vec::with_capacity(expected.len());
    for (marker, tokens, cost, cumulative, cum_cost) in expected {
        let idx = lines
            .iter()
            .position(|l| l.contains(marker))
            .unwrap_or_else(|| panic!("row {marker:?} missing:\n{stdout}"));
        let line = lines[idx];
        assert!(
            line.contains(tokens),
            "row {marker:?} missing tokens {tokens:?}: {line}",
        );
        assert!(
            line.contains(cost),
            "row {marker:?} missing cost {cost:?}: {line}",
        );
        assert!(
            line.contains(cumulative),
            "row {marker:?} missing cumulative {cumulative:?}: {line}",
        );
        assert!(
            line.contains(cum_cost),
            "row {marker:?} missing cum_cost {cum_cost:?}: {line}",
        );
        positions.push(idx);
    }
    for window in positions.windows(2) {
        assert!(
            window[0] < window[1],
            "expected chronological order; positions: {positions:?}",
        );
    }
}

#[test]
fn show_errors_on_unknown_session_id() {
    let cache = isolated_cache();
    let unknown = "00000000-0000-0000-0000-000000000000";
    let output = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg(unknown)
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone();
    let stderr = String::from_utf8(output).unwrap();
    assert!(
        stderr.contains("no session"),
        "stderr should contain 'no session'; got:\n{stderr}",
    );
    assert!(
        stderr.contains(unknown),
        "stderr should contain the unknown id; got:\n{stderr}",
    );
}

#[test]
fn list_dedups_resumed_session_assistant_turns() {
    // Two .jsonl files in one project share an assistant turn:
    //   dddd0001 (original, mtime older): msg_A (input 100), msg_B (input 200)
    //   dddd0002 (resumed,  mtime newer): msg_A (replay), msg_C (input 400),
    //                                     msg_D (sidechain, input 800)
    //
    // With dedup: original keeps msg_A + msg_B → tokens=300; resumed
    // sees msg_A as already-seen and drops it, so it shows
    // msg_C + msg_D → tokens=1200 (NOT 1300, which would be the
    // un-deduped sum).
    let cache = isolated_cache();
    let original_mtime = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    let resumed_mtime = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_500);
    let project = copy_fixture_to_tempdir_with_mtimes(
        &dedup_projects_fixture_dir(),
        &[
            ("dddd0001-0001-0001-0001-000000000001.jsonl", original_mtime),
            ("dddd0002-0002-0002-0002-000000000002.jsonl", resumed_mtime),
        ],
    );

    let stdout_bytes = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(project.path())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).unwrap();

    let original_uuid = "dddd0001-0001-0001-0001-000000000001";
    let resumed_uuid = "dddd0002-0002-0002-0002-000000000002";

    let original_row = stdout
        .lines()
        .find(|l| l.contains(original_uuid))
        .unwrap_or_else(|| panic!("original row missing:\n{stdout}"));
    let resumed_row = stdout
        .lines()
        .find(|l| l.contains(resumed_uuid))
        .unwrap_or_else(|| panic!("resumed row missing:\n{stdout}"));

    // Strip trailing UUID + cost cell to expose the tokens column
    // at the line end.
    let strip_uuid_and_cost = |l: &str, uuid: &str| -> String {
        let after_uuid = l
            .trim_end()
            .strip_suffix(uuid)
            .unwrap_or_else(|| panic!("row missing uuid {uuid}: {l}"))
            .trim_end();
        let dollar_idx = after_uuid
            .rfind('$')
            .unwrap_or_else(|| panic!("expected `$` cost cell:\n{l}"));
        after_uuid[..dollar_idx].trim_end().to_string()
    };
    let original_pre_cost = strip_uuid_and_cost(original_row, original_uuid);
    let resumed_pre_cost = strip_uuid_and_cost(resumed_row, resumed_uuid);
    assert!(
        original_pre_cost.ends_with(" 0.30k"),
        "original tokens should be 300 (renders as 0.30k); got: {original_pre_cost}",
    );
    assert!(
        resumed_pre_cost.ends_with(" 1.20k"),
        "resumed tokens should be 1200 (msg_A replayed turn deduped); \
         got: {resumed_pre_cost}",
    );
    assert!(
        !resumed_pre_cost.ends_with(" 1.30k"),
        "resumed must NOT include msg_A — that is the dedup regression; \
         got: {resumed_pre_cost}",
    );
}

#[test]
fn show_dedups_resumed_session() {
    // `cclens show <resumed_uuid>` must reflect the same dedup as
    // `list`: the rendered table for the resumed session shows only
    // msg_C and the sidechain msg_D, not the replayed msg_A.
    let cache = isolated_cache();
    let original_mtime = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    let resumed_mtime = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_500);
    let project = copy_fixture_to_tempdir_with_mtimes(
        &dedup_projects_fixture_dir(),
        &[
            ("dddd0001-0001-0001-0001-000000000001.jsonl", original_mtime),
            ("dddd0002-0002-0002-0002-000000000002.jsonl", resumed_mtime),
        ],
    );

    let stdout_bytes = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(project.path())
        .arg("show")
        .arg("dddd0002-0002-0002-0002-000000000002")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).unwrap();

    // The replayed assistant text must NOT appear in the resumed
    // session's rendered table — it was filtered out by dedup before
    // the session was rendered.
    assert!(
        !stdout.contains("replayed reply A"),
        "replayed turn should be filtered out by dedup; got:\n{stdout}",
    );

    // The first new turn's text drives the assistant-cluster preview
    // (`assistant_cluster_preview` returns the cluster's first
    // non-empty text block), so its content is visible.
    assert!(
        stdout.contains("reply C (new)"),
        "new turn msg_C should be visible; got:\n{stdout}",
    );

    // The user-row tokens column sums (input + cache_creation) across
    // the assistant cluster. With dedup: msg_C (input 400) + msg_D
    // sidechain (input 800) = 1200. Without dedup, the replayed msg_A
    // would inflate this to 1300 — so 1200 confirms the dedup, AND
    // confirms the sidechain still contributes its tokens (the
    // alternative — a regression that filtered sidechains by role
    // flag — would yield 400, not 1200).
    let user_row = stdout
        .lines()
        .find(|l| l.contains("resumed prompt"))
        .unwrap_or_else(|| panic!("user row missing:\n{stdout}"));
    assert!(
        user_row.contains(" 1.20k "),
        "user row tokens should be 1200 (msg_C 400 + sidechain msg_D 800; renders as 1.20k); \
         got: {user_row}",
    );
    assert!(
        !user_row.contains(" 1.30k "),
        "user row must NOT include msg_A replay tokens; got: {user_row}",
    );
}

#[test]
fn list_includes_subagent_contribution_in_totals() {
    // The `delta` fixture session in -Users-test-beta carries:
    //   parent: 2 assistant turns, billable = 100+200 + 50+100 = 450
    //   subagent 1 (tw-code-reviewer): 20+30 = 50
    //   subagent 2 (tw-code-reviewer): 15+25 = 40
    // Total = 540 billable. Cost (claude-opus-4-7 rates):
    //   parent T1: 100*15e-6 + 200*75e-6 = 0.0165
    //   parent T2: 50*15e-6 + 100*75e-6 = 0.00825
    //   sub1:      20*15e-6 + 30*75e-6  = 0.00255
    //   sub2:      15*15e-6 + 25*75e-6  = 0.0021
    //   total                           = 0.02940 → "$0.0294"
    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).unwrap();

    let delta_uuid = "eeee5555-5555-5555-5555-555555555550";
    let delta_row = stdout
        .lines()
        .find(|l| l.contains(delta_uuid))
        .unwrap_or_else(|| panic!("delta row missing:\n{stdout}"));

    let trimmed = delta_row.trim_end();
    let after_uuid = trimmed
        .strip_suffix(delta_uuid)
        .unwrap_or_else(|| panic!("row missing trailing UUID: {delta_row}"))
        .trim_end();
    let dollar_idx = after_uuid
        .rfind('$')
        .unwrap_or_else(|| panic!("expected `$` cost cell: {delta_row}"));
    let cost_cell = &after_uuid[dollar_idx..];
    let pre_cost = after_uuid[..dollar_idx].trim_end();
    assert!(
        pre_cost.ends_with(" 0.54k"),
        "delta tokens column should include subagent contribution (540 = parent 450 + subagent 90; \
         renders as 0.54k); got: {pre_cost}",
    );
    assert_eq!(cost_cell, "$0.0294", "delta cost cell");
}

#[test]
fn list_skips_unreadable_subagent_jsonl_without_aborting() {
    // Fixture project with one parent session that has a subagent
    // sidecar referencing an unreadable agent jsonl (a directory in
    // its place). The session's row should still appear in `list`,
    // showing the parent's contribution only, without erroring.
    let tmp = tempfile::tempdir().unwrap();
    let project = tmp.path().join("-Users-test-broken");
    std::fs::create_dir(&project).unwrap();
    let session_stem = "ffff6666-6666-6666-6666-666666666666";
    std::fs::write(
        project.join(format!("{session_stem}.jsonl")),
        // Parent: one user turn + one assistant turn (input 100 +
        // output 200 = 300 billable on claude-opus-4-7).
        "{\"type\":\"user\",\"timestamp\":\"2026-04-26T10:00:00Z\",\"cwd\":\"/Users/test/broken\",\"message\":{\"content\":\"hello\"}}\n\
         {\"type\":\"assistant\",\"timestamp\":\"2026-04-26T10:00:05Z\",\"cwd\":\"/Users/test/broken\",\"message\":{\"model\":\"claude-opus-4-7\",\"usage\":{\"input_tokens\":100,\"output_tokens\":200,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}}\n",
    )
    .unwrap();
    let subagents = project.join(session_stem).join("subagents");
    std::fs::create_dir_all(&subagents).unwrap();
    // Sidecar present and parseable; the agent-1.jsonl path is a
    // *directory* rather than a file, so `parse_jsonl` errors → the
    // build_subagent_turns helper returns None → that subagent is
    // skipped.
    std::fs::write(
        subagents.join("agent-1.meta.json"),
        r#"{"agentType":"tw-code-reviewer","description":"broken case"}"#,
    )
    .unwrap();
    std::fs::create_dir(subagents.join("agent-1.jsonl")).unwrap();

    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(tmp.path())
        .arg("list")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).unwrap();
    let row = stdout
        .lines()
        .find(|l| l.contains(session_stem))
        .unwrap_or_else(|| panic!("session row missing:\n{stdout}"));
    // Parent-only billable = 300 (subagent skipped due to unreadable
    // jsonl). Cost = 100*15e-6 + 200*75e-6 = 0.0165 → "$0.0165".
    let after_uuid = row
        .trim_end()
        .strip_suffix(session_stem)
        .unwrap()
        .trim_end();
    let dollar_idx = after_uuid.rfind('$').unwrap();
    let cost_cell = &after_uuid[dollar_idx..];
    let pre_cost = after_uuid[..dollar_idx].trim_end();
    assert!(
        pre_cost.ends_with(" 0.30k"),
        "tokens should be parent-only 300 (subagent skipped; renders as 0.30k); got: {pre_cost}",
    );
    assert_eq!(cost_cell, "$0.0165", "cost cell");
}

#[test]
fn show_interleaves_subagent_rows_by_timestamp() {
    // Run `cclens show` on the delta fixture (parent with two
    // tw-code-reviewer subagent invocations interleaved). Verify
    // that subagent rows appear inline with parent rows, sorted by
    // timestamp, and that the cumulative-at-bottom equals the same
    // session's `cclens list` total (540).
    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg("eeee5555-5555-5555-5555-555555555550")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();

    let user_line = lines
        .iter()
        .position(|l| l.contains("Review auth and session changes"))
        .expect("parent user row missing");
    let assistant_line = lines
        .iter()
        .position(|l| l.contains("Dispatching reviewers"))
        .expect("parent assistant row missing");
    let sub1_line = lines
        .iter()
        .position(|l| l.contains("Review auth changes"))
        .expect("first subagent row missing");
    let sub2_line = lines
        .iter()
        .position(|l| l.contains("Review session-handler refactor"))
        .expect("second subagent row missing");

    // The parent's exchange spans both Task tool_uses (the second
    // user turn is tool_result-only, so it's absorbed); subagent
    // exchanges interleave by their own user-turn timestamp.
    assert!(
        user_line < assistant_line,
        "parent user row precedes assistant"
    );
    assert!(
        assistant_line < sub1_line,
        "first subagent row should follow parent assistant",
    );
    assert!(
        sub1_line < sub2_line,
        "subagent rows should be in timestamp order",
    );

    // Cumulative-at-bottom equals the list total of 540 (renders as 0.54k).
    let last_sub_line = lines[sub2_line];
    assert!(
        last_sub_line.contains(" 0.54k "),
        "cumulative-at-bottom should equal list total 540 (renders as 0.54k); got: {last_sub_line}",
    );
}

#[test]
fn show_disambiguates_same_type_subagent_invocations_by_description() {
    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg("eeee5555-5555-5555-5555-555555555550")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).unwrap();

    // Both subagent rows render with their own description visible
    // in the content cell, so the user can tell two same-agent_type
    // invocations apart.
    assert!(
        stdout.contains("(tw-code-reviewer · \"Review auth changes\")"),
        "first subagent row must carry its description; got:\n{stdout}",
    );
    assert!(
        stdout.contains("(tw-code-reviewer · \"Review session-handler refactor\")"),
        "second subagent row must carry its description; got:\n{stdout}",
    );
}

#[test]
fn show_works_on_zero_billable_session() {
    let cache = isolated_cache();
    let stdout_bytes = cclens_command(cache.path(), &pricing_fixture_url("litellm-mini.json"))
        .args(["--projects-dir"])
        .arg(projects_fixture_dir())
        .arg("show")
        .arg("cccc3333-3333-3333-3333-333333333333")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout_bytes).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();

    let header = lines
        .iter()
        .find(|l| l.contains("datetime"))
        .expect("header row missing");
    for col in [
        "role",
        "tokens",
        "cost",
        "cumulative",
        "cum_cost",
        "content",
    ] {
        assert!(header.contains(col), "header missing {col}: {header}");
    }

    let data_rows: Vec<&&str> = lines
        .iter()
        .filter(|l| l.contains("hello but nothing replies"))
        .collect();
    assert_eq!(
        data_rows.len(),
        1,
        "expected exactly one data row; got {}:\n{stdout}",
        data_rows.len(),
    );
    let row = data_rows[0];

    // Orphan user row: tokens=`—`, cost=`—` (display), but the
    // accumulator delta is Some(0.0) so cum_cost is `$0.0000` (not
    // `—`). cumulative is 0 since no assistant turn contributed.
    assert!(
        row.contains('—'),
        "orphan row should show em-dash in tokens/cost; got: {row}",
    );
    assert!(
        row.contains("$0.0000"),
        "orphan row cum_cost should be $0.0000; got: {row}",
    );
    // Strip content marker, then trailing `$0.0000` (cum_cost), then
    // verify the cumulative cell shows zero (renders as `0.00k`).
    let trimmed = row.trim_end();
    let idx = trimmed
        .rfind("hello but nothing replies")
        .expect("row missing content marker");
    let after_content = trimmed[..idx].trim_end();
    let after_cum_cost = after_content
        .strip_suffix("$0.0000")
        .expect("cum_cost should be $0.0000")
        .trim_end();
    assert!(
        after_cum_cost.ends_with("0.00k"),
        "orphan row cumulative should be 0 (renders as 0.00k); got: {row}",
    );
}
