use std::path::PathBuf;

use assert_cmd::Command;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/projects")
}

#[test]
fn list_renders_sessions_oldest_first_with_correct_totals() {
    let out = Command::cargo_bin("cclens")
        .unwrap()
        .args(["--projects-dir"])
        .arg(fixtures_dir())
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
    assert!(header.contains("project"));
    assert!(header.contains("title"));
    assert!(header.contains("tokens"));
    assert!(header.contains("id"));

    // gamma has no assistant turns, so the zero-billable filter drops it.
    // Scope the check to non-header lines so we assert "no data row references
    // gamma" specifically (the header doesn't mention project names).
    assert!(
        lines
            .iter()
            .filter(|l| !l.contains("datetime"))
            .all(|l| !l.contains("gamma")),
        "expected no row to reference project 'gamma' (zero-billable filter); stdout was:\n{stdout}",
    );

    // The `id` column is rightmost; right-aligned tokens sit just before it.
    // Strip the trailing UUID to put tokens back at the line end so token
    // assertions can anchor on the line end (avoiding spurious matches on
    // timestamp digits).
    let strip_trailing_uuid = |l: &str, uuid: &str| {
        l.trim_end()
            .strip_suffix(uuid)
            .unwrap_or_else(|| panic!("row missing expected trailing UUID {uuid}:\n{l}"))
            .trim_end()
            .to_string()
    };

    let alpha_uuid = "aaaa1111-1111-1111-1111-111111111111";
    let alpha = lines
        .iter()
        .find(|l| l.contains("/test-cmd hello world"))
        .expect("alpha row missing");
    assert!(alpha.contains("alpha"));
    assert!(alpha.contains(alpha_uuid), "alpha row missing id: {alpha}");
    assert!(
        strip_trailing_uuid(alpha, alpha_uuid).ends_with(" 650"),
        "alpha row: {alpha}",
    );

    let beta_prose_uuid = "bbbb2222-2222-2222-2222-222222222222";
    let beta_prose = lines
        .iter()
        .find(|l| l.contains("How do I configure neovim folds?"))
        .expect("beta-prose row missing");
    assert!(beta_prose.contains("beta"));
    assert!(
        beta_prose.contains(beta_prose_uuid),
        "beta-prose row missing id: {beta_prose}",
    );
    assert!(
        strip_trailing_uuid(beta_prose, beta_prose_uuid).ends_with(" 100"),
        "beta-prose row: {beta_prose}",
    );

    let beta_late_uuid = "dddd4444-4444-4444-4444-444444444444";
    let beta_late = lines
        .iter()
        .find(|l| l.contains("after malformed"))
        .expect("after-malformed row missing");
    assert!(beta_late.contains("beta"));
    assert!(
        beta_late.contains(beta_late_uuid),
        "beta-late row missing id: {beta_late}",
    );
    assert!(
        strip_trailing_uuid(beta_late, beta_late_uuid).ends_with(" 10"),
        "beta-late row: {beta_late}",
    );

    // Ordering: alpha (2026-04-01) appears before beta-prose (2026-04-15)
    // which appears before beta-after-malformed (2026-04-20). Use full-title
    // markers so a future fixture containing a bare `/test-cmd` doesn't steal
    // the alpha lookup.
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
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::cargo_bin("cclens")
        .unwrap()
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
    // Guards against a future regression that emits a spurious warning
    // or placeholder row on empty input.
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
    let out = Command::cargo_bin("cclens")
        .unwrap()
        .args(["--projects-dir"])
        .arg(fixtures_dir())
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
    for col in ["role", "tokens", "cumulative", "content"] {
        assert!(header.contains(col), "header missing {col}: {header}");
    }

    // Strip the trailing content-column marker, then trim padding, leaving the
    // cumulative column at the right edge where its value can be anchored.
    // Mirrors the `strip_trailing_uuid` discipline in the list test above.
    let strip_trailing_marker = |line: &str, marker: &str| -> String {
        let trimmed = line.trim_end();
        let idx = trimmed
            .rfind(marker)
            .unwrap_or_else(|| panic!("row missing marker {marker:?}:\n{line}"));
        trimmed[..idx].trim_end().to_string()
    };

    // (content-marker, tokens-column match, cumulative-column value)
    // Tokens anchors use ` <N> ` (surrounding spaces) — small integers like
    // `60` or `120` would otherwise collide with timestamp fragments. The
    // orphan row pins on the em-dash directly.
    //
    // ` answer ` is space-padded because `answer` alone would also match
    // inside `"third question with no response"`'s `"answer"`-adjacent
    // context on some future fixture — the padded form restricts the match
    // to whole-word occurrences in the rendered table.
    let expected = [
        ("/test-cmd demo", " 315 ", "315"),
        ("reading the file +2 tool uses", " 280 ", "595"),
        ("follow-up question", " 120 ", "715"),
        (" answer ", " 60 ", "775"),
        ("third question with no response", "—", "775"),
    ];
    let mut positions: Vec<usize> = Vec::with_capacity(expected.len());
    for (marker, tokens, cumulative) in expected {
        let idx = lines
            .iter()
            .position(|l| l.contains(marker))
            .unwrap_or_else(|| panic!("row {marker:?} missing:\n{stdout}"));
        let line = lines[idx];
        assert!(
            line.contains(tokens),
            "row {marker:?} missing tokens {tokens:?}: {line}",
        );
        let cols = strip_trailing_marker(line, marker.trim());
        assert!(
            cols.ends_with(cumulative),
            "row {marker:?} cumulative should be {cumulative}; got: {line}",
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
    let unknown = "00000000-0000-0000-0000-000000000000";
    let output = Command::cargo_bin("cclens")
        .unwrap()
        .args(["--projects-dir"])
        .arg(fixtures_dir())
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
fn show_works_on_zero_billable_session() {
    let stdout_bytes = Command::cargo_bin("cclens")
        .unwrap()
        .args(["--projects-dir"])
        .arg(fixtures_dir())
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
    for col in ["role", "tokens", "cumulative", "content"] {
        assert!(
            header.contains(col),
            "header should contain column {col}; got: {header}",
        );
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
    // Strip the content-column marker so the cumulative column sits at the
    // right edge, then pin the trailing value.
    let trimmed = row.trim_end();
    let idx = trimmed
        .rfind("hello but nothing replies")
        .expect("row missing content marker");
    let cols = trimmed[..idx].trim_end();
    assert!(
        row.contains('—'),
        "orphan row should show em-dash in tokens; got: {row}",
    );
    assert!(
        cols.ends_with('0'),
        "orphan row cumulative should be 0; got: {row}",
    );
}
