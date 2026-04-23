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

    // Right-aligned tokens end the row; anchor token assertions to the trimmed
    // line end so timestamp digits can't spuriously match token digits.
    let alpha = lines
        .iter()
        .find(|l| l.contains("/test-cmd hello world"))
        .expect("alpha row missing");
    assert!(alpha.contains("alpha"));
    assert!(alpha.trim_end().ends_with(" 650"), "alpha row: {alpha}");

    let beta_prose = lines
        .iter()
        .find(|l| l.contains("How do I configure neovim folds?"))
        .expect("beta-prose row missing");
    assert!(beta_prose.contains("beta"));
    assert!(
        beta_prose.trim_end().ends_with(" 100"),
        "beta-prose row: {beta_prose}",
    );

    let beta_late = lines
        .iter()
        .find(|l| l.contains("after malformed"))
        .expect("after-malformed row missing");
    assert!(beta_late.contains("beta"));
    assert!(
        beta_late.trim_end().ends_with(" 10"),
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
