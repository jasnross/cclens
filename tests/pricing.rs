// `allow-expect-in-tests` / `allow-unwrap-in-tests` in clippy.toml cover
// code inside `#[test]` functions only. This file has module-level
// helpers (`isolated_cache`, `run_cclens_with_env`) that use `unwrap`
// and `expect`, so the file-wide allow is the minimum escape hatch —
// per-item allows would also work but scatter the intent.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;

use assert_cmd::Command;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pricing")
        .join(name)
}

fn fixture_url(name: &str) -> String {
    format!("file://{}", fixture_path(name).display())
}

fn isolated_cache() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn run_cclens_with_env(args: &[&str], pricing_url: &str, cache_dir: &std::path::Path) -> Vec<u8> {
    Command::cargo_bin("cclens")
        .unwrap()
        .env("CCLENS_PRICING_URL", pricing_url)
        .env("CCLENS_CACHE_DIR", cache_dir)
        .args(args)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone()
}

/// Run `cclens` expecting a non-zero exit and return its combined
/// stderr. Mirrors `run_cclens_with_env` but for failure paths.
fn run_cclens_expect_failure(
    args: &[&str],
    pricing_url: &str,
    cache_dir: &std::path::Path,
) -> Vec<u8> {
    Command::cargo_bin("cclens")
        .unwrap()
        .env("CCLENS_PRICING_URL", pricing_url)
        .env("CCLENS_CACHE_DIR", cache_dir)
        .args(args)
        .assert()
        .failure()
        .get_output()
        .stderr
        .clone()
}

#[test]
fn pricing_refresh_writes_cache() {
    let cache = isolated_cache();
    let stdout = run_cclens_with_env(
        &["pricing", "refresh"],
        &fixture_url("litellm-mini.json"),
        cache.path(),
    );
    let out = String::from_utf8(stdout).unwrap();
    assert!(
        out.contains("Claude entries: 3"),
        "expected 3 Claude entries in output; got:\n{out}",
    );
    let cache_file = cache.path().join("litellm-pricing.json");
    assert!(cache_file.exists(), "cache file should exist after refresh");
    let body = std::fs::read_to_string(&cache_file).unwrap();
    // Cache stores the raw fixture content verbatim.
    assert!(body.contains("claude-opus-4-7"));
    assert!(
        body.contains("gpt-4"),
        "cache must store raw catalog, not filtered"
    );
}

#[test]
fn pricing_info_reports_after_refresh() {
    let cache = isolated_cache();
    // First refresh, then info.
    let _ = run_cclens_with_env(
        &["pricing", "refresh"],
        &fixture_url("litellm-mini.json"),
        cache.path(),
    );
    let stdout = run_cclens_with_env(
        &["pricing", "info"],
        &fixture_url("litellm-mini.json"),
        cache.path(),
    );
    let out = String::from_utf8(stdout).unwrap();
    let cache_file = cache.path().join("litellm-pricing.json");
    assert!(
        out.contains(&cache_file.display().to_string()),
        "info should show tempdir cache path; got:\n{out}",
    );
    assert!(out.contains("exists: true"), "got:\n{out}");
    assert!(out.contains("Claude entries: 3"), "got:\n{out}");
    // Size must be greater than zero when the file exists.
    assert!(
        !out.contains("size: 0 bytes"),
        "expected non-zero size; got:\n{out}",
    );
}

#[test]
fn pricing_info_when_cache_missing() {
    let cache = isolated_cache();
    let stdout = run_cclens_with_env(
        &["pricing", "info"],
        &fixture_url("litellm-mini.json"),
        cache.path(),
    );
    let out = String::from_utf8(stdout).unwrap();
    assert!(out.contains("exists: false"), "got:\n{out}");
    assert!(out.contains("size: 0 bytes"), "got:\n{out}");
    // When the file doesn't exist we can't parse it for an entry count,
    // so the "entry_count is None" branch fires -> `(unreadable)`.
    assert!(out.contains("Claude entries: (unreadable)"), "got:\n{out}");
}

#[test]
fn pricing_refresh_reports_previous_size() {
    let cache = isolated_cache();
    // First refresh with empty fixture so the subsequent refresh sees a
    // non-zero previous size.
    let _ = run_cclens_with_env(
        &["pricing", "refresh"],
        &fixture_url("litellm-empty.json"),
        cache.path(),
    );
    let stdout = run_cclens_with_env(
        &["pricing", "refresh"],
        &fixture_url("litellm-mini.json"),
        cache.path(),
    );
    let out = String::from_utf8(stdout).unwrap();
    // Previous size must be present and non-zero.
    let prev_line = out
        .lines()
        .find(|l| l.contains("previous size"))
        .expect("previous size line missing");
    assert!(
        !prev_line.contains("previous size: 0 bytes"),
        "expected non-zero previous size on second refresh; got: {prev_line}",
    );
}

#[test]
fn pricing_refresh_failure_preserves_previous_cache() {
    // Establish a known-good cache.
    let cache = isolated_cache();
    let _ = run_cclens_with_env(
        &["pricing", "refresh"],
        &fixture_url("litellm-mini.json"),
        cache.path(),
    );
    let cache_file = cache.path().join("litellm-pricing.json");
    let before = std::fs::read(&cache_file).expect("good cache should exist");

    // Attempt a refresh against a missing fixture — must exit non-zero
    // and leave the previously-good cache untouched.
    let stderr = run_cclens_expect_failure(
        &["pricing", "refresh"],
        "file:///tmp/cclens-nonexistent-fixture-that-must-not-exist.json",
        cache.path(),
    );
    let stderr_s = String::from_utf8(stderr).unwrap();
    assert!(
        stderr_s.contains("fetch failed"),
        "stderr should mention fetch failure; got:\n{stderr_s}",
    );

    let after = std::fs::read(&cache_file).expect("cache should still exist");
    assert_eq!(
        before, after,
        "failed refresh must not overwrite the previous cache",
    );
}

#[test]
fn pricing_info_on_corrupt_cache() {
    let cache = isolated_cache();
    let cache_file = cache.path().join("litellm-pricing.json");
    // Write garbage JSON directly to the cache location.
    std::fs::write(&cache_file, r#"{"broken":"#).unwrap();

    let stdout = run_cclens_with_env(
        &["pricing", "info"],
        &fixture_url("litellm-mini.json"),
        cache.path(),
    );
    let out = String::from_utf8(stdout).unwrap();
    assert!(out.contains("exists: true"), "got:\n{out}");
    assert!(
        out.contains("Claude entries: (unreadable)"),
        "corrupt cache should produce (unreadable); got:\n{out}",
    );
    // Size is the byte count of the corrupt payload — anything non-zero
    // is acceptable (confirms the file was stat-ed).
    assert!(
        !out.contains("size: 0 bytes"),
        "expected non-zero size; got:\n{out}",
    );
}
