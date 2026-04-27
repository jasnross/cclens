//! Walks user-controlled context-file locations and produces a
//! `Vec<ContextFile>` for the `inputs` subcommand.
//!
//! Public API:
//! - `discover_inventory(&InventoryConfig) -> Vec<ContextFile>` — walks
//!   the user-global locations (`~/.claude/CLAUDE.md`, `rules/`,
//!   `skills/`, `agents/`) and the plugin cache. Project-local + ancestor
//!   `CLAUDE.md` walks are per-session and live in `walk_for_session`.
//! - `walk_for_session(&Path, &InventoryConfig) -> Vec<ContextFile>` —
//!   per-session walk: ancestor `CLAUDE.md` chain + `<cwd>/.claude/`
//!   subtree. Called by `attribution::extend_inventory_for_session`.
//! - `ContextFile` / `ContextFileKind` / `CacheTier` / `Scope` —
//!   typed records the walker emits.
//! - `InventoryConfig` — overridable filesystem roots; production
//!   callers use `Default::default()`, tests construct directly.
//!
//! Per-entry graceful degradation: an unreadable rule file, an
//! unparseable `installed_plugins.json`, or a tokenizer failure on one
//! file never aborts the walk. Same `let Ok(...) = ... else { continue;
//! }` pattern used in `discovery` and `parsing`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Once;

use serde::Deserialize;

// ---- core types ----

/// Categorical kind of a discovered context file. Drives the
/// rendering label and the per-load attribution dispatch in
/// `attribution::compute_rows`.
///
/// Variants with `plugin` / `marketplace` fields carry the
/// `<plugin>@<marketplace>` parts parsed out of the
/// `installed_plugins.json` key so the renderer can show the plugin
/// each file came from.
///
/// `Clone` is `cfg(test)`-only on purpose: production code never needs
/// to clone these — the walker emits each `ContextFile` once and
/// `attribution::compute_rows` consumes them by value. Tests that
/// need to re-run a fixture loop get `.clone()` for free without
/// production sites silently allocating.
#[cfg_attr(test, derive(Clone))]
#[derive(Debug)]
pub enum ContextFileKind {
    GlobalClaudeMd,
    UserRule,
    UserSkill,
    UserAgent,
    PluginSkill {
        plugin: String,
        marketplace: String,
    },
    PluginRule {
        plugin: String,
        marketplace: String,
    },
    PluginAgent {
        plugin: String,
        marketplace: String,
    },
    /// Any `CLAUDE.md` found by the cwd-ancestor walk in
    /// `walk_for_session`.
    ProjectClaudeMd,
    /// `<cwd>/.claude/skills/<name>/SKILL.md`.
    ProjectLocalSkill,
    /// `<cwd>/.claude/commands/<name>.md`.
    ProjectLocalCommand,
    ProjectLocalRule,
    ProjectLocalAgent,
}

/// Which prompt-cache tier a `cache_creation` event was billed against.
///
/// Tier is a property of the load *event*, not of the file body —
/// `attribution::SessionMeta::primary_tier` and
/// `OnDemandLoad::tier` carry the tier per session / per load. A
/// single file (e.g. CLAUDE.md) can attribute at 1h in one session
/// and 5m in another (parent session uses 1h, subagent uses 5m).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    /// System-prompt region tier — billed at the 1h cache-creation
    /// rate. Always-loaded content (`CLAUDE.md`, rules) and matched
    /// agent files attribute here in parent sessions.
    Long1h,
    /// On-demand / in-conversation cache point tier — billed at the
    /// 5m cache-creation rate. Skill / command bodies attribute here
    /// (per the harness's observed convention), as do always-loaded
    /// files in subagent sessions whose `primary_tier` is 5m.
    Short5m,
}

/// In-scope predicate for a `ContextFile`. The walker produces this
/// alongside each file; `attribution` uses it to filter the inventory
/// per session.
///
/// `Clone` is `cfg(test)`-only — see `ContextFileKind` for the
/// production-allocation-visibility rationale. Walkers inside this
/// module use the local `clone_scope` helper to make per-row
/// duplications visible.
#[cfg_attr(test, derive(Clone))]
#[derive(Debug)]
pub enum Scope {
    /// In scope for every session.
    Global,
    /// In scope when `session_cwd.starts_with(root)`.
    CwdSubtree { root: PathBuf },
}

impl Scope {
    /// Whether this scope applies to a session running with `session_cwd`.
    #[must_use]
    pub fn matches(&self, session_cwd: &Path) -> bool {
        match self {
            Self::Global => true,
            Self::CwdSubtree { root } => session_cwd.starts_with(root),
        }
    }
}

/// One discovered context file with everything `attribution` needs to
/// price it.
///
/// `tier` is intentionally not a field — under the per-load attribution
/// model the cache tier comes from session evidence (the
/// `cache_creation` event the load triggered), not from a property of
/// the file itself. `attribution::compute_rows` reads the tier off
/// each `SessionMeta`'s `primary_tier` / `OnDemandLoad.tier` rather
/// than off `ContextFileKind`.
///
/// `Clone` is `cfg(test)`-only — production code consumes
/// `ContextFile` exactly once via `compute_rows`.
#[cfg_attr(test, derive(Clone))]
#[derive(Debug)]
pub struct ContextFile {
    pub path: PathBuf,
    pub kind: ContextFileKind,
    pub tokens: u64,
    pub scope: Scope,
}

impl ContextFile {
    /// The harness-name a session's `Turn.content` evidence will refer
    /// to this file by, or `None` for always-loaded kinds (CLAUDE.md
    /// and rules) where evidence isn't keyed by name.
    ///
    /// Mapping:
    /// - Agent kinds (`UserAgent`, `PluginAgent`, `ProjectLocalAgent`):
    ///   the file stem (`tw-code-reviewer.md` → `tw-code-reviewer`),
    ///   matched against `SessionKind::Subagent { agent_type }`.
    /// - Skill kinds (`UserSkill`, `PluginSkill`,
    ///   `ProjectLocalSkill`): the parent directory's name (skills are
    ///   subdir-keyed at `<root>/skills/<name>/SKILL.md`), matched
    ///   against the path that follows the `Base directory for this
    ///   skill: ` prefix in user-turn content.
    /// - `ProjectLocalCommand`: the file stem
    ///   (`commands/foo.md` → `foo`), matched against the
    ///   `<command-name>/foo</command-name>` tag in user-turn content.
    /// - Always-loaded kinds (`GlobalClaudeMd`, `ProjectClaudeMd`,
    ///   `UserRule`, `PluginRule`, `ProjectLocalRule`): `None`. These
    ///   files load once per session (parent + each subagent that
    ///   inherits scope) regardless of any harness identifier.
    #[must_use]
    pub fn identifier(&self) -> Option<String> {
        match &self.kind {
            ContextFileKind::UserAgent
            | ContextFileKind::PluginAgent { .. }
            | ContextFileKind::ProjectLocalAgent
            | ContextFileKind::ProjectLocalCommand => self
                .path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned()),
            ContextFileKind::UserSkill
            | ContextFileKind::PluginSkill { .. }
            | ContextFileKind::ProjectLocalSkill => self
                .path
                .parent()
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().into_owned()),
            ContextFileKind::GlobalClaudeMd
            | ContextFileKind::ProjectClaudeMd
            | ContextFileKind::UserRule
            | ContextFileKind::PluginRule { .. }
            | ContextFileKind::ProjectLocalRule => None,
        }
    }
}

/// Overridable filesystem roots for the walker. In production, callers
/// construct via `Default::default()`; tests override individual fields.
///
/// The two path fields are independent — overriding `claude_home` does
/// **not** retroactively re-derive `installed_plugins_path`. A test
/// that wants both must set both.
#[derive(Debug)]
pub struct InventoryConfig {
    /// Root of the user's `~/.claude/` tree. The walker reads
    /// `<claude_home>/{CLAUDE.md,rules,skills,agents}` and uses this
    /// path as the "stop here" boundary for the per-session ancestor
    /// `CLAUDE.md` walk.
    pub claude_home: PathBuf,
    /// Absolute path to `installed_plugins.json`. Plugin content is
    /// driven entirely from this file's `installPath` entries — the
    /// walker does not scan the plugin cache directory.
    pub installed_plugins_path: PathBuf,
}

const CCLENS_CLAUDE_HOME_ENV: &str = "CCLENS_CLAUDE_HOME";

impl Default for InventoryConfig {
    fn default() -> Self {
        // Mirrors `pricing::resolve_cache_file`'s "empty means unset"
        // rule: `CCLENS_CLAUDE_HOME=""` falls through to the production
        // default rather than being interpreted as "use CWD".
        let claude_home = match std::env::var(CCLENS_CLAUDE_HOME_ENV) {
            Ok(v) if !v.is_empty() => PathBuf::from(v),
            _ => dirs::home_dir()
                .map(|h| h.join(".claude"))
                .unwrap_or_default(),
        };
        let installed_plugins_path = claude_home.join("plugins/installed_plugins.json");
        Self {
            claude_home,
            installed_plugins_path,
        }
    }
}

// ---- raw schema (installed_plugins.json) ----

/// Top-level shape of `~/.claude/plugins/installed_plugins.json`.
///
/// Field names mirror the on-disk schema verbatim per cclens's
/// "let serde do the work" principle. Unknown fields (`version` etc.)
/// are silently ignored by serde's default behavior.
#[derive(Deserialize)]
struct RawInstalledPlugins {
    #[serde(default)]
    plugins: std::collections::HashMap<String, Vec<RawPluginInstance>>,
}

/// One element of the per-plugin install array. The harness appends to
/// this array on update; we use `entries.last()` per the plan.
#[derive(Deserialize)]
struct RawPluginInstance {
    #[serde(rename = "installPath")]
    install_path: String,
}

// ---- tokenizer ----

/// Tokenize a file's contents and return the count, or `None` on any
/// failure (file unreadable, non-UTF-8 content, tokenizer error).
///
/// Reads via `fs::read_to_string` which transparently follows symlinks
/// — required for the user's symlinked global `CLAUDE.md`. Uses
/// `tiktoken_rs::cl100k_base_singleton()` so the BPE asset loads at
/// most once per process.
fn tokenize_file_count(path: &Path) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    let bpe = tiktoken_rs::cl100k_base_singleton();
    let count = bpe.encode_ordinary(&text).len();
    u64::try_from(count).ok()
}

// ---- walkers ----

/// Walk the user-global locations under `claude_home`.
fn walk_global(config: &InventoryConfig) -> Vec<ContextFile> {
    let mut out = Vec::new();

    // Direct lookup: <claude_home>/CLAUDE.md (may be a symlink).
    let global_claude_md = config.claude_home.join("CLAUDE.md");
    if let Some(tokens) = tokenize_file_count(&global_claude_md) {
        out.push(ContextFile {
            path: global_claude_md,
            kind: ContextFileKind::GlobalClaudeMd,
            tokens,
            scope: Scope::Global,
        });
    }

    // <claude_home>/rules/*.md
    collect_md_files_in_dir(
        &config.claude_home.join("rules"),
        &mut out,
        || ContextFileKind::UserRule,
        &Scope::Global,
    );

    // <claude_home>/skills/<name>/SKILL.md
    collect_skill_md_files_in_dir(
        &config.claude_home.join("skills"),
        &mut out,
        || ContextFileKind::UserSkill,
        || Scope::Global,
    );

    // <claude_home>/agents/*.md
    collect_md_files_in_dir(
        &config.claude_home.join("agents"),
        &mut out,
        || ContextFileKind::UserAgent,
        &Scope::Global,
    );

    out
}

/// Walk every `*.md` file in `dir` (one level deep) and append a
/// `ContextFile` per readable + tokenizable entry. `kind_for` returns
/// the kind for each match.
fn collect_md_files_in_dir(
    dir: &Path,
    out: &mut Vec<ContextFile>,
    kind_for: impl Fn() -> ContextFileKind,
    scope_template: &Scope,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        let Some(tokens) = tokenize_file_count(&path) else {
            continue;
        };
        out.push(ContextFile {
            path,
            kind: kind_for(),
            tokens,
            scope: clone_scope(scope_template),
        });
    }
}

/// Walk every `<dir>/<name>/SKILL.md` (skills are subdir-keyed unlike
/// rules/agents which are file-keyed).
fn collect_skill_md_files_in_dir(
    dir: &Path,
    out: &mut Vec<ContextFile>,
    kind_for: impl Fn() -> ContextFileKind,
    scope_for: impl Fn() -> Scope,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let skill_dir = entry.path();
        if !skill_dir.is_dir() {
            continue;
        }
        let skill_md = skill_dir.join("SKILL.md");
        let Some(tokens) = tokenize_file_count(&skill_md) else {
            continue;
        };
        out.push(ContextFile {
            path: skill_md,
            kind: kind_for(),
            tokens,
            scope: scope_for(),
        });
    }
}

/// `Scope` deliberately does not derive `Clone`: the only field
/// `CwdSubtree` carries is a `PathBuf`, and a derived `.clone()` at
/// downstream call sites would silently duplicate path allocations
/// per row. Confining the duplication to this one in-walker helper
/// keeps those allocations visible.
fn clone_scope(scope: &Scope) -> Scope {
    match scope {
        Scope::Global => Scope::Global,
        Scope::CwdSubtree { root } => Scope::CwdSubtree { root: root.clone() },
    }
}

/// Walk the plugin cache. Reads `installed_plugins.json` for the set of
/// installed plugins, then walks each plugin's `installPath` for
/// `skills/`, `rules/`, `agents/`.
///
/// Missing or unparseable JSON returns an empty inventory plus one
/// stderr warning per process (same `Once`-gated pattern as
/// `pricing::warn_once`).
fn walk_plugins(config: &InventoryConfig) -> Vec<ContextFile> {
    let Ok(body) = fs::read_to_string(&config.installed_plugins_path) else {
        // Missing file is the common case for users with no plugins
        // installed — silently empty inventory, no warning.
        return Vec::new();
    };
    let parsed: RawInstalledPlugins = match serde_json::from_str(&body) {
        Ok(p) => p,
        Err(e) => {
            warn_once(&format!(
                "could not parse {}: {e}; plugin context files will be omitted",
                config.installed_plugins_path.display(),
            ));
            return Vec::new();
        }
    };

    let mut out = Vec::new();
    for (key, entries) in parsed.plugins {
        let Some((plugin, marketplace)) = parse_plugin_key(&key) else {
            warn_once(&format!(
                "skipping plugin entry with unrecognized key {key:?} (expected `<plugin>@<marketplace>`)",
            ));
            continue;
        };
        let Some(latest) = entries.last() else {
            // Empty array — nothing to walk.
            continue;
        };
        let install_path = PathBuf::from(&latest.install_path);

        let plugin_owned = plugin.to_string();
        let market_owned = marketplace.to_string();
        // Skills: <installPath>/skills/<name>/SKILL.md
        collect_skill_md_files_in_dir(
            &install_path.join("skills"),
            &mut out,
            || ContextFileKind::PluginSkill {
                plugin: plugin_owned.clone(),
                marketplace: market_owned.clone(),
            },
            || Scope::Global,
        );
        // Rules: <installPath>/rules/*.md
        let plugin_for_rules = plugin_owned.clone();
        let market_for_rules = market_owned.clone();
        collect_md_files_in_dir(
            &install_path.join("rules"),
            &mut out,
            move || ContextFileKind::PluginRule {
                plugin: plugin_for_rules.clone(),
                marketplace: market_for_rules.clone(),
            },
            &Scope::Global,
        );
        // Agents: <installPath>/agents/*.md
        let plugin_for_agents = plugin_owned;
        let market_for_agents = market_owned;
        collect_md_files_in_dir(
            &install_path.join("agents"),
            &mut out,
            move || ContextFileKind::PluginAgent {
                plugin: plugin_for_agents.clone(),
                marketplace: market_for_agents.clone(),
            },
            &Scope::Global,
        );
    }
    out
}

/// Split `<plugin>@<marketplace>` into its two parts. Returns `None`
/// when the key has no `@` separator.
fn parse_plugin_key(key: &str) -> Option<(&str, &str)> {
    let (plugin, marketplace) = key.split_once('@')?;
    if plugin.is_empty() || marketplace.is_empty() {
        return None;
    }
    Some((plugin, marketplace))
}

/// Walk per-session locations rooted at `session_cwd`.
///
/// Two phases:
/// 1. Ancestor `CLAUDE.md` walk: from `session_cwd` upward, stopping
///    *before* `claude_home` and the filesystem root.
/// 2. `<session_cwd>/.claude/{skills,commands,rules,agents}/`.
///
/// `session_cwd` is the session's recorded `Turn.cwd` (the directory
/// the user was working in when the session ran), **not** the cwd
/// where `cclens` itself was invoked. Public so
/// `attribution::extend_inventory_for_session` can call it across
/// module boundaries.
#[must_use]
pub fn walk_for_session(session_cwd: &Path, config: &InventoryConfig) -> Vec<ContextFile> {
    let mut out = Vec::new();
    walk_ancestor_claude_mds(session_cwd, config, &mut out);
    walk_project_local_subtree(session_cwd, &mut out);
    out
}

fn walk_ancestor_claude_mds(
    session_cwd: &Path,
    config: &InventoryConfig,
    out: &mut Vec<ContextFile>,
) {
    let mut current: &Path = session_cwd;
    loop {
        // Stop at the filesystem root: `parent()` returns `None` for
        // `/`, which means we never check `/CLAUDE.md`.
        if current.parent().is_none() {
            break;
        }
        // Stop at (without inspecting) `~/.claude` so a session whose
        // cwd happens to be under `~/` doesn't pull in the global
        // CLAUDE.md as a "project" CLAUDE.md.
        if current == config.claude_home {
            break;
        }
        let claude_md = current.join("CLAUDE.md");
        if let Some(tokens) = tokenize_file_count(&claude_md) {
            out.push(ContextFile {
                path: claude_md,
                kind: ContextFileKind::ProjectClaudeMd,
                tokens,
                scope: Scope::CwdSubtree {
                    root: current.to_path_buf(),
                },
            });
        }
        let Some(parent) = current.parent() else {
            break;
        };
        current = parent;
    }
}

fn walk_project_local_subtree(session_cwd: &Path, out: &mut Vec<ContextFile>) {
    let claude_local = session_cwd.join(".claude");
    if !claude_local.is_dir() {
        return;
    }
    let cwd_root = session_cwd.to_path_buf();

    // Skills: <cwd>/.claude/skills/<name>/SKILL.md
    let cwd_for_skills = cwd_root.clone();
    collect_skill_md_files_in_dir(
        &claude_local.join("skills"),
        out,
        || ContextFileKind::ProjectLocalSkill,
        || Scope::CwdSubtree {
            root: cwd_for_skills.clone(),
        },
    );

    // Commands: <cwd>/.claude/commands/*.md
    let cwd_for_commands = cwd_root.clone();
    collect_md_files_in_dir(
        &claude_local.join("commands"),
        out,
        || ContextFileKind::ProjectLocalCommand,
        &Scope::CwdSubtree {
            root: cwd_for_commands,
        },
    );

    // Rules: <cwd>/.claude/rules/*.md
    let cwd_for_rules = cwd_root.clone();
    collect_md_files_in_dir(
        &claude_local.join("rules"),
        out,
        || ContextFileKind::ProjectLocalRule,
        &Scope::CwdSubtree {
            root: cwd_for_rules,
        },
    );

    // Agents: <cwd>/.claude/agents/*.md
    collect_md_files_in_dir(
        &claude_local.join("agents"),
        out,
        || ContextFileKind::ProjectLocalAgent,
        &Scope::CwdSubtree { root: cwd_root },
    );
}

// ---- entry point ----

/// Walk the user-global locations and the plugin cache, returning one
/// `ContextFile` per discovered + tokenizable entry.
///
/// Project-local + ancestor `CLAUDE.md` walks are **not** called from
/// here — they are per-session and threaded through
/// `attribution::compute_rows` (Phase 3) so dedup can key on resulting
/// file paths across sessions sharing a cwd ancestor.
#[must_use]
pub fn discover_inventory(config: &InventoryConfig) -> Vec<ContextFile> {
    let mut out = walk_global(config);
    out.extend(walk_plugins(config));
    out
}

// ---- diagnostics ----

/// Gate for the parse-failure warning so we print it at most once per
/// process. Mirrors `pricing::warn_once`'s pattern.
static WARN_ONCE: Once = Once::new();

fn warn_once(message: &str) {
    WARN_ONCE.call_once(|| {
        eprintln!("cclens: {message}");
    });
}

#[cfg(test)]
mod tests {
    use std::fs as stdfs;

    use super::*;

    // --- tokenizer ---

    #[test]
    fn tokenize_file_count_returns_some_for_simple_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("hello.md");
        stdfs::write(&path, "hello world\n").unwrap();
        let n = tokenize_file_count(&path).expect("should tokenize");
        assert!(n > 0, "expected non-zero token count, got {n}");
    }

    #[test]
    fn tokenize_file_count_returns_none_for_unreadable_path() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist.md");
        assert!(tokenize_file_count(&missing).is_none());
    }

    #[test]
    fn tokenize_file_count_handles_unicode() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("unicode.md");
        // Mix multi-byte characters across CJK, emoji, accented
        // ASCII; the tokenizer must not crash on any of them.
        stdfs::write(&path, "こんにちは 🌟 café\n").unwrap();
        let n = tokenize_file_count(&path).expect("should tokenize unicode");
        assert!(n > 0);
    }

    // --- walk_global ---

    fn build_global_tree(tmp: &Path) -> InventoryConfig {
        let claude_home = tmp.join(".claude");
        stdfs::create_dir_all(&claude_home).unwrap();
        InventoryConfig {
            claude_home: claude_home.clone(),
            installed_plugins_path: claude_home.join("plugins/installed_plugins.json"),
        }
    }

    #[test]
    fn walk_global_collects_claude_md_rules_skills_agents() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        let claude_home = &config.claude_home;
        stdfs::write(claude_home.join("CLAUDE.md"), "# Global rules\n").unwrap();
        stdfs::create_dir_all(claude_home.join("rules")).unwrap();
        stdfs::write(claude_home.join("rules/r1.md"), "rule 1\n").unwrap();
        stdfs::create_dir_all(claude_home.join("skills/foo")).unwrap();
        stdfs::write(claude_home.join("skills/foo/SKILL.md"), "skill foo\n").unwrap();
        stdfs::create_dir_all(claude_home.join("agents")).unwrap();
        stdfs::write(claude_home.join("agents/a1.md"), "agent 1\n").unwrap();

        let files = walk_global(&config);
        assert_eq!(files.len(), 4, "expected 4 files, got: {files:#?}");
        assert!(
            files
                .iter()
                .any(|f| matches!(f.kind, ContextFileKind::GlobalClaudeMd))
        );
        assert!(
            files
                .iter()
                .any(|f| matches!(f.kind, ContextFileKind::UserRule))
        );
        assert!(
            files
                .iter()
                .any(|f| matches!(f.kind, ContextFileKind::UserSkill))
        );
        assert!(
            files
                .iter()
                .any(|f| matches!(f.kind, ContextFileKind::UserAgent))
        );
        for f in &files {
            assert!(matches!(f.scope, Scope::Global));
            assert!(f.tokens > 0);
        }
    }

    #[test]
    fn walk_global_skips_non_md_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        stdfs::create_dir_all(config.claude_home.join("rules")).unwrap();
        stdfs::write(config.claude_home.join("rules/yes.md"), "ok\n").unwrap();
        stdfs::write(config.claude_home.join("rules/no.txt"), "ignored\n").unwrap();
        stdfs::create_dir_all(config.claude_home.join("rules/subdir")).unwrap();

        let files = walk_global(&config);
        assert_eq!(files.len(), 1);
        assert!(matches!(files[0].kind, ContextFileKind::UserRule));
        assert!(files[0].path.ends_with("yes.md"));
    }

    #[test]
    fn walk_global_handles_missing_subdir_gracefully() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        stdfs::write(config.claude_home.join("CLAUDE.md"), "# Global\n").unwrap();
        // Only CLAUDE.md exists; rules/, skills/, agents/ all absent.
        let files = walk_global(&config);
        assert_eq!(files.len(), 1);
        assert!(matches!(files[0].kind, ContextFileKind::GlobalClaudeMd));
    }

    #[cfg(unix)]
    #[test]
    fn walk_global_handles_symlinked_claude_md() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        let target = tmp.path().join("real-claude-md.md");
        stdfs::write(&target, "real content body\n").unwrap();
        symlink(&target, config.claude_home.join("CLAUDE.md")).unwrap();

        let files = walk_global(&config);
        assert_eq!(files.len(), 1);
        let f = &files[0];
        assert!(matches!(f.kind, ContextFileKind::GlobalClaudeMd));
        // The reported path is the symlink (what the walker found),
        // but tokens reflect the target's content (read_to_string
        // followed the link).
        assert!(f.path.ends_with("CLAUDE.md"));
        let target_tokens = tokenize_file_count(&target).expect("target tokenizes");
        assert_eq!(f.tokens, target_tokens);
    }

    // --- walk_plugins ---

    #[test]
    fn walk_plugins_parses_installed_plugins_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());

        // Two plugins, one with version "1.0.0", one with "unknown".
        let plugin_a_path = tmp.path().join("cache/test-marketplace/test-plugin/1.0.0");
        stdfs::create_dir_all(plugin_a_path.join("skills/test")).unwrap();
        stdfs::write(plugin_a_path.join("skills/test/SKILL.md"), "test\n").unwrap();

        let plugin_b_path = tmp.path().join("cache/test-marketplace/odd-plugin/unknown");
        stdfs::create_dir_all(plugin_b_path.join("skills/odd")).unwrap();
        stdfs::write(plugin_b_path.join("skills/odd/SKILL.md"), "odd\n").unwrap();

        stdfs::create_dir_all(config.claude_home.join("plugins")).unwrap();
        let json = format!(
            r#"{{
                "plugins": {{
                    "test-plugin@test-marketplace": [{{"installPath": {plugin_a:?}, "version": "1.0.0"}}],
                    "odd-plugin@test-marketplace": [{{"installPath": {plugin_b:?}, "version": "unknown"}}]
                }}
            }}"#,
            plugin_a = plugin_a_path.to_string_lossy(),
            plugin_b = plugin_b_path.to_string_lossy(),
        );
        stdfs::write(&config.installed_plugins_path, json).unwrap();

        let mut files = walk_plugins(&config);
        files.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(files.len(), 2, "expected 2 plugin skills, got: {files:#?}");
        for f in &files {
            assert!(matches!(f.kind, ContextFileKind::PluginSkill { .. }));
            assert!(matches!(f.scope, Scope::Global));
            assert!(f.tokens > 0);
        }
    }

    #[test]
    fn walk_plugins_returns_empty_when_json_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        // installed_plugins.json was never created.
        assert!(walk_plugins(&config).is_empty());
    }

    #[test]
    fn walk_plugins_returns_empty_when_json_unparseable() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        stdfs::create_dir_all(config.claude_home.join("plugins")).unwrap();
        stdfs::write(&config.installed_plugins_path, "{ this is not json").unwrap();
        // Empty result; warn_once may or may not have fired in this process
        // depending on test order — assertion is on the inventory only.
        assert!(walk_plugins(&config).is_empty());
    }

    // --- walk_for_session ---

    #[test]
    fn walk_for_session_finds_ancestor_claude_md() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        let root = tmp.path().join("root");
        let proj = root.join("sub/proj");
        stdfs::create_dir_all(&proj).unwrap();
        stdfs::write(root.join("CLAUDE.md"), "# Project\n").unwrap();

        let files = walk_for_session(&proj, &config);
        let claude_md_rows: Vec<_> = files
            .iter()
            .filter(|f| matches!(f.kind, ContextFileKind::ProjectClaudeMd))
            .collect();
        assert_eq!(claude_md_rows.len(), 1);
        let f = claude_md_rows[0];
        assert_eq!(f.path, root.join("CLAUDE.md"));
        match &f.scope {
            Scope::CwdSubtree { root: r } => assert_eq!(r, &root),
            Scope::Global => panic!("expected CwdSubtree, got Global"),
        }
    }

    #[test]
    fn walk_for_session_finds_project_local_skills() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        let proj = tmp.path().join("proj");
        stdfs::create_dir_all(proj.join(".claude/skills/foo")).unwrap();
        stdfs::write(proj.join(".claude/skills/foo/SKILL.md"), "skill\n").unwrap();

        let files = walk_for_session(&proj, &config);
        let skills: Vec<_> = files
            .iter()
            .filter(|f| matches!(f.kind, ContextFileKind::ProjectLocalSkill))
            .collect();
        assert_eq!(skills.len(), 1);
        match &skills[0].scope {
            Scope::CwdSubtree { root } => assert_eq!(root, &proj),
            Scope::Global => panic!("expected CwdSubtree"),
        }
    }

    #[test]
    fn walk_for_session_handles_missing_project_local_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let config = build_global_tree(tmp.path());
        let proj = tmp.path().join("proj");
        stdfs::create_dir_all(&proj).unwrap();
        // No .claude/ subdir, no ancestor CLAUDE.md.
        let files = walk_for_session(&proj, &config);
        assert!(
            files.is_empty(),
            "expected empty inventory, got: {files:#?}"
        );
    }

    // --- Scope ---

    #[test]
    fn scope_matches_global_for_any_cwd() {
        let s = Scope::Global;
        assert!(s.matches(Path::new("/anywhere")));
        assert!(s.matches(Path::new("/")));
    }

    #[test]
    fn scope_matches_cwd_subtree_only_for_descendants() {
        let s = Scope::CwdSubtree {
            root: PathBuf::from("/a/b"),
        };
        assert!(s.matches(Path::new("/a/b")));
        assert!(s.matches(Path::new("/a/b/c")));
        assert!(!s.matches(Path::new("/a")));
        assert!(!s.matches(Path::new("/a/c")));
    }

    // --- ContextFile::identifier ---

    fn ctx(path: &str, kind: ContextFileKind) -> ContextFile {
        ContextFile {
            path: PathBuf::from(path),
            kind,
            tokens: 0,
            scope: Scope::Global,
        }
    }

    #[test]
    fn identifier_returns_file_stem_for_agent_kinds() {
        assert_eq!(
            ctx("/g/agents/tw-code-reviewer.md", ContextFileKind::UserAgent)
                .identifier()
                .as_deref(),
            Some("tw-code-reviewer"),
        );
        assert_eq!(
            ctx(
                "/proj/.claude/agents/local-helper.md",
                ContextFileKind::ProjectLocalAgent
            )
            .identifier()
            .as_deref(),
            Some("local-helper"),
        );
        assert_eq!(
            ctx(
                "/cache/plug/agents/some-agent.md",
                ContextFileKind::PluginAgent {
                    plugin: "p".into(),
                    marketplace: "m".into()
                }
            )
            .identifier()
            .as_deref(),
            Some("some-agent"),
        );
    }

    #[test]
    fn identifier_returns_file_stem_for_command_kind() {
        assert_eq!(
            ctx(
                "/proj/.claude/commands/foo.md",
                ContextFileKind::ProjectLocalCommand
            )
            .identifier()
            .as_deref(),
            Some("foo"),
        );
    }

    #[test]
    fn identifier_returns_parent_dir_name_for_skill_kinds() {
        // Skills are subdir-keyed: <root>/skills/<NAME>/SKILL.md.
        // identifier() must return <NAME>, not the SKILL.md stem.
        assert_eq!(
            ctx(
                "/g/skills/tw-implement-plan/SKILL.md",
                ContextFileKind::UserSkill
            )
            .identifier()
            .as_deref(),
            Some("tw-implement-plan"),
        );
        assert_eq!(
            ctx(
                "/proj/.claude/skills/local-skill/SKILL.md",
                ContextFileKind::ProjectLocalSkill
            )
            .identifier()
            .as_deref(),
            Some("local-skill"),
        );
        assert_eq!(
            ctx(
                "/cache/plug/skills/plugin-skill/SKILL.md",
                ContextFileKind::PluginSkill {
                    plugin: "p".into(),
                    marketplace: "m".into()
                }
            )
            .identifier()
            .as_deref(),
            Some("plugin-skill"),
        );
    }

    #[test]
    fn identifier_returns_none_for_always_loaded_kinds() {
        // CLAUDE.md and rules are always-loaded; the harness doesn't
        // refer to them by name in the JSONL evidence.
        for kind in [
            ContextFileKind::GlobalClaudeMd,
            ContextFileKind::ProjectClaudeMd,
            ContextFileKind::UserRule,
            ContextFileKind::ProjectLocalRule,
            ContextFileKind::PluginRule {
                plugin: "p".into(),
                marketplace: "m".into(),
            },
        ] {
            let f = ctx("/some/file.md", kind);
            assert!(
                f.identifier().is_none(),
                "expected identifier() = None for {:?}",
                f.kind,
            );
        }
    }
}
