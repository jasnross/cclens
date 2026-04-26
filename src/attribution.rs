//! Attributes inventory tokens to per-tier cache-creation events
//! observed in the JSONL stream and produces ranked rows + coverage.
//!
//! Public API:
//! - `SessionMeta` — per-session summary derived from a `Vec<Turn>`.
//!   Carries event counts and observed-token sums per tier, plus the
//!   metadata `compute_rows` and the `inputs` filters need.
//! - `session_meta_from_turns(...)` — fold a turn list into a
//!   `SessionMeta`. Returns `None` if no turns carried a timestamp.
//! - `AttributionFilter` — `--session` / `--project` / `--since` /
//!   `--until` predicate. Applied to `SessionMeta` before
//!   `compute_rows`.
//! - `extend_inventory_for_session(...)` — append per-session
//!   `walk_for_session` results into a shared inventory, dedup-keyed
//!   on resulting file paths so a shared ancestor `CLAUDE.md` doesn't
//!   double-count.
//! - `compute_rows(...)` — fold inventory + session metadata + pricing
//!   into ranked `AttributionRow`s. Tier routing comes from
//!   `ContextFileKind::tier()`; per-event pricing comes from
//!   `PricingCatalog::cost_for_cache_creation_{1h,5m}`.
//! - `compute_coverage(...)` — per-tier `(observed, attributed,
//!   ratio)` triple covering every session in the row set.
//!
//! Strict `None` propagation (mirroring `aggregation::total_session_cost`):
//! a single in-scope session whose model the catalog can't price
//! collapses the affected row's cost to `None`. The `events` and
//! `estimated_tokens_billed` figures still reflect the full event
//! count — only the cost column collapses.

use std::collections::HashSet;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

use crate::domain::{Role, Turn};
use crate::inventory::{CacheTier, ContextFile, InventoryConfig, walk_for_session};
use crate::pricing::PricingCatalog;

// ---- core types ----

/// Per-session summary derived from a `Vec<Turn>` after dedup.
///
/// Field semantics:
/// - `id` — the JSONL stem (passed in by the caller).
/// - `cwd` — first non-`None` `Turn.cwd` across any role, mirroring
///   `aggregation::derive_project_short_name`'s rule.
/// - `project_short_name` — `cwd.file_name()` if the cwd was found,
///   else `project_dir.file_name()`. Carrying it pre-derived keeps
///   `AttributionFilter::accepts` a single-argument predicate.
/// - `started_at` — earliest `Turn.timestamp`.
/// - `model` — most-frequent model across this session's assistant
///   turns, ties broken by first-occurrence. Used as the per-event
///   pricing model.
/// - `events_1h` / `events_5m` — count of assistant turns whose
///   matching `cache_creation.ephemeral_*` is non-zero. Multiplier
///   used by `compute_rows` for the matching-tier rows.
/// - `observed_1h_tokens` / `observed_5m_tokens` — sum of
///   `cache_creation.ephemeral_*` across all assistant turns. Used
///   by `compute_coverage` for the matching tier's denominator.
#[derive(Debug)]
pub struct SessionMeta {
    pub id: String,
    pub cwd: Option<PathBuf>,
    pub project_short_name: String,
    pub started_at: DateTime<Utc>,
    pub model: Option<String>,
    pub events_1h: u64,
    pub events_5m: u64,
    pub observed_1h_tokens: u64,
    pub observed_5m_tokens: u64,
}

/// One ranked attribution row.
///
/// `events` is the per-row multiplier — 1h-tier rows sum
/// `meta.events_1h`, 5m-tier rows sum `meta.events_5m`.
/// `attributed_cost` is priced at the row's tier rate.
#[derive(Debug)]
pub struct AttributionRow {
    pub file: ContextFile,
    pub events: u64,
    pub estimated_tokens_billed: u64,
    pub attributed_cost: Option<f64>,
}

/// Per-tier coverage figures for the rendered footer.
///
/// `ratio` is `None` when `observed_tokens == 0` (avoids
/// divide-by-zero, renders as `n/a`).
#[derive(Debug)]
pub struct TierCoverage {
    pub observed_tokens: u64,
    pub attributed_tokens: u64,
    pub ratio: Option<f64>,
}

/// Both tiers reported independently so the user can see "I have good
/// coverage of my 1h system-prompt cost but the 5m skill cost is
/// mostly unattributable" or vice versa.
#[derive(Debug)]
pub struct CoverageStats {
    pub long_1h: TierCoverage,
    pub short_5m: TierCoverage,
}

/// `--session` / `--project` / `--since` / `--until` predicate. The
/// `--min-tokens` / `--min-cost` row-level filter is applied
/// separately by `run_inputs` after attribution.
#[derive(Debug, Default)]
pub struct AttributionFilter {
    pub session_id: Option<String>,
    pub project_name: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
}

impl AttributionFilter {
    /// Whether this filter accepts `meta`. `since` and `until` are
    /// inclusive on both ends.
    #[must_use]
    pub fn accepts(&self, meta: &SessionMeta) -> bool {
        if let Some(id) = &self.session_id
            && meta.id != *id
        {
            return false;
        }
        if let Some(name) = &self.project_name
            && meta.project_short_name != *name
        {
            return false;
        }
        if let Some(since) = &self.since
            && meta.started_at < *since
        {
            return false;
        }
        if let Some(until) = &self.until
            && meta.started_at > *until
        {
            return false;
        }
        true
    }
}

// ---- per-session metadata extraction ----

/// Fold a turn list into a `SessionMeta`. Returns `None` if no turn
/// carried a timestamp (matches `aggregation::aggregate`'s rule).
//
// `events_1h`/`events_5m` and `observed_1h_tokens`/`observed_5m_tokens`
// are the domain-essential per-tier counter names — silencing
// `similar_names` here is a deliberate readability tradeoff in favor
// of the names the rest of the module reads back.
#[allow(clippy::similar_names)]
#[must_use]
pub fn session_meta_from_turns(
    session_id: String,
    project_dir: &Path,
    turns: &[Turn],
) -> Option<SessionMeta> {
    let started_at = turns.iter().filter_map(|t| t.timestamp).min()?;
    let cwd = turns.iter().find_map(|t| t.cwd.clone());
    let project_short_name = cwd
        .as_ref()
        .and_then(|c| c.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| {
            project_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
    let model = most_frequent_assistant_model(turns);

    let mut events_1h: u64 = 0;
    let mut events_5m: u64 = 0;
    let mut observed_1h_tokens: u64 = 0;
    let mut observed_5m_tokens: u64 = 0;
    for turn in turns {
        match &turn.role {
            Role::Assistant => {}
            Role::User | Role::Attachment | Role::System | Role::Other(_) => continue,
        }
        let Some(usage) = turn.usage.as_ref() else {
            continue;
        };
        if usage.cache_creation.ephemeral_1h > 0 {
            events_1h += 1;
        }
        if usage.cache_creation.ephemeral_5m > 0 {
            events_5m += 1;
        }
        observed_1h_tokens += usage.cache_creation.ephemeral_1h;
        observed_5m_tokens += usage.cache_creation.ephemeral_5m;
    }

    Some(SessionMeta {
        id: session_id,
        cwd,
        project_short_name,
        started_at,
        model,
        events_1h,
        events_5m,
        observed_1h_tokens,
        observed_5m_tokens,
    })
}

/// Pick the most-frequent assistant model. Iteration order is
/// preserved (Vec insertion order = first-occurrence order); ties
/// resolve to the first-seen model.
///
/// Avoids `HashMap` deliberately — `HashMap` iteration order is
/// nondeterministic in Rust, so a tied vote could resolve differently
/// across runs.
fn most_frequent_assistant_model(turns: &[Turn]) -> Option<String> {
    let mut counts: Vec<(String, u64)> = Vec::new();
    for turn in turns {
        match &turn.role {
            Role::Assistant => {}
            Role::User | Role::Attachment | Role::System | Role::Other(_) => continue,
        }
        let Some(model) = turn.model.as_deref() else {
            continue;
        };
        if let Some(entry) = counts.iter_mut().find(|(m, _)| m == model) {
            entry.1 += 1;
        } else {
            counts.push((model.to_string(), 1));
        }
    }
    let mut best: Option<&(String, u64)> = None;
    for entry in &counts {
        // Strict `>` lets the first occurrence of a tied count win.
        if best.is_none_or(|b| entry.1 > b.1) {
            best = Some(entry);
        }
    }
    best.map(|(m, _)| m.clone())
}

// ---- per-session inventory extension ----

/// Append the per-session `walk_for_session` results into
/// `global_inventory`, dedup-keyed on the resulting file path.
///
/// Why dedupe by file path, not by cwd: two sessions with cwds
/// `/proj/sub1` and `/proj/sub2` both resolve `/proj/CLAUDE.md`
/// via ancestor walking. Without path-keyed dedup, the shared
/// `CLAUDE.md` would appear in the inventory twice and
/// `compute_rows` would double-count its events. The walker still
/// runs per session — the saving here is structural correctness,
/// not performance.
pub fn extend_inventory_for_session<S: BuildHasher>(
    global_inventory: &mut Vec<ContextFile>,
    seen_paths: &mut HashSet<PathBuf, S>,
    cwd: &Path,
    config: &InventoryConfig,
) {
    for file in walk_for_session(cwd, config) {
        if seen_paths.insert(file.path.clone()) {
            global_inventory.push(file);
        }
    }
}

// ---- compute rows + coverage ----

/// Fold the inventory + session metadata + pricing catalog into
/// ranked rows. See module docs for the strict-`None` cost rule.
///
/// Sort order: descending by `attributed_cost.unwrap_or(0.0)`, then
/// descending by `estimated_tokens_billed`, then ascending by
/// `file.path`. Deterministic across runs even when costs collide
/// or collapse to `None`.
#[must_use]
pub fn compute_rows(
    inventory: Vec<ContextFile>,
    session_metas: &[SessionMeta],
    catalog: &PricingCatalog,
) -> Vec<AttributionRow> {
    let mut rows: Vec<AttributionRow> = inventory
        .into_iter()
        .map(|file| build_row(file, session_metas, catalog))
        .collect();
    rows.sort_by(|a, b| {
        let a_cost = a.attributed_cost.unwrap_or(0.0);
        let b_cost = b.attributed_cost.unwrap_or(0.0);
        b_cost
            .partial_cmp(&a_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.estimated_tokens_billed.cmp(&a.estimated_tokens_billed))
            .then_with(|| a.file.path.cmp(&b.file.path))
    });
    rows
}

#[allow(clippy::cast_precision_loss)]
fn build_row(
    file: ContextFile,
    session_metas: &[SessionMeta],
    catalog: &PricingCatalog,
) -> AttributionRow {
    let tier = file.kind.tier();
    let mut total_events: u64 = 0;
    let mut total_cost: Option<f64> = Some(0.0);

    for meta in session_metas {
        let Some(meta_cwd) = meta.cwd.as_deref() else {
            continue;
        };
        if !file.scope.matches(meta_cwd) {
            continue;
        }
        let session_events = match tier {
            CacheTier::Long1h => meta.events_1h,
            CacheTier::Short5m => meta.events_5m,
        };
        total_events += session_events;
        if session_events == 0 {
            // No cost contribution from this session; the unknown-
            // model rule only collapses the row when at least one
            // event would have been billed.
            continue;
        }
        if total_cost.is_none() {
            // Already collapsed; keep accumulating events but skip
            // the price lookup.
            continue;
        }
        let per_event = match tier {
            CacheTier::Long1h => {
                catalog.cost_for_cache_creation_1h(file.tokens, meta.model.as_deref())
            }
            CacheTier::Short5m => {
                catalog.cost_for_cache_creation_5m(file.tokens, meta.model.as_deref())
            }
        };
        match per_event {
            Some(c) => {
                total_cost = total_cost.map(|t| t + c * session_events as f64);
            }
            None => {
                total_cost = None;
            }
        }
    }

    let estimated_tokens_billed = file.tokens.saturating_mul(total_events);
    AttributionRow {
        file,
        events: total_events,
        estimated_tokens_billed,
        attributed_cost: total_cost,
    }
}

/// Per-tier coverage: `observed` from session metas, `attributed`
/// from rows whose tier matches.
#[must_use]
pub fn compute_coverage(session_metas: &[SessionMeta], rows: &[AttributionRow]) -> CoverageStats {
    let mut long_observed: u64 = 0;
    let mut short_observed: u64 = 0;
    for meta in session_metas {
        long_observed += meta.observed_1h_tokens;
        short_observed += meta.observed_5m_tokens;
    }
    let mut long_attributed: u64 = 0;
    let mut short_attributed: u64 = 0;
    for row in rows {
        match row.file.kind.tier() {
            CacheTier::Long1h => long_attributed += row.estimated_tokens_billed,
            CacheTier::Short5m => short_attributed += row.estimated_tokens_billed,
        }
    }
    CoverageStats {
        long_1h: tier_coverage(long_observed, long_attributed),
        short_5m: tier_coverage(short_observed, short_attributed),
    }
}

#[allow(clippy::cast_precision_loss)]
fn tier_coverage(observed: u64, attributed: u64) -> TierCoverage {
    let ratio = if observed > 0 {
        Some(attributed as f64 / observed as f64)
    } else {
        None
    };
    TierCoverage {
        observed_tokens: observed,
        attributed_tokens: attributed,
        ratio,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::domain::{CacheCreation, Usage};
    use crate::inventory::{ContextFileKind, Scope};

    // --- test helpers ---

    fn ts(s: &str) -> DateTime<Utc> {
        s.parse().unwrap()
    }

    #[allow(clippy::similar_names)]
    fn assistant_turn(
        ts_str: &str,
        model: Option<&str>,
        ephemeral_5m: u64,
        ephemeral_1h: u64,
    ) -> Turn {
        Turn {
            timestamp: Some(ts(ts_str)),
            role: Role::Assistant,
            model: model.map(str::to_string),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input: 0,
                output: 0,
                cache_creation: CacheCreation {
                    ephemeral_5m,
                    ephemeral_1h,
                },
                cache_read: 0,
            }),
            content: None,
            cwd: None,
        }
    }

    fn assistant_turn_no_usage(ts_str: &str, model: Option<&str>) -> Turn {
        Turn {
            timestamp: Some(ts(ts_str)),
            role: Role::Assistant,
            model: model.map(str::to_string),
            message_id: None,
            request_id: None,
            usage: None,
            content: None,
            cwd: None,
        }
    }

    fn user_turn(ts_str: &str, cwd: Option<&Path>) -> Turn {
        Turn {
            timestamp: Some(ts(ts_str)),
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: None,
            cwd: cwd.map(Path::to_path_buf),
        }
    }

    fn ctx_file(path: &str, kind: ContextFileKind, tokens: u64, scope: Scope) -> ContextFile {
        ContextFile {
            path: PathBuf::from(path),
            kind,
            tokens,
            scope,
        }
    }

    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    fn meta_with(
        id: &str,
        cwd: Option<&Path>,
        project: &str,
        started_at_str: &str,
        model: Option<&str>,
        events_1h: u64,
        events_5m: u64,
        observed_1h: u64,
        observed_5m: u64,
    ) -> SessionMeta {
        SessionMeta {
            id: id.to_string(),
            cwd: cwd.map(Path::to_path_buf),
            project_short_name: project.to_string(),
            started_at: ts(started_at_str),
            model: model.map(str::to_string),
            events_1h,
            events_5m,
            observed_1h_tokens: observed_1h,
            observed_5m_tokens: observed_5m,
        }
    }

    fn sample_catalog() -> PricingCatalog {
        // Real LiteLLM-shape JSON for one Claude model with distinct
        // 1h and 5m rates; mirrors `pricing::tests::sample_pricing`.
        let json = r#"{
            "claude-opus-4-7": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_creation_input_token_cost_above_1hr": 0.000006,
                "cache_read_input_token_cost": 0.0000003
            }
        }"#;
        PricingCatalog::from_raw_json(json).unwrap()
    }

    // --- session_meta_from_turns ---

    #[test]
    fn session_meta_counts_per_tier_events_separately() {
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("claude-opus-4-7"), 0, 100),
            assistant_turn("2026-04-01T10:01:00Z", Some("claude-opus-4-7"), 100, 0),
            assistant_turn("2026-04-01T10:02:00Z", Some("claude-opus-4-7"), 50, 50),
            assistant_turn_no_usage("2026-04-01T10:03:00Z", Some("claude-opus-4-7")),
        ];
        let meta =
            session_meta_from_turns("session-id".to_string(), Path::new("/proj"), &turns).unwrap();
        assert_eq!(meta.events_1h, 2, "expected events_1h == 2");
        assert_eq!(meta.events_5m, 2, "expected events_5m == 2");
        assert_eq!(meta.observed_1h_tokens, 150);
        assert_eq!(meta.observed_5m_tokens, 150);
    }

    #[test]
    fn session_meta_picks_majority_model() {
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("model-a"), 100, 0),
            assistant_turn("2026-04-01T10:01:00Z", Some("model-a"), 100, 0),
            assistant_turn("2026-04-01T10:02:00Z", Some("model-b"), 100, 0),
        ];
        let meta = session_meta_from_turns("id".to_string(), Path::new("/proj"), &turns).unwrap();
        assert_eq!(meta.model.as_deref(), Some("model-a"));
    }

    #[test]
    fn session_meta_breaks_ties_by_first_occurrence() {
        // Two models, equal counts. First-seen must win.
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("model-x"), 100, 0),
            assistant_turn("2026-04-01T10:01:00Z", Some("model-y"), 100, 0),
        ];
        let meta = session_meta_from_turns("id".to_string(), Path::new("/proj"), &turns).unwrap();
        assert_eq!(meta.model.as_deref(), Some("model-x"));
    }

    #[test]
    fn session_meta_picks_first_cwd_from_any_role() {
        // User turn with cwd=/x precedes assistant turn with cwd=/y;
        // user's cwd wins (matches aggregation::derive_project_short_name).
        let mut user = user_turn("2026-04-01T10:00:00Z", Some(Path::new("/x")));
        user.timestamp = Some(ts("2026-04-01T10:00:00Z"));
        let mut asst = assistant_turn("2026-04-01T10:01:00Z", Some("model-a"), 100, 0);
        asst.cwd = Some(PathBuf::from("/y"));
        let turns = vec![user, asst];
        let meta = session_meta_from_turns("id".to_string(), Path::new("/proj"), &turns).unwrap();
        assert_eq!(meta.cwd, Some(PathBuf::from("/x")));
        assert_eq!(meta.project_short_name, "x");
    }

    #[test]
    fn session_meta_returns_none_for_no_timestamps() {
        // Empty turns → None.
        let empty: Vec<Turn> = Vec::new();
        assert!(session_meta_from_turns("id".to_string(), Path::new("/p"), &empty).is_none());

        // Only-untimestamped turns → also None.
        let untimestamped = vec![Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: None,
            cwd: None,
        }];
        assert!(
            session_meta_from_turns("id".to_string(), Path::new("/p"), &untimestamped).is_none()
        );
    }

    #[test]
    fn session_meta_falls_back_to_project_dir_name_when_no_cwd() {
        let turns = vec![assistant_turn(
            "2026-04-01T10:00:00Z",
            Some("model-a"),
            100,
            0,
        )];
        let meta = session_meta_from_turns("id".to_string(), Path::new("/some/proj-name"), &turns)
            .unwrap();
        assert!(meta.cwd.is_none());
        assert_eq!(meta.project_short_name, "proj-name");
    }

    // --- compute_rows ---

    #[test]
    fn compute_rows_routes_1h_kind_to_events_1h() {
        let inv = vec![ctx_file(
            "/g/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            Scope::Global,
        )];
        let metas = vec![
            meta_with(
                "s1",
                Some(Path::new("/a")),
                "a",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                3,
                7,
                300,
                700,
            ),
            meta_with(
                "s2",
                Some(Path::new("/b")),
                "b",
                "2026-04-01T11:00:00Z",
                Some("claude-opus-4-7"),
                5,
                11,
                500,
                1100,
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].events, 8, "1h-tier file must sum events_1h");
        assert_eq!(rows[0].estimated_tokens_billed, 800);
    }

    #[test]
    fn compute_rows_routes_5m_kind_to_events_5m() {
        let inv = vec![ctx_file(
            "/g/skills/foo/SKILL.md",
            ContextFileKind::UserSkill,
            100,
            Scope::Global,
        )];
        let metas = vec![
            meta_with(
                "s1",
                Some(Path::new("/a")),
                "a",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                3,
                7,
                300,
                700,
            ),
            meta_with(
                "s2",
                Some(Path::new("/b")),
                "b",
                "2026-04-01T11:00:00Z",
                Some("claude-opus-4-7"),
                5,
                11,
                500,
                1100,
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].events, 18, "5m-tier file must sum events_5m");
        assert_eq!(rows[0].estimated_tokens_billed, 1800);
    }

    #[test]
    fn compute_rows_attributes_subtree_file_only_to_in_scope_sessions() {
        let inv = vec![ctx_file(
            "/proj/CLAUDE.md",
            ContextFileKind::ProjectClaudeMd,
            100,
            Scope::CwdSubtree {
                root: PathBuf::from("/proj"),
            },
        )];
        let metas = vec![
            meta_with(
                "s1",
                Some(Path::new("/proj/sub")),
                "sub",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                3,
                0,
                300,
                0,
            ),
            meta_with(
                "s2",
                Some(Path::new("/other")),
                "other",
                "2026-04-01T11:00:00Z",
                Some("claude-opus-4-7"),
                5,
                0,
                500,
                0,
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(
            rows[0].events, 3,
            "out-of-scope session must not contribute"
        );
    }

    #[test]
    fn compute_rows_prices_5m_at_5m_rate_not_1h() {
        let inv = vec![ctx_file(
            "/g/skills/x/SKILL.md",
            ContextFileKind::UserSkill,
            1000,
            Scope::Global,
        )];
        let metas = vec![meta_with(
            "s1",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            0,
            1,
            0,
            1000,
        )];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        // 1000 * 1 events * 3.75e-6 (5m rate) = 0.00375; NOT 6e-6 (1h rate).
        let cost = rows[0].attributed_cost.expect("priced");
        assert!(
            (cost - 0.003_75).abs() < 1e-12,
            "got {cost}, expected 5m rate"
        );
    }

    #[test]
    fn compute_rows_collapses_cost_to_none_on_unknown_model() {
        let inv = vec![ctx_file(
            "/g/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            Scope::Global,
        )];
        let metas = vec![
            meta_with(
                "s1",
                Some(Path::new("/a")),
                "a",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                3,
                0,
                300,
                0,
            ),
            // Unknown model with non-zero events → row's cost = None.
            meta_with(
                "s2",
                Some(Path::new("/b")),
                "b",
                "2026-04-01T11:00:00Z",
                Some("claude-fake-9-9"),
                5,
                0,
                500,
                0,
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(rows[0].events, 8, "events sum still reflects both sessions");
        assert!(
            rows[0].attributed_cost.is_none(),
            "unknown-model session must collapse cost"
        );
    }

    #[test]
    fn compute_rows_sorts_descending_by_cost_then_tokens_then_path() {
        // Three rows with distinct (cost, tokens, path) so the sort
        // outcome is unambiguous.
        let inv = vec![
            // tokens=200 with 1 event: cost = 200 * 1 * 6e-6 = 0.0012
            ctx_file(
                "/aaa/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                200,
                Scope::Global,
            ),
            // tokens=100 with 1 event: cost = 100 * 1 * 6e-6 = 0.0006
            ctx_file(
                "/bbb/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                100,
                Scope::Global,
            ),
            // tokens=200 with 0 events (different scope): cost = 0
            ctx_file(
                "/ccc/proj/CLAUDE.md",
                ContextFileKind::ProjectClaudeMd,
                200,
                Scope::CwdSubtree {
                    root: PathBuf::from("/somewhere-else"),
                },
            ),
        ];
        let metas = vec![meta_with(
            "s1",
            Some(Path::new("/scope")),
            "scope",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            1,
            0,
            200,
            0,
        )];
        // Run multiple times to confirm deterministic ordering.
        // `ContextFile`'s `Clone` derive is `cfg(test)`-gated so
        // `inv.clone()` works inside tests but production sites can't
        // accidentally allocate.
        for _ in 0..5 {
            let rows = compute_rows(inv.clone(), &metas, &sample_catalog());
            assert_eq!(rows[0].file.path, PathBuf::from("/aaa/CLAUDE.md"));
            assert_eq!(rows[1].file.path, PathBuf::from("/bbb/CLAUDE.md"));
            assert_eq!(rows[2].file.path, PathBuf::from("/ccc/proj/CLAUDE.md"));
        }
    }

    #[test]
    fn compute_rows_includes_zero_event_in_scope_files() {
        let inv = vec![ctx_file(
            "/g/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            Scope::Global,
        )];
        // No sessions → zero events.
        let rows = compute_rows(inv, &[], &sample_catalog());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].events, 0);
        assert_eq!(rows[0].estimated_tokens_billed, 0);
        assert_eq!(rows[0].attributed_cost, Some(0.0));
    }

    // --- compute_coverage ---

    #[test]
    fn compute_coverage_handles_zero_observed_in_both_tiers() {
        let cov = compute_coverage(&[], &[]);
        assert_eq!(cov.long_1h.observed_tokens, 0);
        assert_eq!(cov.long_1h.attributed_tokens, 0);
        assert!(cov.long_1h.ratio.is_none());
        assert_eq!(cov.short_5m.observed_tokens, 0);
        assert!(cov.short_5m.ratio.is_none());
    }

    #[test]
    fn compute_coverage_ratios_correctly_per_tier() {
        let metas = vec![meta_with(
            "s1",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            0,
            0,
            5_000, // observed_1h
            2_000, // observed_5m
        )];
        // Hand-crafted rows with known attributed sums per tier.
        let rows = vec![
            AttributionRow {
                file: ctx_file(
                    "/g/CLAUDE.md",
                    ContextFileKind::GlobalClaudeMd,
                    100,
                    Scope::Global,
                ),
                events: 40,
                estimated_tokens_billed: 4_000,
                attributed_cost: Some(0.0),
            },
            AttributionRow {
                file: ctx_file(
                    "/g/skills/x/SKILL.md",
                    ContextFileKind::UserSkill,
                    50,
                    Scope::Global,
                ),
                events: 30,
                estimated_tokens_billed: 1_500,
                attributed_cost: Some(0.0),
            },
        ];
        let cov = compute_coverage(&metas, &rows);
        assert_eq!(cov.long_1h.observed_tokens, 5_000);
        assert_eq!(cov.long_1h.attributed_tokens, 4_000);
        assert!((cov.long_1h.ratio.unwrap() - 0.8).abs() < 1e-12);
        assert_eq!(cov.short_5m.observed_tokens, 2_000);
        assert_eq!(cov.short_5m.attributed_tokens, 1_500);
        assert!((cov.short_5m.ratio.unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn compute_coverage_observed_sums_match_independent_per_tier_sum() {
        // Fixed-point invariant for the observed-token side: build a
        // Vec<Turn> with deterministic per-tier contributions, sum
        // them by hand, then run the same turns through
        // session_meta_from_turns + compute_coverage and assert the
        // reported observed_*_tokens match. Catches regressions in
        // session_meta_from_turns's per-tier sums and in the
        // parsing → SessionMeta → CoverageStats flow.
        //
        // The attributed-token side of compute_coverage is exercised
        // by compute_coverage_ratios_correctly_per_tier, which
        // hand-crafts rows with known tier sums.
        //
        // Per-turn (5m, 1h) contributions:
        // (50, 100), (200, _), (_, 250), no-usage, (100, 75).
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("claude-opus-4-7"), 50, 100),
            assistant_turn("2026-04-01T10:01:00Z", Some("claude-opus-4-7"), 200, 0),
            assistant_turn("2026-04-01T10:02:00Z", Some("claude-opus-4-7"), 0, 250),
            assistant_turn_no_usage("2026-04-01T10:03:00Z", Some("claude-opus-4-7")),
            assistant_turn("2026-04-01T10:04:00Z", Some("claude-opus-4-7"), 100, 75),
        ];
        // Independently summed (omitting the zero contributions).
        let expected_1h: u64 = 100 + 250 + 75;
        let expected_5m: u64 = 50 + 200 + 100;
        assert_eq!(expected_1h, 425);
        assert_eq!(expected_5m, 350);

        let meta = session_meta_from_turns("s1".to_string(), Path::new("/proj"), &turns).unwrap();
        let cov = compute_coverage(&[meta], &[]);
        assert_eq!(cov.long_1h.observed_tokens, expected_1h);
        assert_eq!(cov.short_5m.observed_tokens, expected_5m);
    }

    // --- AttributionFilter ---

    #[test]
    fn attribution_filter_session_id_match() {
        let filter = AttributionFilter {
            session_id: Some("target".to_string()),
            ..Default::default()
        };
        let target = meta_with(
            "target",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            None,
            0,
            0,
            0,
            0,
        );
        let other = meta_with(
            "other",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            None,
            0,
            0,
            0,
            0,
        );
        assert!(filter.accepts(&target));
        assert!(!filter.accepts(&other));
    }

    #[test]
    fn attribution_filter_since_until_inclusive() {
        let filter = AttributionFilter {
            since: Some(ts("2026-04-10T00:00:00Z")),
            until: Some(ts("2026-04-20T00:00:00Z")),
            ..Default::default()
        };
        let on_since = meta_with(
            "s1",
            Some(Path::new("/a")),
            "a",
            "2026-04-10T00:00:00Z",
            None,
            0,
            0,
            0,
            0,
        );
        let on_until = meta_with(
            "s2",
            Some(Path::new("/a")),
            "a",
            "2026-04-20T00:00:00Z",
            None,
            0,
            0,
            0,
            0,
        );
        let before = meta_with(
            "s3",
            Some(Path::new("/a")),
            "a",
            "2026-04-09T23:59:59Z",
            None,
            0,
            0,
            0,
            0,
        );
        let after = meta_with(
            "s4",
            Some(Path::new("/a")),
            "a",
            "2026-04-20T00:00:01Z",
            None,
            0,
            0,
            0,
            0,
        );
        assert!(filter.accepts(&on_since));
        assert!(filter.accepts(&on_until));
        assert!(!filter.accepts(&before));
        assert!(!filter.accepts(&after));
    }

    #[test]
    fn attribution_filter_project_name_match() {
        let filter = AttributionFilter {
            project_name: Some("foo".to_string()),
            ..Default::default()
        };
        let foo = meta_with(
            "s1",
            Some(Path::new("/a")),
            "foo",
            "2026-04-01T10:00:00Z",
            None,
            0,
            0,
            0,
            0,
        );
        let bar = meta_with(
            "s2",
            Some(Path::new("/a")),
            "bar",
            "2026-04-01T10:00:00Z",
            None,
            0,
            0,
            0,
            0,
        );
        assert!(filter.accepts(&foo));
        assert!(!filter.accepts(&bar));
    }

    // --- extend_inventory_for_session ---

    #[test]
    fn extend_inventory_for_session_dedupes_repeated_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().join("proj");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(proj.join("CLAUDE.md"), "# proj\n").unwrap();

        let config = InventoryConfig {
            claude_home: tmp.path().join(".claude"),
            installed_plugins_path: tmp.path().join(".claude/plugins/installed_plugins.json"),
        };
        let mut inventory: Vec<ContextFile> = Vec::new();
        let mut seen: HashSet<PathBuf> = HashSet::new();
        extend_inventory_for_session(&mut inventory, &mut seen, &proj, &config);
        let after_first = inventory.len();
        // Second call with the same cwd must not duplicate.
        extend_inventory_for_session(&mut inventory, &mut seen, &proj, &config);
        assert_eq!(inventory.len(), after_first);
    }

    #[test]
    fn extend_inventory_for_session_dedupes_shared_ancestor_claude_md() {
        // Two distinct session cwds /proj/sub1 and /proj/sub2 both
        // resolve /proj/CLAUDE.md via ancestor walking. The shared
        // file must appear in the inventory exactly once.
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().join("proj");
        let sub1 = proj.join("sub1");
        let sub2 = proj.join("sub2");
        std::fs::create_dir_all(&sub1).unwrap();
        std::fs::create_dir_all(&sub2).unwrap();
        std::fs::write(proj.join("CLAUDE.md"), "# shared\n").unwrap();

        let config = InventoryConfig {
            claude_home: tmp.path().join(".claude"),
            installed_plugins_path: tmp.path().join(".claude/plugins/installed_plugins.json"),
        };
        let mut inventory: Vec<ContextFile> = Vec::new();
        let mut seen: HashSet<PathBuf> = HashSet::new();
        extend_inventory_for_session(&mut inventory, &mut seen, &sub1, &config);
        extend_inventory_for_session(&mut inventory, &mut seen, &sub2, &config);

        let proj_claude_mds: Vec<_> = inventory
            .iter()
            .filter(|f| matches!(f.kind, ContextFileKind::ProjectClaudeMd))
            .filter(|f| f.path == proj.join("CLAUDE.md"))
            .collect();
        assert_eq!(
            proj_claude_mds.len(),
            1,
            "shared ancestor CLAUDE.md must dedupe to a single entry, got: {inventory:#?}"
        );
    }
}
