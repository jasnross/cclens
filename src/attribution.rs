//! Attributes inventory tokens to per-load cache-creation evidence
//! extracted from the JSONL stream and produces ranked rows + coverage.
//!
//! Public API:
//! - `SessionMeta` — per-session summary derived from a `Vec<Turn>`.
//!   Carries the session's `primary_tier`, observed-token sums per
//!   tier, and `on_demand_loads` (skill/command invocations extracted
//!   from `Turn.content`), plus the metadata `compute_rows` and the
//!   `inputs` filters need.
//! - `SessionKind` — `Parent` vs. `Subagent { agent_type }`
//!   discriminator. `Subagent` payload's `agent_type` is matched
//!   against `ContextFile::identifier()` so an agent file is credited
//!   only by the subagent transcripts that actually loaded it.
//! - `OnDemandLoad` / `OnDemandKind` — per-(skill, command) load
//!   record extracted from user-turn content.
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
//!   into ranked `AttributionRow`s. Each file's row reports
//!   `loads_1h` / `loads_5m` — the actual count of cache-creation
//!   events that loaded its body, not a session-wide multiplier. Per-
//!   tier pricing comes from `PricingCatalog::cost_for_cache_creation_{1h,5m}`.
//! - `compute_coverage(...)` — per-tier `(observed, attributed,
//!   ratio)` triple summed across every parent + subagent session.
//!
//! Strict `None` propagation (mirroring `aggregation::total_session_cost`):
//! a single contributing session whose model the catalog can't price
//! collapses the affected row's cost to `None`. The `loads_*` and
//! `estimated_tokens_billed` figures still reflect the full load
//! count — only the cost column collapses.
//!
//! Known limitations of the per-load model:
//! - **Skill auto-trigger detection.** The harness can auto-load a
//!   skill (when trigger conditions match) without a `Skill` `tool_use`
//!   or slash command. The skill body still appears in the JSONL
//!   stream, but distinguishing auto-load from manual is non-trivial
//!   — auto-loaded skills currently miss attribution. Revisit if it
//!   bites.
//! - **Plugin / user skill-name collisions.** If a user skill and a
//!   plugin skill share a directory name, both are credited. Accepted
//!   edge case.
//! - **`agentType` ↔ file-stem normalization.** Subagent crediting
//!   matches the sidecar's `agentType` against the agent file's stem
//!   verbatim. Older Claude Code versions occasionally emitted
//!   `agentType` strings using `:` separators (e.g.
//!   `tw:code-reviewer`) where the inventory file stem uses `-`
//!   (`tw-code-reviewer.md`). Such mismatches silently miss
//!   attribution; the matcher does not normalize.
//! - **Cache invalidation modeling.** 1h TTL expiry, prefix changes
//!   mid-session, and manual `/clear` could re-load CLAUDE.md / rules
//!   within a single session. We assume one load per session per
//!   always-loaded file. Acceptable approximation given the data we
//!   have.

use std::collections::HashSet;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::domain::{Role, Turn};
use crate::inventory::{
    CacheTier, ContextFile, ContextFileKind, InventoryConfig, walk_for_session,
};
use crate::pricing::PricingCatalog;

// ---- core types ----

/// Per-session summary derived from a `Vec<Turn>` after dedup.
///
/// Field semantics:
/// - `id` — the JSONL stem (passed in by the caller).
/// - `kind` — discriminates a parent session from a subagent
///   transcript. `compute_rows`' agent-row dispatch uses the
///   `Subagent { agent_type }` payload to credit only the matching
///   agent inventory entry.
/// - `cwd` — first non-`None` `Turn.cwd` across any role, mirroring
///   `aggregation::derive_project_short_name`'s rule. Subagent metas
///   constructed by `run_inputs` fall back to the parent session's
///   cwd when the subagent JSONL has none.
/// - `project_short_name` — `cwd.file_name()` if the cwd was found,
///   else `project_dir.file_name()`. Carrying it pre-derived keeps
///   `AttributionFilter::accepts` a single-argument predicate.
/// - `started_at` — earliest `Turn.timestamp`.
/// - `model` — most-frequent model across this session's assistant
///   turns, ties broken by first-occurrence. Used as the per-load
///   pricing model.
/// - `primary_tier` — tier of this session's first `cache_creation`
///   event (1h preferred when both fire on the same turn, since the
///   1h delta is where always-loaded content lives). Falls back to
///   `Long1h` when the session has no `cache_creation` activity at
///   all. `compute_rows` uses this as the tier for always-loaded
///   inventory rows (CLAUDE.md, rules) and matched agent rows.
/// - `observed_1h_tokens` / `observed_5m_tokens` — sum of
///   `cache_creation.ephemeral_*` across all assistant turns. Used
///   by `compute_coverage` for the matching tier's denominator.
/// - `on_demand_loads` — slash-command and skill-injection events
///   extracted from `Turn.content`. Each entry is one observed load
///   in this session; `compute_rows` counts them per identifier when
///   crediting skill / command rows.
#[derive(Debug)]
pub struct SessionMeta {
    pub id: String,
    pub kind: SessionKind,
    pub cwd: Option<PathBuf>,
    pub project_short_name: String,
    pub started_at: DateTime<Utc>,
    pub model: Option<String>,
    pub primary_tier: CacheTier,
    pub observed_1h_tokens: u64,
    pub observed_5m_tokens: u64,
    pub on_demand_loads: Vec<OnDemandLoad>,
}

impl SessionMeta {
    /// `Some(primary_tier)` when this session emitted at least one
    /// `cache_creation` event, else `None`. Used by
    /// `evidence_for_file_in_session` to gate always-loaded and
    /// agent-row attribution: a session that never actually loaded
    /// anything (zero `cache_creation` activity in its turn list)
    /// must NOT be credited with implicit CLAUDE.md / rule loads,
    /// which would otherwise inflate the row + coverage numerator.
    #[must_use]
    pub fn observed_tier(&self) -> Option<CacheTier> {
        if self.observed_1h_tokens == 0 && self.observed_5m_tokens == 0 {
            None
        } else {
            Some(self.primary_tier)
        }
    }
}

/// Discriminates a parent session JSONL from a subagent transcript.
///
/// The `Subagent` variant carries the `agentType` resolved from the
/// sibling `.meta.json` sidecar; `compute_rows` matches it against
/// `ContextFile::identifier()` to credit only the matching agent
/// file rather than every agent in scope.
#[derive(Debug, Clone)]
pub enum SessionKind {
    Parent,
    Subagent { agent_type: String },
}

/// One mid-conversation on-demand load extracted from user-turn
/// content: a slash-command invocation or a skill-body injection.
///
/// `identifier` is the harness name (matched against
/// `ContextFile::identifier()`); `tier` is always `Short5m` per
/// observed convention (skill / command bodies are loaded by the
/// harness at the 5m tier, even when the next assistant turn shows
/// 1h `cache_creation` activity — the 1h delta covers unrelated
/// long-lived content getting renewed).
#[derive(Debug, Clone)]
pub struct OnDemandLoad {
    pub kind: OnDemandKind,
    pub identifier: String,
    pub tier: CacheTier,
}

/// Distinguishes a skill-body load from a slash-command-body load.
/// Both attribute at `CacheTier::Short5m`; the discriminator lets
/// `compute_rows` route each load to the right inventory kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnDemandKind {
    Skill,
    Command,
}

/// One ranked attribution row.
///
/// `loads_1h` and `loads_5m` are the per-tier load counts: the count
/// of `cache_creation` events at each tier that loaded this file's
/// body. A row that loads at both tiers (e.g. CLAUDE.md from a 1h-
/// tier parent session plus a 5m-tier subagent) reports both > 0.
/// `attributed_cost` is the sum of per-tier costs:
/// `loads_1h × tokens × rate_1h + loads_5m × tokens × rate_5m`.
#[derive(Debug)]
pub struct AttributionRow {
    pub file: ContextFile,
    pub loads_1h: u64,
    pub loads_5m: u64,
    pub estimated_tokens_billed: u64,
    pub attributed_cost: Option<f64>,
}

impl AttributionRow {
    /// Total load count across both tiers.
    #[must_use]
    pub fn total_loads(&self) -> u64 {
        self.loads_1h + self.loads_5m
    }

    /// Tier label for the rendered `tier` column. `1h+5m` when this
    /// file loaded at both tiers (parent + subagent on different
    /// tiers); `—` when no session loaded the file at all (in scope
    /// but never actually triggered).
    #[must_use]
    pub fn tier_label(&self) -> &'static str {
        match (self.loads_1h, self.loads_5m) {
            (0, 0) => "—",
            (_, 0) => "1h",
            (0, _) => "5m",
            (_, _) => "1h+5m",
        }
    }
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
// `observed_1h_tokens`/`observed_5m_tokens` are the domain-essential
// per-tier counter names — silencing `similar_names` here is a
// deliberate readability tradeoff in favor of the names the rest of
// the module reads back.
#[allow(clippy::similar_names)]
#[must_use]
pub fn session_meta_from_turns(
    kind: SessionKind,
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

    let mut primary_tier: Option<CacheTier> = None;
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
        observed_1h_tokens += usage.cache_creation.ephemeral_1h;
        observed_5m_tokens += usage.cache_creation.ephemeral_5m;
        // Primary tier: tier of the first assistant turn whose
        // cache_creation actually fired. 1h takes precedence on a
        // mixed turn — the 1h delta is the system-prompt-region
        // tier where always-loaded content lives.
        if primary_tier.is_none() {
            if usage.cache_creation.ephemeral_1h > 0 {
                primary_tier = Some(CacheTier::Long1h);
            } else if usage.cache_creation.ephemeral_5m > 0 {
                primary_tier = Some(CacheTier::Short5m);
            }
        }
    }
    // No cache_creation activity at all → default to 1h. Doesn't
    // matter for attribution (no loads will be credited) but keeps
    // the field well-defined.
    let primary_tier = primary_tier.unwrap_or(CacheTier::Long1h);

    let on_demand_loads = extract_on_demand_loads(turns);

    Some(SessionMeta {
        id: session_id,
        kind,
        cwd,
        project_short_name,
        started_at,
        model,
        primary_tier,
        observed_1h_tokens,
        observed_5m_tokens,
        on_demand_loads,
    })
}

/// Walk every user turn's `Turn.content` for on-demand load evidence:
/// slash-command invocations (`<command-name>/foo</command-name>` in
/// string content) and skill-body injections (text blocks beginning
/// with `Base directory for this skill: ` in array content).
///
/// Dispatch shape mirrors what's observed in real Claude Code data:
/// slash commands always arrive as a string `Turn.content`, skill
/// injections always arrive as an array of text blocks. Widening
/// either parser to handle the other shape would risk false positives
/// on real prose — leave them split unless the data actually changes.
///
/// All on-demand loads attribute at `CacheTier::Short5m` — the harness
/// loads skill / command bodies at the 5m tier per observed convention.
fn extract_on_demand_loads(turns: &[Turn]) -> Vec<OnDemandLoad> {
    let mut out: Vec<OnDemandLoad> = Vec::new();
    for turn in turns {
        match &turn.role {
            Role::User => {}
            Role::Assistant | Role::Attachment | Role::System | Role::Other(_) => continue,
        }
        let Some(content) = turn.content.as_ref() else {
            continue;
        };
        if let Some(s) = content.as_str()
            && let Some(cmd) = extract_command_identifier(s)
        {
            out.push(OnDemandLoad {
                kind: OnDemandKind::Command,
                identifier: cmd,
                tier: CacheTier::Short5m,
            });
        }
        if let Some(arr) = content.as_array() {
            for block in arr {
                if block.get("type").and_then(Value::as_str) != Some("text") {
                    continue;
                }
                let Some(text) = block.get("text").and_then(Value::as_str) else {
                    continue;
                };
                if let Some(skill) = extract_skill_identifier(text) {
                    out.push(OnDemandLoad {
                        kind: OnDemandKind::Skill,
                        identifier: skill,
                        tier: CacheTier::Short5m,
                    });
                }
            }
        }
    }
    out
}

/// Pull the command identifier from a `<command-name>/foo</command-name>`
/// string. Mirrors `aggregation::extract_slash_command_title`'s
/// parsing but strips the leading `/` so the result matches a
/// `ProjectLocalCommand` file stem (`commands/foo.md` → `foo`).
///
/// Single-match contract: returns at most one identifier per call,
/// taken from the first `<command-name>` tag in the string. Two
/// commands concatenated into one user turn (rare — happens only
/// when transcripts are pasted) are under-counted by one. Acceptable
/// approximation given the alternative (iterate via `match_indices`)
/// adds parser complexity for a vanishingly rare case.
fn extract_command_identifier(s: &str) -> Option<String> {
    let name_open = s.find("<command-name>")?;
    let name_content_start = name_open + "<command-name>".len();
    let name_close_offset = s[name_content_start..].find("</command-name>")?;
    let name = &s[name_content_start..name_content_start + name_close_offset];
    let stripped = name.strip_prefix('/').unwrap_or(name);
    if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    }
}

/// Pull the skill directory name from a `Base directory for this
/// skill: <path>` text block. The skill-name `ContextFile::identifier`
/// returns is the parent directory of `SKILL.md`; the path the harness
/// emits points at that same directory (no trailing `/SKILL.md`), so
/// `Path::file_name()` on the trimmed path is the matching identifier.
fn extract_skill_identifier(text: &str) -> Option<String> {
    const PREFIX: &str = "Base directory for this skill: ";
    let after = text.strip_prefix(PREFIX)?;
    // Path may be followed by whitespace / newlines / additional
    // body text — take the first whitespace-delimited token.
    let path_str = after.split_whitespace().next()?;
    Path::new(path_str)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
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
/// descending by `total_loads()`, then descending by
/// `estimated_tokens_billed`, then ascending by `file.path`. Four-key
/// sort ensures full determinism even when cost ties (e.g. multiple
/// rows with the same per-tier cost but different load counts) or
/// when costs collapse to `None`.
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
            .then_with(|| b.total_loads().cmp(&a.total_loads()))
            .then_with(|| b.estimated_tokens_billed.cmp(&a.estimated_tokens_billed))
            .then_with(|| a.file.path.cmp(&b.file.path))
    });
    rows
}

// `loads_1h` / `loads_5m` are the domain-essential per-tier counters
// — silencing `similar_names` here is a deliberate readability
// tradeoff (mirroring `session_meta_from_turns`'s same exemption).
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn build_row(
    file: ContextFile,
    session_metas: &[SessionMeta],
    catalog: &PricingCatalog,
) -> AttributionRow {
    let mut loads_1h: u64 = 0;
    let mut loads_5m: u64 = 0;
    let mut total_cost: Option<f64> = Some(0.0);

    for meta in session_metas {
        let Some(meta_cwd) = meta.cwd.as_deref() else {
            continue;
        };
        if !file.scope.matches(meta_cwd) {
            continue;
        }
        let (n_loads, tier) = evidence_for_file_in_session(&file, meta);
        if n_loads == 0 {
            continue;
        }
        match tier {
            CacheTier::Long1h => loads_1h += n_loads,
            CacheTier::Short5m => loads_5m += n_loads,
        }
        if total_cost.is_none() {
            // Already collapsed; keep accumulating loads but skip
            // the price lookup.
            continue;
        }
        let per_load = match tier {
            CacheTier::Long1h => {
                catalog.cost_for_cache_creation_1h(file.tokens, meta.model.as_deref())
            }
            CacheTier::Short5m => {
                catalog.cost_for_cache_creation_5m(file.tokens, meta.model.as_deref())
            }
        };
        match per_load {
            Some(c) => {
                total_cost = total_cost.map(|t| t + c * n_loads as f64);
            }
            None => {
                total_cost = None;
            }
        }
    }

    let estimated_tokens_billed = file.tokens.saturating_mul(loads_1h + loads_5m);
    AttributionRow {
        file,
        loads_1h,
        loads_5m,
        estimated_tokens_billed,
        attributed_cost: total_cost,
    }
}

/// Decide how many loads the given session contributes for the given
/// file, and at which tier.
///
/// Dispatch by `(file.kind, session.kind)`:
/// - **Always-loaded kinds** (CLAUDE.md, rules) load exactly once per
///   in-scope session at the session's `primary_tier`. Both parent
///   and subagent sessions count.
/// - **Agent kinds** are credited only when the session is a
///   `Subagent { agent_type }` whose payload matches
///   `file.identifier()`. Parent sessions never load agent files
///   directly. The match contributes one load at the session's
///   `primary_tier` (the harness loads the agent body alongside the
///   subagent's system prompt).
/// - **Skill / Command kinds** are credited per matching
///   `OnDemandLoad` entry in the session's `on_demand_loads` vec —
///   the count of injections / invocations directly observed in the
///   JSONL stream. The tier comes from the matched load's `tier`
///   field (always `Short5m` per the extractor's convention).
///
/// Scope matching is the caller's responsibility (handled by the
/// outer loop in `build_row` before this dispatch fires).
fn evidence_for_file_in_session(file: &ContextFile, session: &SessionMeta) -> (u64, CacheTier) {
    match &file.kind {
        ContextFileKind::GlobalClaudeMd
        | ContextFileKind::ProjectClaudeMd
        | ContextFileKind::UserRule
        | ContextFileKind::PluginRule { .. }
        | ContextFileKind::ProjectLocalRule => match session.observed_tier() {
            Some(tier) => (1, tier),
            // Session emitted zero `cache_creation` activity — nothing
            // actually loaded, so don't pretend always-loaded files
            // were billed. The tier value is a don't-care since
            // `n_loads == 0` short-circuits the caller in `build_row`.
            None => (0, CacheTier::Long1h),
        },
        ContextFileKind::UserAgent
        | ContextFileKind::PluginAgent { .. }
        | ContextFileKind::ProjectLocalAgent => match (&session.kind, session.observed_tier()) {
            (SessionKind::Subagent { agent_type }, Some(tier))
                if file.identifier().as_deref() == Some(agent_type.as_str()) =>
            {
                (1, tier)
            }
            _ => (0, CacheTier::Long1h),
        },
        ContextFileKind::UserSkill
        | ContextFileKind::PluginSkill { .. }
        | ContextFileKind::ProjectLocalSkill => {
            count_on_demand_matches(file, session, OnDemandKind::Skill)
        }
        ContextFileKind::ProjectLocalCommand => {
            count_on_demand_matches(file, session, OnDemandKind::Command)
        }
    }
}

/// Count `OnDemandLoad` entries that match `file.identifier()` and
/// `expected_kind`. Returns `(count, tier)` where `tier` is the first
/// match's `tier` (always `Short5m` in practice; abstracted here so
/// the dispatch shape stays uniform). When there's no match returns
/// `(0, Short5m)` — the tier is a don't-care because `n_loads == 0`
/// short-circuits the caller.
fn count_on_demand_matches(
    file: &ContextFile,
    session: &SessionMeta,
    expected_kind: OnDemandKind,
) -> (u64, CacheTier) {
    let Some(identifier) = file.identifier() else {
        return (0, CacheTier::Short5m);
    };
    let mut count: u64 = 0;
    let mut first_tier: Option<CacheTier> = None;
    for load in &session.on_demand_loads {
        if load.kind != expected_kind {
            continue;
        }
        if load.identifier != identifier {
            continue;
        }
        count += 1;
        if first_tier.is_none() {
            first_tier = Some(load.tier);
        }
    }
    (count, first_tier.unwrap_or(CacheTier::Short5m))
}

/// Per-tier coverage: `observed` summed from every session meta
/// (parent + subagent), `attributed` summed from the row set's per-
/// tier load counts × tokens.
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
        long_attributed =
            long_attributed.saturating_add(row.loads_1h.saturating_mul(row.file.tokens));
        short_attributed =
            short_attributed.saturating_add(row.loads_5m.saturating_mul(row.file.tokens));
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
    use crate::domain::{CacheCreation, TurnOrigin, Usage};
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
            origin: TurnOrigin::default(),
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
            origin: TurnOrigin::default(),
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
            origin: TurnOrigin::default(),
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
        kind: SessionKind,
        id: &str,
        cwd: Option<&Path>,
        project: &str,
        started_at_str: &str,
        model: Option<&str>,
        primary_tier: CacheTier,
        observed_1h: u64,
        observed_5m: u64,
        on_demand_loads: Vec<OnDemandLoad>,
    ) -> SessionMeta {
        SessionMeta {
            id: id.to_string(),
            kind,
            cwd: cwd.map(Path::to_path_buf),
            project_short_name: project.to_string(),
            started_at: ts(started_at_str),
            model: model.map(str::to_string),
            primary_tier,
            observed_1h_tokens: observed_1h,
            observed_5m_tokens: observed_5m,
            on_demand_loads,
        }
    }

    /// Convenience constructor for an `OnDemandLoad`. Tier is always
    /// `Short5m` per the production extractor's convention.
    fn skill_load(identifier: &str) -> OnDemandLoad {
        OnDemandLoad {
            kind: OnDemandKind::Skill,
            identifier: identifier.to_string(),
            tier: CacheTier::Short5m,
        }
    }

    fn command_load(identifier: &str) -> OnDemandLoad {
        OnDemandLoad {
            kind: OnDemandKind::Command,
            identifier: identifier.to_string(),
            tier: CacheTier::Short5m,
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
    fn session_meta_records_observed_tokens_per_tier() {
        // Per-tier observed-token sums — coverage's denominator side.
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("claude-opus-4-7"), 0, 100),
            assistant_turn("2026-04-01T10:01:00Z", Some("claude-opus-4-7"), 100, 0),
            assistant_turn("2026-04-01T10:02:00Z", Some("claude-opus-4-7"), 50, 50),
            assistant_turn_no_usage("2026-04-01T10:03:00Z", Some("claude-opus-4-7")),
        ];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "session-id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.observed_1h_tokens, 150);
        assert_eq!(meta.observed_5m_tokens, 150);
    }

    #[test]
    fn session_meta_primary_tier_from_first_cache_creation_event() {
        // First assistant turn with cache_creation > 0 sets the
        // primary tier. A turn that only carries 5m → primary = 5m.
        let turns = vec![
            assistant_turn_no_usage("2026-04-01T09:59:00Z", Some("claude-opus-4-7")),
            assistant_turn("2026-04-01T10:00:00Z", Some("claude-opus-4-7"), 200, 0),
            assistant_turn("2026-04-01T10:01:00Z", Some("claude-opus-4-7"), 0, 100),
        ];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.primary_tier, CacheTier::Short5m);
    }

    #[test]
    fn session_meta_primary_tier_prefers_1h_on_mixed_first_event() {
        // First cache_creation event fires both tiers — 1h wins
        // because that's where always-loaded content lives.
        let turns = vec![assistant_turn(
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            50,
            100,
        )];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.primary_tier, CacheTier::Long1h);
    }

    #[test]
    fn session_meta_primary_tier_falls_back_to_1h_on_no_cache_activity() {
        // No cache_creation activity at all — primary tier is still
        // well-defined (defaults to 1h). Doesn't matter for
        // attribution because no loads will be credited anyway.
        let turns = vec![assistant_turn(
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            0,
            0,
        )];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.primary_tier, CacheTier::Long1h);
    }

    #[test]
    fn session_meta_extracts_skill_load_from_injection() {
        // User turn with array content carrying a "Base directory for
        // this skill: <path>" text block → one OnDemandLoad::Skill
        // identified by the path's last directory name.
        let mut user = user_turn("2026-04-01T10:00:00Z", None);
        user.content = Some(serde_json::json!([
            { "type": "text", "text": "Base directory for this skill: /Users/x/.claude/skills/tw-implement-plan\n\nbody" },
        ]));
        let turns = vec![
            user,
            assistant_turn("2026-04-01T10:00:05Z", Some("claude-opus-4-7"), 100, 0),
        ];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.on_demand_loads.len(), 1);
        let load = &meta.on_demand_loads[0];
        assert_eq!(load.kind, OnDemandKind::Skill);
        assert_eq!(load.identifier, "tw-implement-plan");
        assert_eq!(load.tier, CacheTier::Short5m);
    }

    #[test]
    fn session_meta_extracts_command_load_from_slash_tag() {
        // String content matching <command-name>/foo</command-name>
        // → one OnDemandLoad::Command identified by `foo` (leading
        // slash stripped to match the inventory file stem).
        let mut user = user_turn("2026-04-01T10:00:00Z", None);
        user.content = Some(Value::String(
            "<command-name>/tw-commit</command-name>\n<command-args>main</command-args>"
                .to_string(),
        ));
        let turns = vec![
            user,
            assistant_turn("2026-04-01T10:00:05Z", Some("claude-opus-4-7"), 100, 0),
        ];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.on_demand_loads.len(), 1);
        let load = &meta.on_demand_loads[0];
        assert_eq!(load.kind, OnDemandKind::Command);
        assert_eq!(load.identifier, "tw-commit");
    }

    #[test]
    fn session_meta_extracts_repeated_skill_load_per_injection() {
        // Same skill injected twice → two OnDemandLoad entries (per-
        // load count, not per-identifier dedup).
        let mut u1 = user_turn("2026-04-01T10:00:00Z", None);
        u1.content = Some(serde_json::json!([
            { "type": "text", "text": "Base directory for this skill: /x/.claude/skills/foo" },
        ]));
        let mut u2 = user_turn("2026-04-01T10:02:00Z", None);
        u2.content = Some(serde_json::json!([
            { "type": "text", "text": "Base directory for this skill: /x/.claude/skills/foo" },
        ]));
        let turns = vec![
            u1,
            assistant_turn("2026-04-01T10:00:05Z", Some("claude-opus-4-7"), 100, 0),
            u2,
            assistant_turn("2026-04-01T10:02:05Z", Some("claude-opus-4-7"), 100, 0),
        ];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        let foo_count = meta
            .on_demand_loads
            .iter()
            .filter(|l| l.kind == OnDemandKind::Skill && l.identifier == "foo")
            .count();
        assert_eq!(foo_count, 2);
    }

    #[test]
    fn session_meta_picks_majority_model() {
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("model-a"), 100, 0),
            assistant_turn("2026-04-01T10:01:00Z", Some("model-a"), 100, 0),
            assistant_turn("2026-04-01T10:02:00Z", Some("model-b"), 100, 0),
        ];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.model.as_deref(), Some("model-a"));
    }

    #[test]
    fn session_meta_breaks_ties_by_first_occurrence() {
        // Two models, equal counts. First-seen must win.
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("model-x"), 100, 0),
            assistant_turn("2026-04-01T10:01:00Z", Some("model-y"), 100, 0),
        ];
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
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
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        assert_eq!(meta.cwd, Some(PathBuf::from("/x")));
        assert_eq!(meta.project_short_name, "x");
    }

    #[test]
    fn session_meta_returns_none_for_no_timestamps() {
        // Empty turns → None.
        let empty: Vec<Turn> = Vec::new();
        assert!(
            session_meta_from_turns(
                SessionKind::Parent,
                "id".to_string(),
                Path::new("/p"),
                &empty
            )
            .is_none()
        );

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
            origin: TurnOrigin::default(),
        }];
        assert!(
            session_meta_from_turns(
                SessionKind::Parent,
                "id".to_string(),
                Path::new("/p"),
                &untimestamped
            )
            .is_none()
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
        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "id".to_string(),
            Path::new("/some/proj-name"),
            &turns,
        )
        .unwrap();
        assert!(meta.cwd.is_none());
        assert_eq!(meta.project_short_name, "proj-name");
    }

    // --- compute_rows: per-load attribution ---

    #[test]
    fn compute_rows_credits_claude_md_once_per_session() {
        // CLAUDE.md is always-loaded, so each in-scope session
        // contributes exactly one load at the session's primary tier
        // — never multiplied by per-session assistant-turn count.
        // Two parent sessions on 1h tier → loads_1h = 2.
        let inv = vec![ctx_file(
            "/g/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            Scope::Global,
        )];
        let metas = vec![
            meta_with(
                SessionKind::Parent,
                "s1",
                Some(Path::new("/a")),
                "a",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                300,
                0,
                Vec::new(),
            ),
            meta_with(
                SessionKind::Parent,
                "s2",
                Some(Path::new("/b")),
                "b",
                "2026-04-01T11:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                500,
                0,
                Vec::new(),
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].loads_1h, 2);
        assert_eq!(rows[0].loads_5m, 0);
        assert_eq!(rows[0].estimated_tokens_billed, 200);
    }

    #[test]
    fn compute_rows_credits_skill_per_on_demand_load() {
        // A skill is credited once per matching OnDemandLoad. Three
        // injections in a single session → loads_5m = 3.
        let inv = vec![ctx_file(
            "/g/skills/foo/SKILL.md",
            ContextFileKind::UserSkill,
            100,
            Scope::Global,
        )];
        let metas = vec![meta_with(
            SessionKind::Parent,
            "s1",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Long1h,
            0,
            300,
            vec![skill_load("foo"), skill_load("foo"), skill_load("foo")],
        )];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].loads_5m, 3);
        assert_eq!(rows[0].loads_1h, 0);
        assert_eq!(rows[0].estimated_tokens_billed, 300);
    }

    #[test]
    fn compute_rows_credits_command_per_invocation() {
        // Same per-load shape, but for a project-local command — the
        // identifier comes from the file stem (`commands/foo.md` →
        // `foo`).
        let inv = vec![ctx_file(
            "/proj/.claude/commands/foo.md",
            ContextFileKind::ProjectLocalCommand,
            50,
            Scope::CwdSubtree {
                root: PathBuf::from("/proj"),
            },
        )];
        let metas = vec![meta_with(
            SessionKind::Parent,
            "s1",
            Some(Path::new("/proj")),
            "proj",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Long1h,
            0,
            100,
            vec![command_load("foo"), command_load("foo")],
        )];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(rows[0].loads_5m, 2);
    }

    #[test]
    fn compute_rows_credits_agent_only_for_matching_subagent() {
        // The bug-report regression: `tw-plan-reviewer.md` was
        // crediting parent-session events. With per-load math, an
        // agent file is credited only by a Subagent meta whose
        // agent_type matches the file's identifier.
        let inv = vec![
            ctx_file(
                "/g/agents/tw-code-reviewer.md",
                ContextFileKind::UserAgent,
                15,
                Scope::Global,
            ),
            ctx_file(
                "/g/agents/tw-plan-reviewer.md",
                ContextFileKind::UserAgent,
                17,
                Scope::Global,
            ),
        ];
        let metas = vec![
            // Parent never loads agent files — must contribute zero
            // loads regardless of its primary tier.
            meta_with(
                SessionKind::Parent,
                "parent",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                500,
                0,
                Vec::new(),
            ),
            meta_with(
                SessionKind::Subagent {
                    agent_type: "tw-code-reviewer".to_string(),
                },
                "agent-1",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T10:01:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Short5m,
                0,
                100,
                Vec::new(),
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        let reviewer = rows
            .iter()
            .find(|r| r.file.path.ends_with("tw-code-reviewer.md"))
            .unwrap();
        let plan_reviewer = rows
            .iter()
            .find(|r| r.file.path.ends_with("tw-plan-reviewer.md"))
            .unwrap();
        assert_eq!(
            reviewer.loads_5m, 1,
            "matching subagent must credit the agent file once"
        );
        assert_eq!(reviewer.loads_1h, 0);
        assert_eq!(
            plan_reviewer.total_loads(),
            0,
            "non-matching agent file must NOT be credited (the bug-report regression)",
        );
    }

    #[test]
    fn compute_rows_attributes_subtree_file_only_to_in_scope_sessions() {
        // Out-of-scope sessions still don't contribute, even with the
        // per-load math.
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
                SessionKind::Parent,
                "in-scope",
                Some(Path::new("/proj/sub")),
                "sub",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                300,
                0,
                Vec::new(),
            ),
            meta_with(
                SessionKind::Parent,
                "out-of-scope",
                Some(Path::new("/other")),
                "other",
                "2026-04-01T11:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                500,
                0,
                Vec::new(),
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(
            rows[0].loads_1h, 1,
            "out-of-scope session must not contribute",
        );
    }

    #[test]
    fn compute_rows_prices_skill_at_5m_rate_not_1h() {
        let inv = vec![ctx_file(
            "/g/skills/x/SKILL.md",
            ContextFileKind::UserSkill,
            1000,
            Scope::Global,
        )];
        let metas = vec![meta_with(
            SessionKind::Parent,
            "s1",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Long1h,
            0,
            1000,
            vec![skill_load("x")],
        )];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        // 1000 tokens × 1 load × 3.75e-6 (5m rate) = 0.00375.
        let cost = rows[0].attributed_cost.expect("priced");
        assert!(
            (cost - 0.003_75).abs() < 1e-12,
            "got {cost}, expected 5m rate"
        );
    }

    #[test]
    fn compute_rows_collapses_cost_to_none_on_unknown_model() {
        // One contributing session has an unknown model → row's cost
        // collapses to None. Loads still count fully (the bug-report
        // session's CLAUDE.md should still report its real load count
        // even when the cost can't be priced).
        let inv = vec![ctx_file(
            "/g/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            Scope::Global,
        )];
        let metas = vec![
            meta_with(
                SessionKind::Parent,
                "known",
                Some(Path::new("/a")),
                "a",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                300,
                0,
                Vec::new(),
            ),
            meta_with(
                SessionKind::Parent,
                "unknown",
                Some(Path::new("/b")),
                "b",
                "2026-04-01T11:00:00Z",
                Some("claude-fake-9-9"),
                CacheTier::Long1h,
                500,
                0,
                Vec::new(),
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(
            rows[0].total_loads(),
            2,
            "loads count both sessions even when one collapses cost",
        );
        assert!(
            rows[0].attributed_cost.is_none(),
            "unknown-model session must collapse cost"
        );
    }

    #[test]
    fn compute_rows_mixed_tier_row_attributes_at_both_tiers() {
        // CLAUDE.md loaded by a 1h-tier parent session AND a 5m-tier
        // subagent session — `loads_1h` AND `loads_5m` both > 0,
        // tier_label() = `1h+5m`, cost = sum across tiers.
        let inv = vec![ctx_file(
            "/g/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            Scope::Global,
        )];
        let metas = vec![
            meta_with(
                SessionKind::Parent,
                "parent",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                500,
                0,
                Vec::new(),
            ),
            meta_with(
                SessionKind::Subagent {
                    agent_type: "tw-code-reviewer".to_string(),
                },
                "subagent",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T10:01:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Short5m,
                0,
                200,
                Vec::new(),
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        assert_eq!(rows[0].loads_1h, 1);
        assert_eq!(rows[0].loads_5m, 1);
        assert_eq!(rows[0].tier_label(), "1h+5m");
        // Cost = 100 × 1 × rate_1h + 100 × 1 × rate_5m
        //      = 100 × 6e-6        + 100 × 3.75e-6
        //      = 0.0006 + 0.000375 = 0.000975.
        let cost = rows[0].attributed_cost.expect("priced");
        assert!(
            (cost - 0.000_975).abs() < 1e-12,
            "got {cost}, expected sum across tiers",
        );
    }

    #[test]
    fn compute_rows_sorts_descending_by_cost_then_loads_then_tokens_then_path() {
        // Four-key sort. Two rows with equal cost ($0) but different
        // load counts must order the larger-loads row first; two
        // rows with equal cost AND equal loads go by tokens then
        // path.
        let inv = vec![
            ctx_file(
                "/aaa/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                200,
                Scope::Global,
            ),
            ctx_file(
                "/bbb/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                100,
                Scope::Global,
            ),
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
            SessionKind::Parent,
            "s1",
            Some(Path::new("/scope")),
            "scope",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Long1h,
            200,
            0,
            Vec::new(),
        )];
        for _ in 0..5 {
            let rows = compute_rows(inv.clone(), &metas, &sample_catalog());
            assert_eq!(rows[0].file.path, PathBuf::from("/aaa/CLAUDE.md"));
            assert_eq!(rows[1].file.path, PathBuf::from("/bbb/CLAUDE.md"));
            assert_eq!(rows[2].file.path, PathBuf::from("/ccc/proj/CLAUDE.md"));
        }
    }

    #[test]
    fn compute_rows_skips_session_with_zero_cache_activity() {
        // Regression guard: a session that never emitted a single
        // `cache_creation` event (e.g., a transient session opened
        // and closed without firing the LLM) must NOT be credited
        // with implicit CLAUDE.md / rule loads. The pre-fix code
        // returned `(1, primary_tier)` unconditionally — even for
        // observed_*_tokens == (0, 0) — silently inflating both row
        // load counts and coverage's attributed numerator.
        let inv = vec![
            ctx_file(
                "/g/CLAUDE.md",
                ContextFileKind::GlobalClaudeMd,
                100,
                Scope::Global,
            ),
            ctx_file(
                "/g/rules/r1.md",
                ContextFileKind::UserRule,
                50,
                Scope::Global,
            ),
        ];
        // Two sessions in scope: one with real cache activity, one
        // with zero. Only the active one should credit loads.
        let metas = vec![
            meta_with(
                SessionKind::Parent,
                "active",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                500, // observed_1h > 0 → observed_tier() = Some(Long1h)
                0,
                Vec::new(),
            ),
            meta_with(
                SessionKind::Parent,
                "idle",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T11:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                0, // observed_1h == 0
                0, // observed_5m == 0 → observed_tier() = None
                Vec::new(),
            ),
        ];
        let rows = compute_rows(inv, &metas, &sample_catalog());
        let claude_md = rows
            .iter()
            .find(|r| r.file.path.ends_with("CLAUDE.md"))
            .unwrap();
        let rule = rows
            .iter()
            .find(|r| r.file.path.ends_with("r1.md"))
            .unwrap();
        assert_eq!(
            claude_md.total_loads(),
            1,
            "zero-activity session must not credit always-loaded files",
        );
        assert_eq!(
            rule.total_loads(),
            1,
            "zero-activity session must not credit always-loaded files",
        );
    }

    #[test]
    fn session_meta_observed_tier_returns_none_for_zero_activity() {
        let m = meta_with(
            SessionKind::Parent,
            "idle",
            Some(Path::new("/proj")),
            "proj",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
        );
        assert!(m.observed_tier().is_none());
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn session_meta_observed_tier_returns_some_when_either_tier_observed() {
        // Either side non-zero → observed_tier returns Some(primary_tier).
        let only_1h = meta_with(
            SessionKind::Parent,
            "s",
            Some(Path::new("/p")),
            "p",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Long1h,
            100,
            0,
            Vec::new(),
        );
        assert_eq!(only_1h.observed_tier(), Some(CacheTier::Long1h));

        let only_5m = meta_with(
            SessionKind::Parent,
            "s",
            Some(Path::new("/p")),
            "p",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Short5m,
            0,
            100,
            Vec::new(),
        );
        assert_eq!(only_5m.observed_tier(), Some(CacheTier::Short5m));
    }

    #[test]
    fn compute_rows_includes_zero_load_in_scope_files() {
        let inv = vec![ctx_file(
            "/g/CLAUDE.md",
            ContextFileKind::GlobalClaudeMd,
            100,
            Scope::Global,
        )];
        // No sessions → zero loads.
        let rows = compute_rows(inv, &[], &sample_catalog());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].total_loads(), 0);
        assert_eq!(rows[0].estimated_tokens_billed, 0);
        assert_eq!(rows[0].attributed_cost, Some(0.0));
        assert_eq!(rows[0].tier_label(), "—");
    }

    #[test]
    fn session_meta_from_turns_propagates_subagent_kind() {
        let turns = vec![assistant_turn(
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            0,
            100,
        )];
        let meta = session_meta_from_turns(
            SessionKind::Subagent {
                agent_type: "tw-code-reviewer".to_string(),
            },
            "agent-abc".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
        match meta.kind {
            SessionKind::Subagent { agent_type } => {
                assert_eq!(agent_type, "tw-code-reviewer");
            }
            SessionKind::Parent => panic!("expected Subagent kind, got Parent"),
        }
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
            SessionKind::Parent,
            "s1",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            Some("claude-opus-4-7"),
            CacheTier::Long1h,
            5_000, // observed_1h
            2_000, // observed_5m
            Vec::new(),
        )];
        // Hand-crafted rows with known per-tier load counts.
        let rows = vec![
            AttributionRow {
                file: ctx_file(
                    "/g/CLAUDE.md",
                    ContextFileKind::GlobalClaudeMd,
                    100,
                    Scope::Global,
                ),
                loads_1h: 40,
                loads_5m: 0,
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
                loads_1h: 0,
                loads_5m: 30,
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
    fn compute_coverage_includes_subagent_observed_tokens() {
        // Phase 2 contract: subagent observed tokens DO enter the
        // denominator (Phase 1 was a temporary gate to keep the
        // snapshot stable during the structural refactor).
        let metas = vec![
            meta_with(
                SessionKind::Parent,
                "parent",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T10:00:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Long1h,
                500,
                0,
                Vec::new(),
            ),
            meta_with(
                SessionKind::Subagent {
                    agent_type: "tw-code-reviewer".to_string(),
                },
                "subagent",
                Some(Path::new("/proj")),
                "proj",
                "2026-04-01T10:01:00Z",
                Some("claude-opus-4-7"),
                CacheTier::Short5m,
                0,
                200,
                Vec::new(),
            ),
        ];
        let cov = compute_coverage(&metas, &[]);
        assert_eq!(cov.long_1h.observed_tokens, 500);
        assert_eq!(
            cov.short_5m.observed_tokens, 200,
            "subagent's 5m observed tokens must enter the denominator",
        );
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn compute_coverage_observed_sums_match_independent_per_tier_sum() {
        // Fixed-point invariant for the observed-token side: build a
        // Vec<Turn> with deterministic per-tier contributions, sum
        // them by hand, then run the same turns through
        // session_meta_from_turns + compute_coverage and assert the
        // reported observed_*_tokens match.
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
        let expected_1h: u64 = 100 + 250 + 75;
        let expected_5m: u64 = 50 + 200 + 100;
        assert_eq!(expected_1h, 425);
        assert_eq!(expected_5m, 350);

        let meta = session_meta_from_turns(
            SessionKind::Parent,
            "s1".to_string(),
            Path::new("/proj"),
            &turns,
        )
        .unwrap();
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
            SessionKind::Parent,
            "target",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
        );
        let other = meta_with(
            SessionKind::Parent,
            "other",
            Some(Path::new("/a")),
            "a",
            "2026-04-01T10:00:00Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
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
            SessionKind::Parent,
            "s1",
            Some(Path::new("/a")),
            "a",
            "2026-04-10T00:00:00Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
        );
        let on_until = meta_with(
            SessionKind::Parent,
            "s2",
            Some(Path::new("/a")),
            "a",
            "2026-04-20T00:00:00Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
        );
        let before = meta_with(
            SessionKind::Parent,
            "s3",
            Some(Path::new("/a")),
            "a",
            "2026-04-09T23:59:59Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
        );
        let after = meta_with(
            SessionKind::Parent,
            "s4",
            Some(Path::new("/a")),
            "a",
            "2026-04-20T00:00:01Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
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
            SessionKind::Parent,
            "s1",
            Some(Path::new("/a")),
            "foo",
            "2026-04-01T10:00:00Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
        );
        let bar = meta_with(
            SessionKind::Parent,
            "s2",
            Some(Path::new("/a")),
            "bar",
            "2026-04-01T10:00:00Z",
            None,
            CacheTier::Long1h,
            0,
            0,
            Vec::new(),
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
