//! Per-dispatch subagent spend, grouped into the rows the `agents`
//! view ranks.
//!
//! Public API:
//! - `Dispatch` — one subagent transcript, deduplicated and priced.
//! - `Pinning` — how an agent file constrains the model a dispatch
//!   runs on; part of the row key, so a disagreement between two
//!   dispatches of one agent splits them into two rows.
//! - `PinningKind` — `Pinning` without its payload, so two `Pinned`
//!   values naming different models filter identically.
//! - `PinningFilter` — the accepted `PinningKind`s, defaulting to the
//!   rows for which no agent file names a concrete model.
//! - `AgentRow` — one table row, keyed by agent type × model × effort
//!   × pinning.
//! - `dispatch_from_turns(...)` — fold one subagent transcript into a
//!   `Dispatch`, priced at its own recorded model.
//! - `resolve_pinning(...)` — classify one dispatch against the
//!   inventory and a frontmatter map. Pure; reads no files.
//! - `group_dispatches(...)` — accumulate dispatches into rows.
//! - `sort_rows(...)` — priced rows first, by cost descending.
//! - `reprice(...)` — price a recorded token bundle at another
//!   model's rates.
//! - `RepricedDelta` / `repriced_delta(...)` — what a repricing
//!   footer reports, including the rows it declined to count.
//!
//! Nothing here reads `attribution`: that module answers "which
//! context files did this session load, and at what tier", and this
//! one answers "what did this agent's dispatches cost". They share
//! input data and share no output.

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::aggregation::majority_assistant_value;
use crate::discovery::SubagentMeta;
use crate::domain::{CacheCreation, CostBreakdown, Role, Turn, Usage};
use crate::filter::{FilterComponent, HonoredBy};
use crate::inventory::{AgentFrontmatter, ContextFile, ContextFileKind, Scope};
use crate::pricing::PricingCatalog;

// ---- core types ----

/// One subagent transcript, deduplicated and priced.
///
/// `model` and `effort` are majority votes across the transcript's
/// assistant turns rather than first-seen values: a run that switches
/// mid-flight is reported as the model it mostly ran on.
#[derive(Debug)]
pub struct Dispatch {
    pub agent_type: String,
    pub is_fork: bool,
    pub cwd: Option<PathBuf>,
    pub project_short_name: String,
    pub started_at: DateTime<Utc>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub usage: Usage,
    pub cost: Option<CostBreakdown>,
}

/// How an agent file constrains the model a dispatch runs on.
///
/// `NoAgentFile` is distinguished from `Unpinned` because the evidence
/// differs in kind: one is a positive reading of a file that names no
/// model, the other is the absence of anything to read.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pinning {
    /// Frontmatter names a concrete model — an alias or a full ID.
    Pinned { declared: String },
    /// Frontmatter carries the `inherit` sentinel.
    Inherit,
    /// An agent file was found and names no model.
    Unpinned,
    /// No agent file was found for this dispatch. Either the agent is
    /// built into the harness, or its file is not installed here — the
    /// two are indistinguishable on disk, so neither is asserted.
    NoAgentFile,
    /// A fork. Structurally runs the parent's model, and the harness
    /// spawns it without an agent definition to read.
    Fork,
}

impl Pinning {
    /// The rendered cell text for this classification.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pinned { .. } => "pinned",
            Self::Inherit => "inherit",
            Self::Unpinned => "unpinned",
            Self::NoAgentFile => "no-agent-file",
            Self::Fork => "fork",
        }
    }

    /// Whether an agent file named a concrete model for this dispatch.
    ///
    /// This is the predicate separating the default filter slice from
    /// `Pinned`. Both the row key and `PinningFilter` read it rather
    /// than re-deriving the rule, so widening the notion of "pinned"
    /// later moves one definition instead of two.
    #[must_use]
    pub fn names_a_concrete_model(&self) -> bool {
        matches!(self, Self::Pinned { .. })
    }

    /// This classification without its payload.
    #[must_use]
    pub fn kind(&self) -> PinningKind {
        match self {
            Self::Pinned { .. } => PinningKind::Pinned,
            Self::Inherit => PinningKind::Inherit,
            Self::Unpinned => PinningKind::Unpinned,
            Self::NoAgentFile => PinningKind::NoAgentFile,
            Self::Fork => PinningKind::Fork,
        }
    }
}

/// `Pinning` without its payload. `PinningFilter` holds these so two
/// `Pinned` values naming different models filter identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinningKind {
    Pinned,
    Inherit,
    Unpinned,
    NoAgentFile,
    Fork,
}

impl PinningKind {
    /// Every kind, in the order `--pinning` lists them.
    pub const ALL: [Self; 5] = [
        Self::Pinned,
        Self::Inherit,
        Self::Unpinned,
        Self::NoAgentFile,
        Self::Fork,
    ];

    /// The flag-value spelling of this kind, shared by `--pinning`'s
    /// parser and every surface that names the active slice.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Inherit => "inherit",
            Self::Unpinned => "unpinned",
            Self::NoAgentFile => "no-agent-file",
            Self::Fork => "fork",
        }
    }
}

/// One row of the agents view.
///
/// `cost` is `None` when any contributing dispatch priced to `None` —
/// strict propagation, never a partial sum presented as a total. The
/// token figures still reflect every dispatch.
#[derive(Debug, Serialize)]
pub struct AgentRow {
    pub agent_type: String,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub declared_effort: Option<String>,
    pub pinning: Pinning,
    pub dispatches: u64,
    pub usage: Usage,
    pub cost: Option<CostBreakdown>,
}

// ---- pinning filter ----

/// Which pinning classifications the agents view admits.
#[derive(Debug, Clone, PartialEq)]
pub struct PinningFilter {
    accepted: Vec<PinningKind>,
}

impl Default for PinningFilter {
    /// The rows for which no agent file names a concrete model — the
    /// set the view was built to size.
    fn default() -> Self {
        Self::new(&[
            PinningKind::Unpinned,
            PinningKind::Inherit,
            PinningKind::NoAgentFile,
        ])
    }
}

impl PinningFilter {
    /// A filter admitting exactly the kinds in `accepted`.
    ///
    /// Normalizes to `PinningKind::ALL` order and drops duplicates, so
    /// equality is set-equality. `describe_active` keys off equality
    /// with the default, and a caller spelling the default set in
    /// another order must not thereby report itself as filtered.
    #[must_use]
    pub fn new(accepted: &[PinningKind]) -> Self {
        Self {
            accepted: PinningKind::ALL
                .into_iter()
                .filter(|k| accepted.contains(k))
                .collect(),
        }
    }

    /// A filter admitting every kind.
    #[must_use]
    pub fn everything() -> Self {
        Self::new(&PinningKind::ALL)
    }

    /// The kinds this filter admits, in the order it holds them.
    #[must_use]
    pub fn accepted(&self) -> &[PinningKind] {
        &self.accepted
    }

    /// Whether this filter admits `pinning`.
    #[must_use]
    pub fn accepts(&self, pinning: &Pinning) -> bool {
        self.accepted.contains(&pinning.kind())
    }

    /// The slice this filter covers, spelled as `--pinning` would.
    /// Named unconditionally by the agents empty state and footer,
    /// because `describe_active` deliberately says nothing at the
    /// default and those are the only surfaces left to say it.
    #[must_use]
    pub fn describe_slice(&self) -> String {
        if self.accepted.is_empty() {
            return "nothing".to_string();
        }
        self.accepted
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The active `--pinning` component, or an empty vector when this
    /// filter equals the default.
    ///
    /// The comparison is against the default rather than against the
    /// full set on purpose. `Query::describe_active` returning an
    /// empty vector is how every surface distinguishes "filtered to
    /// nothing" from "nothing to show"; a component the default always
    /// emitted would make that vector never empty, retiring the
    /// distinction crate-wide.
    ///
    /// The default is a filter setting rather than a semantic —
    /// `--pinning pinned` asks what the deliberate choices cost, the
    /// full set asks what the whole roster costs. That rule leaves the
    /// agents empty state and footer as the only places a narrowing
    /// default is visible, which is why both name the slice
    /// unconditionally via `describe_slice`.
    #[must_use]
    pub fn describe_active(&self) -> Vec<FilterComponent> {
        if *self == Self::default() {
            return Vec::new();
        }
        vec![FilterComponent {
            text: format!("--pinning {}", self.describe_slice()),
            honored_by: HonoredBy::AGENTS_ONLY,
        }]
    }
}

// ---- per-dispatch fold ----

/// Fold one subagent transcript into a `Dispatch`, priced at its own
/// recorded model.
///
/// Returns `None` when no turn carries a timestamp, mirroring
/// `attribution::session_meta_from_turns`'s contract. `turns` is
/// expected to be deduplicated already — the caller owns that pass,
/// because the `seen` set's scope is what decides which spend counts.
///
/// A transcript recording no cwd inherits the parent's, so a subagent
/// can still match a project-local agent file.
#[must_use]
pub fn dispatch_from_turns(
    meta: &SubagentMeta,
    parent_cwd: Option<&Path>,
    parent_short_name: &str,
    turns: &[Turn],
    catalog: &PricingCatalog,
) -> Option<Dispatch> {
    let started_at = turns.iter().filter_map(|t| t.timestamp).min()?;
    let cwd = turns
        .iter()
        .find_map(|t| t.cwd.clone())
        .or_else(|| parent_cwd.map(Path::to_path_buf));
    let project_short_name = turns
        .iter()
        .find_map(|t| t.cwd.as_ref())
        .and_then(|c| c.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| parent_short_name.to_string());

    let model = majority_assistant_value(turns, |turn| turn.model.as_deref());
    let effort = majority_assistant_value(turns, |turn| turn.effort.as_deref());

    let mut usage = Usage {
        input: 0,
        output: 0,
        cache_creation: CacheCreation::default(),
        cache_read: 0,
    };
    for turn in turns {
        match &turn.role {
            Role::Assistant => {}
            Role::User | Role::Attachment | Role::System | Role::Other(_) => continue,
        }
        let Some(u) = turn.usage.as_ref() else {
            continue;
        };
        usage.input += u.input;
        usage.output += u.output;
        usage.cache_creation.ephemeral_5m += u.cache_creation.ephemeral_5m;
        usage.cache_creation.ephemeral_1h += u.cache_creation.ephemeral_1h;
        usage.cache_read += u.cache_read;
    }

    let cost = catalog.cost_for_turn(&usage, model.as_deref());

    Some(Dispatch {
        agent_type: meta.agent_type.clone(),
        is_fork: meta.is_fork,
        cwd,
        project_short_name,
        started_at,
        model,
        effort,
        usage,
        cost,
    })
}

// ---- pinning resolution ----

/// Whether `kind` is one of the three agent kinds.
fn is_agent_kind(kind: &ContextFileKind) -> bool {
    matches!(
        kind,
        ContextFileKind::UserAgent
            | ContextFileKind::PluginAgent { .. }
            | ContextFileKind::ProjectLocalAgent
    )
}

/// Precedence rank of an agent kind, lower winning.
///
/// Project-local beats user-global beats plugin, per
/// <https://code.claude.com/docs/en/sub-agents.md#scope-priority-order>.
/// Non-agent kinds are excluded by `is_agent_kind` before ranking, so
/// their rank is unreachable and arbitrary.
fn agent_kind_rank(kind: &ContextFileKind) -> u8 {
    match kind {
        ContextFileKind::ProjectLocalAgent => 0,
        ContextFileKind::UserAgent => 1,
        ContextFileKind::PluginAgent { .. } => 2,
        ContextFileKind::GlobalClaudeMd
        | ContextFileKind::UserRule
        | ContextFileKind::UserSkill
        | ContextFileKind::PluginSkill { .. }
        | ContextFileKind::PluginRule { .. }
        | ContextFileKind::ProjectClaudeMd
        | ContextFileKind::ProjectLocalSkill
        | ContextFileKind::ProjectLocalCommand
        | ContextFileKind::ProjectLocalRule => u8::MAX,
    }
}

/// The inventory entry whose agent file defines `dispatch`, if any.
fn winning_agent_file<'a>(
    inventory: &'a [ContextFile],
    dispatch: &Dispatch,
) -> Option<&'a ContextFile> {
    let cwd = dispatch.cwd.as_deref();
    inventory
        .iter()
        .filter(|f| is_agent_kind(&f.kind))
        .filter(|f| f.identifier().as_deref() == Some(dispatch.agent_type.as_str()))
        // A dispatch that recorded no cwd can still be defined by a
        // globally-scoped file. Dropping those too would misfile
        // genuinely pinned spend as `NoAgentFile`, which the default
        // filter admits — inflating the very slice this view sizes.
        .filter(|f| cwd.map_or_else(|| matches!(f.scope, Scope::Global), |c| f.scope.matches(c)))
        .min_by(|a, b| {
            (agent_kind_rank(&a.kind), a.path.as_path())
                .cmp(&(agent_kind_rank(&b.kind), b.path.as_path()))
        })
}

/// Classify how an agent file constrains `dispatch`'s model, and read
/// the effort that same file declares.
///
/// One resolution feeds both, so the two columns can never disagree
/// about which file defined the dispatch. A fork short-circuits before
/// the inventory is consulted and therefore declares no effort either:
/// it has no agent definition, so a file that merely shares its name
/// says nothing about it.
fn classify<S: BuildHasher>(
    inventory: &[ContextFile],
    frontmatter: &HashMap<PathBuf, AgentFrontmatter, S>,
    dispatch: &Dispatch,
) -> (Pinning, Option<String>) {
    if dispatch.is_fork {
        return (Pinning::Fork, None);
    }
    let Some(file) = winning_agent_file(inventory, dispatch) else {
        return (Pinning::NoAgentFile, None);
    };
    let entry = frontmatter.get(&file.path);
    // An entry absent from the map reads as a file naming no model,
    // which is what an unreadable or block-less file also means.
    let pinning = match entry.and_then(|fm| fm.declared_model.as_deref()) {
        Some("inherit") => Pinning::Inherit,
        Some(m) => Pinning::Pinned {
            declared: m.to_string(),
        },
        None => Pinning::Unpinned,
    };
    (pinning, entry.and_then(|fm| fm.declared_effort.clone()))
}

/// Classify how an agent file constrains `dispatch`'s model.
///
/// Reads no files: `frontmatter` is supplied by the caller, which is
/// what keeps this a pure function over data and its tests free of
/// filesystem fixtures.
///
/// Two properties are worth naming. A dispatch with no recorded cwd
/// matches no `CwdSubtree` scope, so it can only match a `Global` one.
/// And the classification describes the agent file *as it stands
/// today* applied to a dispatch that ran earlier — putting pinning in
/// the row key is what makes that disagreement visible as two rows
/// rather than resolving it into one.
#[must_use]
pub fn resolve_pinning<S: BuildHasher>(
    inventory: &[ContextFile],
    frontmatter: &HashMap<PathBuf, AgentFrontmatter, S>,
    dispatch: &Dispatch,
) -> Pinning {
    classify(inventory, frontmatter, dispatch).0
}

// ---- row grouping ----

/// Accumulate dispatches into rows keyed by agent type × model ×
/// effort × pinning.
///
/// Accumulates into a `Vec` for the same determinism reason
/// `majority_assistant_value` does: `HashMap` iteration order is
/// nondeterministic, and row order feeds a ranked table.
#[must_use]
pub fn group_dispatches<S: BuildHasher>(
    dispatches: Vec<Dispatch>,
    inventory: &[ContextFile],
    frontmatter: &HashMap<PathBuf, AgentFrontmatter, S>,
) -> Vec<AgentRow> {
    let mut rows: Vec<AgentRow> = Vec::new();
    for dispatch in dispatches {
        let (pinning, declared_effort) = classify(inventory, frontmatter, &dispatch);
        let existing = rows.iter_mut().find(|r| {
            r.agent_type == dispatch.agent_type
                && r.model == dispatch.model
                && r.effort == dispatch.effort
                && r.pinning == pinning
        });
        let Some(row) = existing else {
            rows.push(AgentRow {
                agent_type: dispatch.agent_type,
                model: dispatch.model,
                effort: dispatch.effort,
                declared_effort,
                pinning,
                dispatches: 1,
                usage: dispatch.usage,
                cost: dispatch.cost,
            });
            continue;
        };
        row.dispatches += 1;
        row.usage.input += dispatch.usage.input;
        row.usage.output += dispatch.usage.output;
        row.usage.cache_creation.ephemeral_5m += dispatch.usage.cache_creation.ephemeral_5m;
        row.usage.cache_creation.ephemeral_1h += dispatch.usage.cache_creation.ephemeral_1h;
        row.usage.cache_read += dispatch.usage.cache_read;
        // Strict propagation: one unpriced dispatch collapses the
        // row's cost, and no later priced dispatch revives it.
        row.cost = match (row.cost, dispatch.cost) {
            (Some(mut acc), Some(add)) => {
                acc += add;
                Some(acc)
            }
            _ => None,
        };
    }
    rows
}

/// Rank rows: priced first by cost descending, then dispatch count
/// descending, then agent type ascending; unpriced last, ordered among
/// themselves by dispatch count descending then agent type ascending.
///
/// Incomparable costs fall back to `Ordering::Equal`, which is only a
/// valid ordering because a cost can never be NaN: every rate is
/// parsed from the catalog's JSON, which cannot spell NaN or infinity,
/// and the arithmetic over them is multiplication and addition of
/// finite values.
pub fn sort_rows(rows: &mut [AgentRow]) {
    rows.sort_by(|a, b| {
        let a_cost = a.cost.map(|c| c.total());
        let b_cost = b.cost.map(|c| c.total());
        // `None` sorts last, so compare on "is unpriced" first.
        a_cost
            .is_none()
            .cmp(&b_cost.is_none())
            .then_with(|| {
                b_cost
                    .unwrap_or(0.0)
                    .partial_cmp(&a_cost.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.dispatches.cmp(&a.dispatches))
            .then_with(|| a.agent_type.cmp(&b.agent_type))
    });
}

// ---- repricing ----

/// Price a recorded token bundle at another model's rates. Pure; no I/O.
///
/// Resolves `target_model` by exact catalog key before pricing, so no
/// caller can reach `PricingCatalog::lookup`'s longest-substring
/// fallback and reprice against a model nobody named.
///
/// It exists for its name and this comment: the result is exact over a
/// counterfactual it does not model, because the same task on a cheaper
/// model produces a smaller bundle. Every caller is therefore reading an
/// upper bound, and a bare `cost_for_components` call at those sites
/// would not say so.
#[must_use]
pub fn reprice(
    usage: &Usage,
    target_model: &str,
    catalog: &PricingCatalog,
) -> Option<CostBreakdown> {
    catalog.lookup_exact(target_model)?;
    catalog.cost_for_turn(usage, Some(target_model))
}

/// What a repricing footer reports.
///
/// The counts are not decoration: a delta summed over a subset that
/// silently dropped rows is a shrunken total passing as a complete one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RepricedDelta {
    /// Sum of `reprice(row).total() - row.cost.total()` over the rows
    /// counted. Both sides are `CostBreakdown`; the scalar reduction
    /// happens here rather than at each call site.
    pub delta: f64,
    pub rows_counted: usize,
    /// Rows skipped because `cost` was absent or `reprice` missed.
    pub rows_unpriced: usize,
    /// Fork rows, skipped even when the filter admits them.
    pub rows_forked: usize,
}

/// Sum of `reprice(row) - row.cost` over the rows handed in, skipping
/// `Fork` rows. Callers pass the rows the active filters left visible;
/// this function privileges no other pinning value and defines no slice
/// of its own.
///
/// A fork always runs its parent's model, so its repriced figure
/// describes a change nobody can make; folding it in would put a
/// meaningless number behind a documented user action.
#[must_use]
pub fn repriced_delta(
    rows: &[AgentRow],
    target_model: &str,
    catalog: &PricingCatalog,
) -> RepricedDelta {
    let mut out = RepricedDelta {
        delta: 0.0,
        rows_counted: 0,
        rows_unpriced: 0,
        rows_forked: 0,
    };
    for row in rows {
        if row.pinning == Pinning::Fork {
            out.rows_forked += 1;
            continue;
        }
        let (Some(current), Some(target)) = (row.cost, reprice(&row.usage, target_model, catalog))
        else {
            out.rows_unpriced += 1;
            continue;
        };
        out.delta += target.total() - current.total();
        out.rows_counted += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use chrono::TimeZone;

    use super::*;
    use crate::domain::TurnOrigin;
    use crate::inventory::Scope;
    use crate::pricing::{ClaudePricing, TieredRate};

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s)
            .expect("valid rfc3339")
            .with_timezone(&Utc)
    }

    /// A catalog pricing `known-model` at $1/MTok on every leg, so a
    /// cost figure reads directly as "tokens, in millionths of a
    /// dollar" and `reprice`'s arithmetic is checkable by hand.
    fn catalog_with(models: &[(&str, f64)]) -> PricingCatalog {
        let mut json = String::from("{");
        for (i, (model, rate)) in models.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            write!(
                json,
                r#""{model}":{{"input_cost_per_token":{rate},
                    "output_cost_per_token":{rate},
                    "cache_read_input_token_cost":{rate},
                    "cache_creation_input_token_cost":{rate}}}"#,
            )
            .expect("write to String");
        }
        json.push('}');
        PricingCatalog::from_raw_json(&json).expect("catalog parses")
    }

    fn assistant_turn(
        ts_str: &str,
        model: Option<&str>,
        effort: Option<&str>,
        tokens: u64,
    ) -> Turn {
        Turn {
            timestamp: Some(ts(ts_str)),
            role: Role::Assistant,
            model: model.map(str::to_string),
            effort: effort.map(str::to_string),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input: tokens,
                output: 0,
                cache_creation: CacheCreation::default(),
                cache_read: 0,
            }),
            content: None,
            cwd: None,
            origin: TurnOrigin::default(),
        }
    }

    fn user_turn(ts_str: &str, cwd: Option<&str>) -> Turn {
        Turn {
            timestamp: Some(ts(ts_str)),
            role: Role::User,
            model: None,
            effort: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: None,
            cwd: cwd.map(PathBuf::from),
            origin: TurnOrigin::default(),
        }
    }

    fn meta(agent_type: &str, is_fork: bool) -> SubagentMeta {
        SubagentMeta {
            agent_type: agent_type.to_string(),
            description: None,
            is_fork,
        }
    }

    fn agent_file(path: &str, kind: ContextFileKind, scope: Scope) -> ContextFile {
        ContextFile {
            path: PathBuf::from(path),
            kind,
            tokens: 0,
            scope,
        }
    }

    fn dispatch_for(agent_type: &str, cwd: Option<&str>, is_fork: bool) -> Dispatch {
        Dispatch {
            agent_type: agent_type.to_string(),
            is_fork,
            cwd: cwd.map(PathBuf::from),
            project_short_name: "proj".to_string(),
            started_at: Utc.timestamp_opt(0, 0).single().expect("epoch"),
            model: Some("known-model".to_string()),
            effort: None,
            usage: Usage {
                input: 10,
                output: 0,
                cache_creation: CacheCreation::default(),
                cache_read: 0,
            },
            cost: Some(CostBreakdown::default()),
        }
    }

    fn frontmatter_map(
        entries: &[(&str, Option<&str>, Option<&str>)],
    ) -> HashMap<PathBuf, AgentFrontmatter> {
        entries
            .iter()
            .map(|(path, model, effort)| {
                (
                    PathBuf::from(*path),
                    AgentFrontmatter {
                        declared_model: model.map(str::to_string),
                        declared_effort: effort.map(str::to_string),
                    },
                )
            })
            .collect()
    }

    // --- dispatch_from_turns ---

    #[test]
    fn dispatch_sums_usage_across_assistant_turns() {
        let catalog = catalog_with(&[("claude-known-model", 1e-6)]);
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("claude-known-model"), None, 10),
            assistant_turn("2026-04-01T10:01:00Z", Some("claude-known-model"), None, 25),
            // A user turn carries no usage and must not contribute.
            user_turn("2026-04-01T10:02:00Z", None),
        ];
        let d = dispatch_from_turns(&meta("a", false), None, "proj", &turns, &catalog)
            .expect("dispatch");
        assert_eq!(d.usage.input, 35);
        assert_eq!(d.started_at, ts("2026-04-01T10:00:00Z"));
        assert!((d.cost.expect("priced").total() - 35e-6).abs() < 1e-12);
    }

    #[test]
    fn dispatch_derives_majority_model_and_effort() {
        // Model and effort both change mid-run; each collapses to the
        // value the transcript mostly ran on, not the first seen.
        let catalog = catalog_with(&[("claude-known-model", 1e-6)]);
        let turns = vec![
            assistant_turn("2026-04-01T10:00:00Z", Some("first-model"), Some("low"), 1),
            assistant_turn(
                "2026-04-01T10:01:00Z",
                Some("claude-known-model"),
                Some("high"),
                1,
            ),
            assistant_turn(
                "2026-04-01T10:02:00Z",
                Some("claude-known-model"),
                Some("high"),
                1,
            ),
        ];
        let d = dispatch_from_turns(&meta("a", false), None, "proj", &turns, &catalog)
            .expect("dispatch");
        assert_eq!(d.model.as_deref(), Some("claude-known-model"));
        assert_eq!(d.effort.as_deref(), Some("high"));
    }

    #[test]
    fn dispatch_without_timestamps_yields_none() {
        let catalog = catalog_with(&[("claude-known-model", 1e-6)]);
        let mut turn = assistant_turn("2026-04-01T10:00:00Z", Some("claude-known-model"), None, 1);
        turn.timestamp = None;
        assert!(dispatch_from_turns(&meta("a", false), None, "proj", &[turn], &catalog).is_none());
    }

    #[test]
    fn dispatch_falls_back_to_parent_cwd() {
        let catalog = catalog_with(&[("claude-known-model", 1e-6)]);
        let turns = vec![assistant_turn(
            "2026-04-01T10:00:00Z",
            Some("claude-known-model"),
            None,
            1,
        )];
        let d = dispatch_from_turns(
            &meta("a", false),
            Some(Path::new("/work/parent-proj")),
            "parent-proj",
            &turns,
            &catalog,
        )
        .expect("dispatch");
        assert_eq!(d.cwd.as_deref(), Some(Path::new("/work/parent-proj")));
        assert_eq!(d.project_short_name, "parent-proj");
    }

    // --- resolve_pinning ---

    fn user_agent_inventory(stem: &str) -> Vec<ContextFile> {
        vec![agent_file(
            &format!("/home/.claude/agents/{stem}.md"),
            ContextFileKind::UserAgent,
            Scope::Global,
        )]
    }

    #[test]
    fn resolve_pinning_reads_concrete_model() {
        let inv = user_agent_inventory("a");
        let fm = frontmatter_map(&[("/home/.claude/agents/a.md", Some("claude-opus-5"), None)]);
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", Some("/work"), false)),
            Pinning::Pinned {
                declared: "claude-opus-5".to_string()
            },
        );
    }

    #[test]
    fn resolve_pinning_reads_inherit_sentinel() {
        let inv = user_agent_inventory("a");
        let fm = frontmatter_map(&[("/home/.claude/agents/a.md", Some("inherit"), None)]);
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", Some("/work"), false)),
            Pinning::Inherit,
        );
    }

    #[test]
    fn resolve_pinning_reads_absent_key_as_unpinned() {
        let inv = user_agent_inventory("a");
        let fm = frontmatter_map(&[("/home/.claude/agents/a.md", None, Some("high"))]);
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", Some("/work"), false)),
            Pinning::Unpinned,
        );
    }

    #[test]
    fn resolve_pinning_treats_an_unmapped_file_as_unpinned() {
        // A matched entry with no frontmatter map entry classifies the
        // same as a file whose block names no model — an unreadable or
        // block-less file says nothing about the model either way.
        let inv = user_agent_inventory("a");
        let fm = HashMap::new();
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", Some("/work"), false)),
            Pinning::Unpinned,
        );
    }

    #[test]
    fn resolve_pinning_yields_no_agent_file_when_nothing_matches() {
        let inv = user_agent_inventory("other");
        let fm = HashMap::new();
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", Some("/work"), false)),
            Pinning::NoAgentFile,
        );
    }

    #[test]
    fn resolve_pinning_returns_fork_before_consulting_the_inventory() {
        // A matching pinned file exists; the structural discriminator
        // must still win, because a fork has no agent definition and
        // the name collision says nothing about it.
        let inv = user_agent_inventory("a");
        let fm = frontmatter_map(&[("/home/.claude/agents/a.md", Some("claude-opus-5"), None)]);
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", Some("/work"), true)),
            Pinning::Fork,
        );
    }

    #[test]
    fn resolve_pinning_matches_a_global_file_for_a_dispatch_with_no_cwd() {
        // A cwd-less dispatch matches no `CwdSubtree` scope, but a
        // globally-scoped file still defines it. Dropping those too
        // would report genuinely pinned spend as `NoAgentFile`, which
        // the default filter admits — inflating the slice the view
        // exists to size.
        let inv = user_agent_inventory("a");
        let fm = frontmatter_map(&[("/home/.claude/agents/a.md", Some("claude-opus-5"), None)]);
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", None, false)),
            Pinning::Pinned {
                declared: "claude-opus-5".to_string()
            },
        );
    }

    #[test]
    fn resolve_pinning_still_ignores_a_cwd_scoped_file_for_a_dispatch_with_no_cwd() {
        let inv = vec![agent_file(
            "/work/.claude/agents/a.md",
            ContextFileKind::ProjectLocalAgent,
            Scope::CwdSubtree {
                root: PathBuf::from("/work"),
            },
        )];
        let fm = frontmatter_map(&[("/work/.claude/agents/a.md", Some("m"), None)]);
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", None, false)),
            Pinning::NoAgentFile,
        );
    }

    #[test]
    fn a_fork_declares_no_effort_even_when_a_name_collides() {
        // The fork short-circuit runs before the inventory is
        // consulted, so both columns agree that nothing is known —
        // a `declared_effort` here would assert in one column what
        // the pinning column says is unknowable.
        let inv = user_agent_inventory("a");
        let fm = frontmatter_map(&[("/home/.claude/agents/a.md", Some("m"), Some("high"))]);
        let rows = group_dispatches(vec![dispatch_for("a", Some("/work"), true)], &inv, &fm);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pinning, Pinning::Fork);
        assert!(
            rows[0].declared_effort.is_none(),
            "a fork must declare no effort, got {:?}",
            rows[0].declared_effort,
        );
    }

    #[test]
    fn pinning_filter_new_normalizes_order_and_duplicates() {
        // Equality is set-equality, so the default spelled in another
        // order must not report itself as an active filter.
        let reordered = PinningFilter::new(&[
            PinningKind::NoAgentFile,
            PinningKind::Unpinned,
            PinningKind::Inherit,
        ]);
        assert_eq!(reordered, PinningFilter::default());
        assert!(reordered.describe_active().is_empty());
        assert_eq!(
            PinningFilter::new(&[PinningKind::Fork, PinningKind::Fork]).describe_slice(),
            "fork",
        );
    }

    #[test]
    fn pinning_kind_all_lists_every_variant() {
        // `wildcard_enum_match_arm` forces `label`/`kind` to grow an
        // arm for a new variant, but a hand-written `ALL` would
        // compile unchanged and silently drop it from `everything()`,
        // `describe_slice`, and the parity test that iterates it. The
        // exhaustive match below is what breaks instead.
        for kind in PinningKind::ALL {
            match kind {
                PinningKind::Pinned
                | PinningKind::Inherit
                | PinningKind::Unpinned
                | PinningKind::NoAgentFile
                | PinningKind::Fork => {}
            }
        }
        assert_eq!(
            PinningKind::ALL.len(),
            5,
            "a new PinningKind must be added to ALL as well as to the match above",
        );
    }

    #[test]
    fn resolve_pinning_prefers_project_local_over_user_global_over_plugin() {
        let plugin = agent_file(
            "/cache/plug/agents/a.md",
            ContextFileKind::PluginAgent {
                plugin: "p".into(),
                marketplace: "m".into(),
                namespace: None,
            },
            Scope::Global,
        );
        let user = agent_file(
            "/home/.claude/agents/a.md",
            ContextFileKind::UserAgent,
            Scope::Global,
        );
        let local = agent_file(
            "/work/.claude/agents/a.md",
            ContextFileKind::ProjectLocalAgent,
            Scope::CwdSubtree {
                root: PathBuf::from("/work"),
            },
        );
        let fm = frontmatter_map(&[
            ("/cache/plug/agents/a.md", Some("plugin-model"), None),
            ("/home/.claude/agents/a.md", Some("user-model"), None),
            ("/work/.claude/agents/a.md", Some("local-model"), None),
        ]);

        let all = vec![plugin, user, local];
        assert_eq!(
            resolve_pinning(&all, &fm, &dispatch_for("a", Some("/work"), false)),
            Pinning::Pinned {
                declared: "local-model".to_string()
            },
        );

        // Removing the project-local entry promotes the user-global
        // one, which in turn outranks the plugin.
        let without_local: Vec<ContextFile> = all
            .into_iter()
            .filter(|f| !matches!(f.kind, ContextFileKind::ProjectLocalAgent))
            .collect();
        assert_eq!(
            resolve_pinning(
                &without_local,
                &fm,
                &dispatch_for("a", Some("/work"), false)
            ),
            Pinning::Pinned {
                declared: "user-model".to_string()
            },
        );
    }

    #[test]
    fn resolve_pinning_ignores_out_of_scope_project_local_file() {
        let inv = vec![agent_file(
            "/elsewhere/.claude/agents/a.md",
            ContextFileKind::ProjectLocalAgent,
            Scope::CwdSubtree {
                root: PathBuf::from("/elsewhere"),
            },
        )];
        let fm = frontmatter_map(&[("/elsewhere/.claude/agents/a.md", Some("m"), None)]);
        assert_eq!(
            resolve_pinning(&inv, &fm, &dispatch_for("a", Some("/work"), false)),
            Pinning::NoAgentFile,
        );
    }

    // --- group_dispatches / sort_rows ---

    #[test]
    fn group_dispatches_splits_rows_on_disagreeing_pinning() {
        // Two dispatches of one agent type at the same model and
        // effort: one runs where a pinned project-local file applies,
        // one does not. Pinning is in the row key, so they split.
        let inv = vec![agent_file(
            "/work/.claude/agents/a.md",
            ContextFileKind::ProjectLocalAgent,
            Scope::CwdSubtree {
                root: PathBuf::from("/work"),
            },
        )];
        let fm = frontmatter_map(&[(
            "/work/.claude/agents/a.md",
            Some("claude-opus-5"),
            Some("high"),
        )]);
        let rows = group_dispatches(
            vec![
                dispatch_for("a", Some("/work"), false),
                dispatch_for("a", Some("/elsewhere"), false),
            ],
            &inv,
            &fm,
        );
        assert_eq!(rows.len(), 2, "expected two rows, got {rows:#?}");
        let pinned = rows
            .iter()
            .find(|r| r.pinning.names_a_concrete_model())
            .expect("a pinned row");
        assert_eq!(pinned.declared_effort.as_deref(), Some("high"));
        assert!(
            rows.iter().any(|r| r.pinning == Pinning::NoAgentFile),
            "expected a no-agent-file row, got {rows:#?}",
        );
    }

    #[test]
    fn group_dispatches_propagates_none_cost_strictly() {
        let mut unpriced = dispatch_for("a", Some("/work"), false);
        unpriced.cost = None;
        let rows = group_dispatches(
            vec![dispatch_for("a", Some("/work"), false), unpriced],
            &[],
            &HashMap::new(),
        );
        assert_eq!(rows.len(), 1);
        assert!(
            rows[0].cost.is_none(),
            "one unpriced dispatch collapses the row"
        );
        // The token total still reflects every dispatch.
        assert_eq!(rows[0].usage.input, 20);
        assert_eq!(rows[0].dispatches, 2);
    }

    fn row(agent_type: &str, cost: Option<f64>, dispatches: u64) -> AgentRow {
        AgentRow {
            agent_type: agent_type.to_string(),
            model: Some("claude-known-model".to_string()),
            effort: None,
            declared_effort: None,
            pinning: Pinning::Unpinned,
            dispatches,
            usage: Usage {
                input: 10,
                output: 0,
                cache_creation: CacheCreation::default(),
                cache_read: 0,
            },
            cost: cost.map(|input| CostBreakdown {
                input,
                ..CostBreakdown::default()
            }),
        }
    }

    #[test]
    fn sort_rows_places_unpriced_rows_last() {
        let mut rows = vec![
            row("cheap", Some(1.0), 1),
            row("unpriced", None, 99),
            row("dear", Some(5.0), 1),
        ];
        sort_rows(&mut rows);
        let order: Vec<&str> = rows.iter().map(|r| r.agent_type.as_str()).collect();
        assert_eq!(order, vec!["dear", "cheap", "unpriced"]);
    }

    // --- repricing ---

    #[test]
    fn reprice_rejects_a_model_that_is_not_an_exact_catalog_key() {
        // `claude-opus-5` is in the catalog; `opus-5` is a spelling
        // `lookup` resolves through its `claude-` prefix fallback.
        // `reprice` must refuse it rather than price against a model
        // nobody named.
        let catalog = catalog_with(&[("claude-opus-5", 2e-6)]);
        let usage = Usage {
            input: 10,
            output: 0,
            cache_creation: CacheCreation::default(),
            cache_read: 0,
        };
        assert!(
            catalog.lookup("opus-5").is_some(),
            "precondition: lookup's prefix fallback resolves it",
        );
        assert!(reprice(&usage, "opus-5", &catalog).is_none());
        assert!(
            (reprice(&usage, "claude-opus-5", &catalog)
                .expect("exact key prices")
                .total()
                - 20e-6)
                .abs()
                < 1e-12,
        );
    }

    #[test]
    fn repriced_delta_excludes_fork_rows_and_counts_them() {
        let catalog = catalog_with(&[("claude-opus-5", 2e-6)]);
        let mut fork = row("forked", Some(0.0), 1);
        fork.pinning = Pinning::Fork;
        let d = repriced_delta(&[fork], "claude-opus-5", &catalog);
        assert_eq!(d.rows_forked, 1);
        assert_eq!(d.rows_counted, 0);
        assert!(d.delta.abs() < f64::EPSILON, "delta was {}", d.delta);
    }

    #[test]
    fn repriced_delta_counts_unpriced_rows_it_skipped() {
        let catalog = catalog_with(&[("claude-opus-5", 2e-6)]);
        let d = repriced_delta(
            &[row("a", None, 1), row("b", Some(0.0), 1)],
            "claude-opus-5",
            &catalog,
        );
        assert_eq!(d.rows_unpriced, 1);
        assert_eq!(d.rows_counted, 1);
        // 10 input tokens at $2/MTok, against a recorded cost of $0.
        assert!((d.delta - 20e-6).abs() < 1e-12, "delta was {}", d.delta);
    }

    // --- PinningFilter ---

    #[test]
    fn pinning_filter_default_admits_unpinned_inherit_and_no_agent_file() {
        let f = PinningFilter::default();
        assert!(f.accepts(&Pinning::Unpinned));
        assert!(f.accepts(&Pinning::Inherit));
        assert!(f.accepts(&Pinning::NoAgentFile));
        assert!(!f.accepts(&Pinning::Fork));
        assert!(!f.accepts(&Pinning::Pinned {
            declared: "m".into()
        }));
    }

    #[test]
    fn pinning_filter_describe_active_is_empty_at_the_default() {
        // The contract `Query::describe_active` depends on: a filter
        // at its default must contribute no component, or every view
        // would report itself as filtered.
        assert!(PinningFilter::default().describe_active().is_empty());
    }

    #[test]
    fn pinning_filter_describe_active_renders_a_non_default_slice() {
        let components = PinningFilter::new(&[PinningKind::Pinned]).describe_active();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].text, "--pinning pinned");
        assert!(
            components[0]
                .honored_by
                .honors(crate::filter::QueryScope::Agents)
        );
        assert!(
            !components[0]
                .honored_by
                .honors(crate::filter::QueryScope::Sessions)
        );
    }

    #[test]
    fn pinning_filter_everything_admits_every_kind() {
        let f = PinningFilter::everything();
        assert!(f.accepts(&Pinning::Fork));
        assert!(f.accepts(&Pinning::Pinned {
            declared: "m".into()
        }));
        assert_eq!(
            f.describe_slice(),
            "pinned,inherit,unpinned,no-agent-file,fork"
        );
    }

    #[test]
    fn pinning_labels_are_the_flag_spellings() {
        // The rendered cell text and the `--pinning` value must agree,
        // or a user cannot filter to the row they are looking at.
        for kind in PinningKind::ALL {
            let pinning = match kind {
                PinningKind::Pinned => Pinning::Pinned {
                    declared: "m".into(),
                },
                PinningKind::Inherit => Pinning::Inherit,
                PinningKind::Unpinned => Pinning::Unpinned,
                PinningKind::NoAgentFile => Pinning::NoAgentFile,
                PinningKind::Fork => Pinning::Fork,
            };
            assert_eq!(pinning.label(), kind.as_str());
            assert_eq!(pinning.kind(), kind);
        }
    }

    #[test]
    fn unused_pricing_type_imports_are_referenced() {
        // `ClaudePricing` / `TieredRate` are named so the catalog
        // helper's shape stays checkable if `from_raw_json` changes.
        let _: fn(&str) -> Option<&ClaudePricing> = |_| None;
        let _ = TieredRate {
            first_200k_rate: 0.0,
            above_200k_rate: 0.0,
        };
    }
}
