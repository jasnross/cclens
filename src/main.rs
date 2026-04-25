mod pricing;

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand};
use comfy_table::presets::NOTHING;
use comfy_table::{CellAlignment, Table};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---- cli ----

#[derive(Parser)]
#[command(
    name = "cclens",
    about = "Browse Claude Code conversations (tokens + cost)"
)]
struct Cli {
    /// Directory to scan for project conversations.
    #[arg(long, default_value_os_t = default_projects_dir())]
    projects_dir: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// List sessions (default).
    ///
    /// The `cost` column includes `cache_read` tokens (priced at the
    /// discounted cache-read rate) — the `tokens` column does not.
    List {
        #[command(flatten)]
        filters: FilterArgs,
    },
    /// Show per-exchange token + cost breakdown for one session.
    ///
    /// Per-row `cost` and running `cum_cost` columns include
    /// `cache_read` tokens; the `tokens` and `cumulative` columns do
    /// not. A row's `cost` cell renders `—` when its model is unknown
    /// to the pricing catalog; once an unknown-model row appears,
    /// every subsequent `cum_cost` cell also renders `—`.
    Show {
        /// Full session UUID (matches a .jsonl filename stem under --projects-dir).
        session_id: String,
        #[command(flatten)]
        filters: FilterArgs,
    },
    /// Manage the pricing catalog cache.
    Pricing {
        #[command(subcommand)]
        action: PricingAction,
    },
}

/// Shared `--min-tokens` / `--min-cost` thresholds for `list` and
/// `show`. Flattened into both subcommands via `#[command(flatten)]`;
/// deliberately not flattened into `pricing` (so `pricing refresh
/// --min-tokens 1` is a clap parse error, not a silent no-op).
///
/// The `Copy` bound matters: a threshold pair is two scalar `Option`s,
/// and copying them around the renderer avoids borrow plumbing without
/// adding a closure.
#[derive(Args, Debug, Clone, Copy, Default)]
struct FilterArgs {
    /// Show only rows with at least N billable tokens (e.g. --min-tokens 50000)
    #[arg(long)]
    min_tokens: Option<u64>,
    /// Show only rows costing at least USD, e.g. --min-cost 0.50; unknown-cost rows excluded
    #[arg(long)]
    min_cost: Option<f64>,
}

impl FilterArgs {
    /// Returns true iff (tokens, cost) clears every active threshold.
    /// `cost == None` (unknown model / unpriceable) fails any active
    /// `--min-cost` check; absent thresholds always pass. The two
    /// `is_none_or` calls collapse "no threshold OR threshold met" into
    /// one line each — the boolean and `&&`s the per-axis decisions
    /// into a logical AND.
    fn matches(&self, tokens: u64, cost: Option<f64>) -> bool {
        let tokens_ok = self.min_tokens.is_none_or(|t| tokens >= t);
        let cost_ok = self.min_cost.is_none_or(|c| cost.is_some_and(|n| n >= c));
        tokens_ok && cost_ok
    }

    /// True iff at least one filter flag is active. Used to gate the
    /// empty-result stderr hint — when no filter is active, an empty
    /// result is just an empty `projects_dir` and gets no hint.
    fn any_active(&self) -> bool {
        self.min_tokens.is_some() || self.min_cost.is_some()
    }

    /// Format the active flags for the empty-result stderr hint:
    /// `--min-tokens 50000`, `--min-cost 0.50`, or both joined by a
    /// space. Cost is formatted with `{}` (Rust's default float
    /// formatter, shortest round-trip representation) so small
    /// thresholds like `--min-cost 0.0001` round-trip faithfully —
    /// `{:.2}` would truncate them to `--min-cost 0.00`.
    fn describe_active(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(t) = self.min_tokens {
            parts.push(format!("--min-tokens {t}"));
        }
        if let Some(c) = self.min_cost {
            parts.push(format!("--min-cost {c}"));
        }
        parts.join(" ")
    }
}

/// Emit `note: no rows matched <flags>` to stderr when filters dropped
/// every row. No-op when no filter is active so the pre-existing
/// "empty `projects_dir` produces no stderr" contract is preserved.
fn emit_empty_result_hint(filters: &FilterArgs) {
    if !filters.any_active() {
        return;
    }
    eprintln!("note: no rows matched {}", filters.describe_active());
}

#[derive(Subcommand, Clone, Copy)]
enum PricingAction {
    /// Re-fetch the `LiteLLM` pricing catalog and overwrite the cache.
    Refresh,
    /// Print cache path, size, mtime, and Claude-entry count.
    Info,
}

fn default_projects_dir() -> PathBuf {
    dirs::home_dir().map_or_else(
        || PathBuf::from(".claude/projects"),
        |h| h.join(".claude/projects"),
    )
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::List {
        filters: FilterArgs::default(),
    }) {
        Command::List { filters } => run_list(&cli.projects_dir, filters),
        Command::Show {
            session_id,
            filters,
        } => run_show(&cli.projects_dir, &session_id, filters),
        Command::Pricing { action } => run_pricing(action),
    }
}

fn run_list(projects_dir: &Path, filters: FilterArgs) -> anyhow::Result<()> {
    let catalog = pricing::load_catalog();
    let project_entries = discover(projects_dir)?;
    let mut sessions = Vec::new();
    for (project_dir, jsonl_paths) in project_entries {
        // Cross-file dedup state, scoped to one project. Resumed
        // sessions replay prior assistant turns verbatim; this set
        // ensures any `(message.id, requestId)` pair contributes cost
        // exactly once across the project's `.jsonl` files. `discover`
        // already returned `jsonl_paths` in mtime-ascending order, so
        // the earliest file containing a given key wins.
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for jsonl_path in jsonl_paths {
            // A single unreadable file should not abort the whole listing.
            let Ok(turns) = parse_jsonl(&jsonl_path) else {
                continue;
            };
            let turns = dedup_assistant_turns(turns, &mut seen);
            let session_id = jsonl_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            // The new threshold filter sits *after* `aggregate`'s
            // existing zero-billable pre-pass, so it composes
            // additively rather than altering the "session is
            // meaningful" contract.
            if let Some(session) = aggregate(&project_dir, session_id, turns, &catalog)
                && filters.matches(session.total_billable, session.total_cost)
            {
                sessions.push(session);
            }
        }
    }
    sessions.sort_by_key(|s| s.started_at);
    println!("{}", render_table(&sessions));
    if sessions.is_empty() {
        emit_empty_result_hint(&filters);
    }
    Ok(())
}

/// Drop assistant turns whose `(message_id, request_id)` pair has
/// already been seen earlier in this project's file walk.
///
/// Mirrors ccusage's `createUniqueHash`: a turn missing either part of
/// the pair passes through (matches their `null`-on-partial-key rule).
/// Non-assistant turns (user, attachment, system, other) are unaffected
/// — they don't carry billable usage and are required for title
/// extraction and exchange grouping. Sidechain markers (`isSidechain`)
/// aren't visible at this layer (`RawLine` doesn't deserialize the
/// flag), so every assistant turn is keyed on `(message_id, request_id)`
/// regardless of sidechain status.
fn dedup_assistant_turns(turns: Vec<Turn>, seen: &mut HashSet<(String, String)>) -> Vec<Turn> {
    turns
        .into_iter()
        .filter(|turn| match &turn.role {
            Role::Assistant => match (turn.message_id.as_ref(), turn.request_id.as_ref()) {
                (Some(mid), Some(rid)) => seen.insert((mid.clone(), rid.clone())),
                // Missing key — pass through unchanged (matches
                // ccusage's createUniqueHash returning null on
                // partial keys).
                _ => true,
            },
            Role::User | Role::Attachment | Role::System | Role::Other(_) => true,
        })
        .collect()
}

fn run_pricing(action: PricingAction) -> anyhow::Result<()> {
    match action {
        PricingAction::Refresh => {
            let report = pricing::refresh_catalog()?;
            println!("Refreshed catalog at {}", report.path.display());
            println!(
                "  previous size: {} bytes → new size: {} bytes",
                report.previous_size, report.new_size,
            );
            println!("  Claude entries: {}", report.entry_count);
            Ok(())
        }
        PricingAction::Info => {
            let info = pricing::cache_info();
            match info.path {
                Some(path) => println!("Cache path: {}", path.display()),
                None => println!("Cache path: (no cache directory available)"),
            }
            println!("  exists: {}", info.exists);
            let mtime = info.last_modified.map_or_else(
                || "(never)".to_string(),
                |ts| {
                    let dt: chrono::DateTime<chrono::Local> = ts.into();
                    dt.format("%Y-%m-%d %H:%M:%S").to_string()
                },
            );
            println!("  last modified: {mtime}");
            println!("  size: {} bytes", info.size);
            let entries = info
                .entry_count
                .map_or_else(|| "(unreadable)".to_string(), |n| n.to_string());
            println!("  Claude entries: {entries}");
            Ok(())
        }
    }
}

fn run_show(projects_dir: &Path, session_id: &str, filters: FilterArgs) -> anyhow::Result<()> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        anyhow::bail!("session id must not be empty");
    }
    let project_entries = discover(projects_dir)?;

    // Locate the project that owns this session. A stem collision
    // *across* projects is unlikely (Claude Code uses fresh UUIDs)
    // but we keep the "multiple sessions match id …" error for
    // global ambiguity. Filenames within a single project directory
    // are unique, so per-project ambiguity isn't possible here.
    let mut matched_project: Option<(PathBuf, Vec<PathBuf>)> = None;
    for (project_dir, jsonl_paths) in project_entries {
        if jsonl_paths.iter().any(|p| stem_matches(p, session_id)) {
            if matched_project.is_some() {
                anyhow::bail!("multiple sessions match id {session_id}");
            }
            matched_project = Some((project_dir, jsonl_paths));
        }
    }
    let Some((_project_dir, jsonl_paths)) = matched_project else {
        anyhow::bail!("no session matches id {session_id}");
    };

    // Walk the project's files in mtime-ascending order, threading the
    // same per-project dedup state used by `run_list`. Stop as soon as
    // we've processed the target file: files later in mtime order
    // can't affect the target's filtered turns.
    let catalog = pricing::load_catalog();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut target_turns: Option<Vec<Turn>> = None;
    for jsonl_path in jsonl_paths {
        let Ok(turns) = parse_jsonl(&jsonl_path) else {
            continue;
        };
        let turns = dedup_assistant_turns(turns, &mut seen);
        if stem_matches(&jsonl_path, session_id) {
            target_turns = Some(turns);
            break;
        }
    }
    let turns =
        target_turns.ok_or_else(|| anyhow::anyhow!("no session matches id {session_id}"))?;
    let exchanges = group_into_exchanges(&turns);
    let (rendered, rows_shown) = render_session(&exchanges, &catalog, &filters);
    println!("{rendered}");
    if rows_shown == 0 {
        emit_empty_result_hint(&filters);
    }
    Ok(())
}

fn stem_matches(path: &Path, session_id: &str) -> bool {
    path.file_stem()
        .is_some_and(|s| s.to_string_lossy() == session_id)
}

// ---- domain ----

#[derive(Debug, Serialize)]
struct Session {
    id: String,
    project_short_name: String,
    started_at: DateTime<Utc>,
    last_activity: DateTime<Utc>,
    title: String,
    turns: Vec<Turn>,
    total_billable: u64,
    /// `Some(sum)` only if every assistant turn's cost resolved to
    /// `Some(_)`. A single unknown-model assistant turn collapses the
    /// whole session to `None` (strict propagation, no partial sums)
    /// — the rendered `cost` cell becomes `—`. Diverges from
    /// `total_billable` by including `cache_read` tokens.
    total_cost: Option<f64>,
}

impl Session {
    #[allow(dead_code)]
    fn duration(&self) -> chrono::Duration {
        self.last_activity - self.started_at
    }
}

#[derive(Debug, Serialize)]
struct Turn {
    timestamp: Option<DateTime<Utc>>,
    role: Role,
    model: Option<String>,
    message_id: Option<String>,
    request_id: Option<String>,
    usage: Option<Usage>,
    content: Option<Value>,
    cwd: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
enum Role {
    User,
    Assistant,
    Attachment,
    System,
    Other(String),
}

#[derive(Debug, Serialize)]
struct Usage {
    input: u64,
    output: u64,
    cache_creation: CacheCreation,
    cache_read: u64,
}

impl Usage {
    fn billable(&self) -> u64 {
        self.input + self.output + self.cache_creation.total()
    }
}

/// Cache-creation token counts split by ephemeral lifetime. The 5m and
/// 1h buckets are billed at distinct rates (the 1h rate is roughly
/// 1.6× the 5m rate for opus-4-7), so cclens keeps them separate from
/// `RawUsage` through to `pricing::cost_for_components`. Crate-root
/// items are visible to descendant modules by default; `pricing.rs`
/// names this type in its signature and reads its fields without any
/// visibility modifier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
struct CacheCreation {
    ephemeral_5m: u64,
    ephemeral_1h: u64,
}

impl CacheCreation {
    fn total(&self) -> u64 {
        self.ephemeral_5m + self.ephemeral_1h
    }
}

// ---- parsing ----

#[derive(Deserialize)]
struct RawLine {
    #[serde(rename = "type", default)]
    line_type: Option<String>,
    #[serde(default)]
    timestamp: Option<DateTime<Utc>>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    message: Option<RawMessage>,
    #[serde(rename = "requestId", default)]
    request_id: Option<String>,
}

#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<RawUsage>,
    #[serde(default)]
    content: Option<Value>,
}

// Field names mirror the on-disk JSON schema — the `_tokens` suffix is contractual.
#[allow(clippy::struct_field_names)]
#[derive(Deserialize)]
struct RawUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    /// Legacy flat scalar — older transcripts emit only this field;
    /// modern transcripts emit the structured `cache_creation` object
    /// instead. The adapter `into_usage` prefers the structured form
    /// when either bucket is non-zero, falling back to the scalar
    /// (treated as 5m).
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_creation: RawCacheCreation,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

/// On-disk shape of the modern `cache_creation` object emitted by
/// recent Claude Code builds:
/// `{"ephemeral_5m_input_tokens": N, "ephemeral_1h_input_tokens": M}`.
/// Older transcripts omit this and emit only the legacy flat scalar
/// `cache_creation_input_tokens` on `RawUsage`; both fields are kept
/// on `RawUsage` so cclens can read either shape.
#[derive(Debug, Default, Deserialize)]
struct RawCacheCreation {
    #[serde(default)]
    ephemeral_5m_input_tokens: u64,
    #[serde(default)]
    ephemeral_1h_input_tokens: u64,
}

fn into_usage(raw: &RawUsage) -> Usage {
    // Prefer the modern bucketed object when present. Treat "either
    // bucket non-zero" as the signal — an explicit
    // `{ephemeral_5m: 0, ephemeral_1h: 0}` is observationally
    // equivalent to falling through to the legacy scalar (also 0 if
    // missing, or the only field set on older transcripts).
    let cache_creation = if raw.cache_creation.ephemeral_5m_input_tokens > 0
        || raw.cache_creation.ephemeral_1h_input_tokens > 0
    {
        CacheCreation {
            ephemeral_5m: raw.cache_creation.ephemeral_5m_input_tokens,
            ephemeral_1h: raw.cache_creation.ephemeral_1h_input_tokens,
        }
    } else {
        // Legacy transcripts: the flat scalar is treated as 5m so it
        // prices identically to today's behavior.
        CacheCreation {
            ephemeral_5m: raw.cache_creation_input_tokens,
            ephemeral_1h: 0,
        }
    };
    Usage {
        input: raw.input_tokens,
        output: raw.output_tokens,
        cache_creation,
        cache_read: raw.cache_read_input_tokens,
    }
}

fn parse_jsonl(path: &Path) -> anyhow::Result<Vec<Turn>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut turns = Vec::new();
    for line_result in reader.lines() {
        let Ok(text) = line_result else { continue };
        if text.trim().is_empty() {
            continue;
        }
        let Ok(raw) = serde_json::from_str::<RawLine>(&text) else {
            continue;
        };
        if let Some(turn) = raw_to_turn(raw) {
            turns.push(turn);
        }
    }
    Ok(turns)
}

fn raw_to_turn(raw: RawLine) -> Option<Turn> {
    let line_type = raw.line_type?;
    let role = match line_type.as_str() {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "attachment" => Role::Attachment,
        "system" => Role::System,
        other => Role::Other(other.to_string()),
    };
    let (model, message_id, usage, content) = raw.message.map_or((None, None, None, None), |msg| {
        (
            msg.model,
            msg.id,
            msg.usage.as_ref().map(into_usage),
            msg.content,
        )
    });
    Some(Turn {
        timestamp: raw.timestamp,
        role,
        model,
        message_id,
        request_id: raw.request_id,
        usage,
        content,
        cwd: raw.cwd,
    })
}

// ---- aggregation ----

fn aggregate(
    project_dir: &Path,
    session_id: String,
    turns: Vec<Turn>,
    catalog: &pricing::PricingCatalog,
) -> Option<Session> {
    let total_billable: u64 = turns.iter().filter_map(billable_from_turn).sum();
    let total_cost = total_session_cost(&turns, catalog);
    // Drop the session only when it's zero-billable AND its cost is
    // *known* to be exactly zero. A session with only `cache_read`
    // tokens has `total_billable == 0` but a non-zero `total_cost` —
    // keep it visible. A session whose cost couldn't be resolved
    // (`None` from an unknown-model assistant turn) also stays
    // visible and renders as `—`. Using `matches!(_, Some(c) if c ==
    // 0.0)` rather than `unwrap_or(0.0) == 0.0` is what preserves the
    // second case — `unwrap_or(0.0)` would silently treat `None` as
    // zero and drop the session.
    if total_billable == 0 && matches!(total_cost, Some(c) if c == 0.0) {
        return None;
    }
    let started_at = turns.iter().filter_map(|t| t.timestamp).min()?;
    let last_activity = turns.iter().filter_map(|t| t.timestamp).max()?;
    let project_short_name = derive_project_short_name(project_dir, &turns);
    let title = extract_title(&turns, &session_id);
    Some(Session {
        id: session_id,
        project_short_name,
        started_at,
        last_activity,
        title,
        turns,
        total_billable,
        total_cost,
    })
}

fn billable_from_turn(turn: &Turn) -> Option<u64> {
    match &turn.role {
        Role::Assistant => turn.usage.as_ref().map(Usage::billable),
        Role::User | Role::Attachment | Role::System | Role::Other(_) => None,
    }
}

/// Sum per-assistant-turn costs with strict `None` propagation. User /
/// attachment / system / other-role turns contribute nothing and never
/// collapse the result to `None`. A single assistant turn that the
/// catalog can't price collapses the whole session to `None`.
fn total_session_cost(turns: &[Turn], catalog: &pricing::PricingCatalog) -> Option<f64> {
    let mut sum = 0.0;
    for turn in turns {
        match &turn.role {
            Role::Assistant => {}
            Role::User | Role::Attachment | Role::System | Role::Other(_) => continue,
        }
        let Some(usage) = turn.usage.as_ref() else {
            // Assistant turn with no usage is treated as zero-cost
            // (matches `billable_from_turn`'s None → 0 fold via
            // `filter_map`); does not collapse the total to `None`.
            continue;
        };
        let cost = catalog.cost_for_turn(usage, turn.model.as_deref())?;
        sum += cost;
    }
    Some(sum)
}

fn derive_project_short_name(project_dir: &Path, turns: &[Turn]) -> String {
    for turn in turns {
        if let Some(cwd) = &turn.cwd
            && let Some(name) = cwd.file_name()
        {
            return name.to_string_lossy().into_owned();
        }
    }
    project_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// Harness-synthetic user-content prefixes that should never be treated as the
// user's intent. Applied to both string content and array `text` blocks — real
// Claude Code stores skill-injection preambles as single-text-block arrays, so
// the check has to cover both shapes.
//
// The first three are XML-like envelopes and won't false-positive in practice.
// "Base directory for this skill: " is plain prose and in theory a real user
// prompt could start with it; accepted as a low-probability tradeoff since
// real sessions always produce it and no real prompt has been seen to match.
const SYNTHETIC_USER_CONTENT_PREFIXES: &[&str] = &[
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    // Skill harness injects the skill file's contents as a user turn whose
    // content array contains one text block starting with this prefix.
    "Base directory for this skill: ",
];

fn extract_title(turns: &[Turn], session_id: &str) -> String {
    for turn in turns {
        match &turn.role {
            Role::User => {}
            Role::Assistant | Role::Attachment | Role::System | Role::Other(_) => continue,
        }
        let Some(content) = &turn.content else {
            continue;
        };
        if let Some(title) = user_display_string(content) {
            return title;
        }
    }
    session_id.to_string()
}

fn is_synthetic_user_text(s: &str) -> bool {
    SYNTHETIC_USER_CONTENT_PREFIXES
        .iter()
        .any(|p| s.starts_with(p))
}

/// Shared content-extraction primitive: returns the string we'd display for a
/// user turn's content, or `None` if the content is harness-synthetic or has
/// no displayable text.
///
/// - For string content: reconstructs a slash command if the content is a
///   `<command-name>…` XML wrapper; otherwise returns the string as-is.
///   Returns `None` for synthetic-prefix content.
/// - For array content: returns the first non-empty, non-synthetic `"text"`
///   block's text verbatim. Slash-command XML wrappers have never been
///   observed inside array text blocks in real data, so reconstruction runs
///   only on string content — if that shape starts appearing, delegate the
///   block's text to `extract_slash_command_title` here.
/// - For all other `Value` shapes: returns `None`.
///
/// Used by both `extract_title` (list view) and `user_content_preview`
/// (show view) so the two views agree on what counts as "what the user said".
/// `is_substantive_user_turn` delegates to this function — a turn is
/// substantive iff this returns `Some(_)`.
fn user_display_string(content: &Value) -> Option<String> {
    if let Some(s) = content.as_str() {
        if is_synthetic_user_text(s) {
            return None;
        }
        if let Some(title) = extract_slash_command_title(s) {
            return Some(title);
        }
        return Some(s.to_string());
    }
    if let Some(arr) = content.as_array() {
        for block in arr {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            if let Some(text) = block.get("text").and_then(Value::as_str)
                && !text.is_empty()
                && !is_synthetic_user_text(text)
            {
                return Some(text.to_string());
            }
        }
    }
    None
}

/// `true` iff a user-role turn carries real user intent (as opposed to harness
/// noise or tool-result-only array content). Defined as "there is a display
/// string we can derive from the content" — this keeps the substantive-check
/// and the display-string extraction consistent by construction.
fn is_substantive_user_turn(turn: &Turn) -> bool {
    match &turn.role {
        Role::User => {}
        Role::Assistant | Role::Attachment | Role::System | Role::Other(_) => return false,
    }
    turn.content
        .as_ref()
        .and_then(user_display_string)
        .is_some()
}

/// A user turn plus the cluster of consecutive assistant turns that follow it
/// (up to the next substantive user turn). The `assistants` vec is empty when
/// the user turn is orphaned at the end of a session with no response.
#[derive(Debug)]
struct Exchange<'a> {
    user: &'a Turn,
    assistants: Vec<&'a Turn>,
}

#[derive(Debug)]
struct ExchangeInProgress<'a> {
    user: &'a Turn,
    assistants: Vec<&'a Turn>,
}

impl<'a> ExchangeInProgress<'a> {
    fn finish(self) -> Exchange<'a> {
        Exchange {
            user: self.user,
            assistants: self.assistants,
        }
    }
}

fn group_into_exchanges(turns: &[Turn]) -> Vec<Exchange<'_>> {
    let mut exchanges: Vec<Exchange<'_>> = Vec::new();
    let mut current: Option<ExchangeInProgress<'_>> = None;

    for turn in turns {
        match &turn.role {
            Role::User => {
                if is_substantive_user_turn(turn) {
                    if let Some(existing) = current.take() {
                        exchanges.push(existing.finish());
                    }
                    current = Some(ExchangeInProgress {
                        user: turn,
                        assistants: Vec::new(),
                    });
                }
                // else: non-substantive user (synthetic wrapper or
                // tool-result-only array) — absorbed, no row.
            }
            Role::Assistant => {
                if let Some(builder) = current.as_mut() {
                    builder.assistants.push(turn);
                }
                // else: assistant before any substantive user — dropped.
            }
            Role::Attachment | Role::System | Role::Other(_) => {
                // Non-user/non-assistant — absorbed, no row.
            }
        }
    }

    if let Some(existing) = current.take() {
        exchanges.push(existing.finish());
    }
    exchanges
}

/// Exchange-level (`tokens`, `cost`) for filter decisions.
///
/// `tokens` matches `Session.total_billable` semantics across the
/// exchange's assistant cluster: `input + output + cache_creation`.
/// `cost` matches `Session.total_cost` semantics: a strict fold over
/// the cluster, collapsing to `None` if any single assistant turn has
/// an unknown model (so any active `--min-cost` excludes the row).
///
/// Orphan exchanges (no assistants) yield `(0, None)` — `None` cost
/// because there is no priced cluster to sum, so any active
/// `--min-cost` excludes the orphan, and `--min-tokens >= 1` excludes
/// it via the zero token count.
///
/// These totals are for filter decisions only — `render_session`
/// continues to compute its own row decomposition inline because that's
/// a presentation concern (user-row vs assistant-row token slicing).
fn exchange_filter_totals(
    exchange: &Exchange<'_>,
    catalog: &pricing::PricingCatalog,
) -> (u64, Option<f64>) {
    if exchange.assistants.is_empty() {
        return (0, None);
    }
    let tokens: u64 = exchange
        .assistants
        .iter()
        .filter_map(|t| t.usage.as_ref())
        .map(Usage::billable)
        .sum();
    let mut sum = 0.0;
    for turn in &exchange.assistants {
        let Some(usage) = turn.usage.as_ref() else {
            // Assistant turn with no usage contributes zero (matches
            // `total_session_cost`'s rule); does not collapse the sum.
            continue;
        };
        let Some(cost) = catalog.cost_for_turn(usage, turn.model.as_deref()) else {
            return (tokens, None);
        };
        sum += cost;
    }
    (tokens, Some(sum))
}

fn extract_slash_command_title(s: &str) -> Option<String> {
    let name_open = s.find("<command-name>")?;
    let name_content_start = name_open + "<command-name>".len();
    let name_close_offset = s[name_content_start..].find("</command-name>")?;
    let name = &s[name_content_start..name_content_start + name_close_offset];
    if name.is_empty() {
        return None;
    }

    let after_name_tag = name_content_start + name_close_offset + "</command-name>".len();
    let tail = &s[after_name_tag..];
    let args = tail.find("<command-args>").and_then(|tag_start| {
        let args_content_start = tag_start + "<command-args>".len();
        tail[args_content_start..]
            .find("</command-args>")
            .map(|end| &tail[args_content_start..args_content_start + end])
    });

    match args {
        Some(a) if !a.is_empty() => Some(format!("{name} {a}")),
        _ => Some(name.to_string()),
    }
}

// ---- discovery ----

fn discover(projects_dir: &Path) -> anyhow::Result<Vec<(PathBuf, Vec<PathBuf>)>> {
    let mut result = Vec::new();
    // Propagate only failure to read the top-level projects dir itself — that's
    // a user-facing config error. Per-entry and per-subdir errors are skipped
    // so one unreadable project doesn't hide the rest.
    for entry in fs::read_dir(projects_dir)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(inner) = fs::read_dir(&path) else {
            continue;
        };
        let mut jsonl_paths = Vec::new();
        for session_entry in inner {
            let Ok(session_entry) = session_entry else {
                continue;
            };
            let session_path = session_entry.path();
            if session_path.is_dir() {
                continue;
            }
            if session_path.extension().is_some_and(|ext| ext == "jsonl") {
                jsonl_paths.push(session_path);
            }
        }
        // Resumed sessions inherit cost from the original; ordering by
        // mtime ascending makes "earliest file wins" deterministic and
        // semantically meaningful for the cross-file dedup pass.
        sort_paths_by_mtime_asc(&mut jsonl_paths);
        result.push((path, jsonl_paths));
    }
    Ok(result)
}

/// Sort `.jsonl` paths by `(mtime, path)` ascending. Files whose
/// metadata or `modified()` call fails sort to the start
/// (`UNIX_EPOCH`); they are then attempted by `parse_jsonl` like any
/// other file and skipped via the existing per-file
/// `let Ok(turns) = parse_jsonl(...) else { continue }` handler if
/// genuinely unreadable. The `path` secondary key makes the order
/// fully deterministic when two files share an mtime (possible on
/// low-resolution filesystems or after `cp -p`); without it, ties
/// would fall back to `read_dir` insertion order, which the OS does
/// not specify.
fn sort_paths_by_mtime_asc(paths: &mut [PathBuf]) {
    paths.sort_by_cached_key(|p| {
        let mtime = fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        (mtime, p.clone())
    });
}

// ---- rendering ----

const TITLE_MAX_CHARS: usize = 80;

// Indices of numeric columns in `render_table`'s header:
//   vec!["datetime", "project", "title", "tokens", "cost", "id"]
//                                         idx 3    idx 4
// Right-alignment is applied at these positions; reordering the header
// requires updating these constants in lockstep.
const TOKENS_COL_INDEX: usize = 3;
const COST_COL_INDEX: usize = 4;

// Indices of numeric columns in `render_session`'s header:
//   vec!["datetime", "role", "tokens", "cost", "cumulative", "cum_cost", "content"]
//                             idx 2    idx 3   idx 4         idx 5
const SHOW_TOKENS_COL_INDEX: usize = 2;
const SHOW_COST_COL_INDEX: usize = 3;
const SHOW_CUMULATIVE_COL_INDEX: usize = 4;
const SHOW_CUM_COST_COL_INDEX: usize = 5;

/// Format an optional cost as `$X.XXXX` or `—` for the unknown-model
/// case. Centralized so list and show share the exact same vocabulary.
fn format_cost_opt(c: Option<f64>) -> String {
    c.map_or_else(|| "—".to_string(), |n| format!("${n:.4}"))
}

fn truncate_title(s: &str, max: usize) -> String {
    // Collapse internal whitespace runs (including `\n`, `\t`) to a single
    // space and trim ends. Comfy-table respects embedded newlines and would
    // otherwise render a single cell across multiple visual rows — real
    // JSONL content (e.g. skill preambles) contains newlines that would
    // break the one-line-per-row invariant without this step.
    let normalized = s.split_whitespace().collect::<Vec<_>>().join(" ");

    if normalized.chars().count() <= max {
        return normalized;
    }
    let mut result: String = normalized.chars().take(max.saturating_sub(1)).collect();
    result.push('…');
    result
}

fn format_local(ts: DateTime<Utc>) -> String {
    ts.with_timezone(&chrono::Local)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

// Returns the empty string on `None` so `render_session` stays infallible
// without `.unwrap()` (banned by the `unwrap_used` lint). In practice the
// timestamp is always present for substantive user turns and assistant turns;
// this branch exists only to keep the code panic-free.
fn format_local_or_empty(ts: Option<DateTime<Utc>>) -> String {
    ts.map_or_else(String::new, format_local)
}

fn render_table(sessions: &[Session]) -> String {
    let mut table = Table::new();
    table.load_preset(NOTHING);
    table.set_header(vec!["datetime", "project", "title", "tokens", "cost", "id"]);
    for session in sessions {
        table.add_row(vec![
            format_local(session.started_at),
            session.project_short_name.clone(),
            truncate_title(&session.title, TITLE_MAX_CHARS),
            session.total_billable.to_string(),
            format_cost_opt(session.total_cost),
            session.id.clone(),
        ]);
    }
    // column_mut returns Option; the columns are guaranteed present because
    // the header above defines them at the indices.
    if let Some(col) = table.column_mut(TOKENS_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(COST_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    format!("{table}")
}

fn user_content_preview(turn: &Turn) -> String {
    // The grouper guarantees the turn is substantive, so `None` is unreachable
    // in practice; `unwrap_or_default` keeps the renderer infallible without
    // requiring `.unwrap()`.
    turn.content
        .as_ref()
        .and_then(user_display_string)
        .unwrap_or_default()
}

fn assistant_cluster_preview(assistants: &[&Turn]) -> String {
    // First non-empty text block across the cluster.
    for a in assistants {
        let Some(Value::Array(blocks)) = a.content.as_ref() else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            if let Some(text) = block.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                return text.to_string();
            }
        }
    }

    // Fallback: deduped tool-use names, in order of first appearance.
    let mut names: Vec<String> = Vec::new();
    for a in assistants {
        let Some(Value::Array(blocks)) = a.content.as_ref() else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            if let Some(name) = block.get("name").and_then(Value::as_str)
                && !names.iter().any(|n| n == name)
            {
                names.push(name.to_string());
            }
        }
    }
    names.join(", ")
}

/// Sum costs across an assistant cluster with strict `None`
/// propagation. The closure picks which token components count for
/// this row (e.g. `(input, 0, cache_creation, cache_read)` for the
/// user row, `(0, output, CacheCreation::default(), 0)` for the
/// assistant row). Any single turn the catalog can't price collapses
/// the whole sum to `None`, matching the session-level rule.
fn strict_fold_assistant_cost(
    assistants: &[&Turn],
    catalog: &pricing::PricingCatalog,
    pick: impl Fn(&Usage) -> (u64, u64, CacheCreation, u64),
) -> Option<f64> {
    let mut sum = 0.0;
    for turn in assistants {
        let Some(usage) = turn.usage.as_ref() else {
            // Assistant turn with no usage: zero contribution, does
            // not collapse the sum to `None`.
            continue;
        };
        let (input, output, cache_creation, cache_read) = pick(usage);
        let cost = catalog.cost_for_components(
            input,
            output,
            cache_creation,
            cache_read,
            turn.model.as_deref(),
        )?;
        sum += cost;
    }
    Some(sum)
}

fn count_tool_uses(assistants: &[&Turn]) -> u64 {
    let mut n: u64 = 0;
    for a in assistants {
        let Some(Value::Array(blocks)) = a.content.as_ref() else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                n += 1;
            }
        }
    }
    n
}

/// Strict-fold the running cost: both `Some` → sum; either `None` →
/// stays `None` for this row and every subsequent row. Callers retain
/// the previous accumulator value so a `None` "latches" through every
/// later row, matching the unknown-model propagation contract.
fn fold_cum_cost(prev: Option<f64>, delta: Option<f64>) -> Option<f64> {
    prev.zip(delta).map(|(a, b)| a + b)
}

/// Render the per-exchange table.
///
/// Returns `(rendered, rows_shown)`. `rows_shown` counts physical
/// rendered rows (1 for an orphan-only visible exchange, 2 for a
/// normal visible exchange). `run_show` uses `rows_shown == 0` to
/// decide whether to emit the empty-result stderr hint.
///
/// Filtering happens here rather than in a pre-pass so that
/// `cumulative` and `cum_cost` continue to fold over **every**
/// exchange — the running totals on visible rows must still match the
/// session-level `list` totals, which means the renderer needs both
/// the unfiltered slice (for accumulation) and the predicate (for
/// skipping `add_row`).
fn render_session(
    exchanges: &[Exchange<'_>],
    catalog: &pricing::PricingCatalog,
    filters: &FilterArgs,
) -> (String, usize) {
    let mut table = Table::new();
    table.load_preset(NOTHING);
    table.set_header(vec![
        "datetime",
        "role",
        "tokens",
        "cost",
        "cumulative",
        "cum_cost",
        "content",
    ]);
    let mut cumulative: u64 = 0;
    let mut cum_cost: Option<f64> = Some(0.0);
    let mut rows_shown: usize = 0;

    for exchange in exchanges {
        let (ex_tokens, ex_cost) = exchange_filter_totals(exchange, catalog);
        let visible = filters.matches(ex_tokens, ex_cost);

        let user_tokens_opt: Option<u64> = if exchange.assistants.is_empty() {
            None
        } else {
            let sum = exchange
                .assistants
                .iter()
                .filter_map(|t| t.usage.as_ref())
                .map(|u| u.input + u.cache_creation.total())
                .sum();
            Some(sum)
        };
        let output_tokens: u64 = exchange
            .assistants
            .iter()
            .filter_map(|t| t.usage.as_ref())
            .map(|u| u.output)
            .sum();

        // Per-row cost decomposes into a *displayed* cost (what the
        // cell shows) and an *accumulator delta* (what's added to the
        // running cum_cost). They diverge on the orphan-user case:
        // an empty assistant cluster displays `—` but contributes
        // `Some(0.0)` to the running total — same way `cumulative +=
        // user_tokens_opt.unwrap_or(0)` treats orphans as a no-op.
        // For non-empty clusters they're equal: any unknown-model turn
        // collapses both to `None`, latching cum_cost through the rest
        // of the session.
        let (user_cost_display, user_cost_delta) = if exchange.assistants.is_empty() {
            (None, Some(0.0))
        } else {
            let cost = strict_fold_assistant_cost(&exchange.assistants, catalog, |usage| {
                (usage.input, 0, usage.cache_creation, usage.cache_read)
            });
            (cost, cost)
        };

        cumulative += user_tokens_opt.unwrap_or(0);
        cum_cost = fold_cum_cost(cum_cost, user_cost_delta);
        if visible {
            table.add_row(vec![
                format_local_or_empty(exchange.user.timestamp),
                "user".to_string(),
                user_tokens_opt.map_or_else(|| "—".to_string(), |n| n.to_string()),
                format_cost_opt(user_cost_display),
                cumulative.to_string(),
                format_cost_opt(cum_cost),
                truncate_title(&user_content_preview(exchange.user), TITLE_MAX_CHARS),
            ]);
            rows_shown += 1;
        }

        if let Some(first_assistant) = exchange.assistants.first() {
            let assistant_cost =
                strict_fold_assistant_cost(&exchange.assistants, catalog, |usage| {
                    (0, usage.output, CacheCreation::default(), 0)
                });
            cumulative += output_tokens;
            cum_cost = fold_cum_cost(cum_cost, assistant_cost);

            if visible {
                let preview = assistant_cluster_preview(&exchange.assistants);
                let n_tools = count_tool_uses(&exchange.assistants);
                let content = if n_tools > 0 {
                    format!("{preview} +{n_tools} tool uses")
                } else {
                    preview
                };
                table.add_row(vec![
                    format_local_or_empty(first_assistant.timestamp),
                    "assistant".to_string(),
                    output_tokens.to_string(),
                    format_cost_opt(assistant_cost),
                    cumulative.to_string(),
                    format_cost_opt(cum_cost),
                    truncate_title(&content, TITLE_MAX_CHARS),
                ]);
                rows_shown += 1;
            }
        }
    }

    if let Some(col) = table.column_mut(SHOW_TOKENS_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(SHOW_COST_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(SHOW_CUMULATIVE_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(SHOW_CUM_COST_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    (format!("{table}"), rows_shown)
}

// ---- tests ----

#[cfg(test)]
mod tests {
    use std::fs as stdfs;
    use std::io::Write;

    use super::*;

    // --- test helpers ---

    fn user_string_turn(content: &str) -> Turn {
        Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::String(content.to_string())),
            cwd: None,
        }
    }

    fn user_array_turn(content: Value) -> Turn {
        Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(content),
            cwd: None,
        }
    }

    fn assistant_turn_with_usage(input: u64, output: u64, cache_creation: u64) -> Turn {
        // Helper takes a flat u64 for backwards-compatible test
        // ergonomics; treated as 5m, matching the legacy wire scalar.
        Turn {
            timestamp: Some("2026-04-01T10:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input,
                output,
                cache_creation: CacheCreation {
                    ephemeral_5m: cache_creation,
                    ephemeral_1h: 0,
                },
                cache_read: 0,
            }),
            content: None,
            cwd: None,
        }
    }

    // --- CLI parsing (Phase 1) ---

    #[test]
    fn bare_invocation_leaves_command_as_none() {
        let cli = Cli::try_parse_from(["cclens"]).unwrap();
        assert!(cli.command.is_none());
    }

    #[test]
    fn explicit_list_parses_as_list_variant() {
        let cli = Cli::try_parse_from(["cclens", "list"]).unwrap();
        assert!(matches!(cli.command, Some(Command::List { .. })));
    }

    #[test]
    fn projects_dir_flag_overrides_default() {
        let cli = Cli::try_parse_from(["cclens", "--projects-dir", "/tmp/foo", "list"]).unwrap();
        assert_eq!(cli.projects_dir, PathBuf::from("/tmp/foo"));
    }

    #[test]
    fn default_projects_dir_ends_in_claude_projects() {
        let cli = Cli::try_parse_from(["cclens"]).unwrap();
        assert!(
            cli.projects_dir.ends_with(".claude/projects"),
            "expected default projects_dir to end in .claude/projects, got {:?}",
            cli.projects_dir,
        );
    }

    // --- domain helpers ---

    #[test]
    fn usage_billable_sums_three_fields() {
        let u = Usage {
            input: 6,
            output: 1186,
            cache_creation: CacheCreation {
                ephemeral_5m: 18998,
                ephemeral_1h: 0,
            },
            cache_read: 17317,
        };
        assert_eq!(u.billable(), 20190);
    }

    // --- title extraction ---

    #[test]
    fn extract_title_from_slash_command_with_args() {
        let content = "<command-name>/tw-create-idea</command-name>\n\
                       <command-args>I want to build a CLI</command-args>";
        let turns = vec![user_string_turn(content)];
        assert_eq!(
            extract_title(&turns, "fallback"),
            "/tw-create-idea I want to build a CLI",
        );
    }

    #[test]
    fn extract_title_from_slash_command_no_args() {
        let content = "<command-name>/tw-commit</command-name>";
        let turns = vec![user_string_turn(content)];
        assert_eq!(extract_title(&turns, "fallback"), "/tw-commit");
    }

    #[test]
    fn extract_title_skips_local_command_caveat() {
        let caveat =
            "<local-command-caveat>heads up: claude ran this locally</local-command-caveat>";
        let turns = vec![
            user_string_turn(caveat),
            user_string_turn("the real question"),
        ];
        assert_eq!(extract_title(&turns, "fallback"), "the real question");
    }

    #[test]
    fn extract_title_skips_local_command_stdout_and_stderr() {
        let stdout = "<local-command-stdout>some captured output</local-command-stdout>";
        let stderr = "<local-command-stderr>oops</local-command-stderr>";
        let turns = vec![
            user_string_turn(stdout),
            user_string_turn(stderr),
            user_string_turn("the real question"),
        ];
        assert_eq!(extract_title(&turns, "fallback"), "the real question");
    }

    #[test]
    fn extract_title_from_array_text_block() {
        let content = serde_json::json!([
            { "type": "tool_result", "content": "..." },
            { "type": "text", "text": "hello from the array form" },
        ]);
        let turns = vec![user_array_turn(content)];
        assert_eq!(
            extract_title(&turns, "fallback"),
            "hello from the array form",
        );
    }

    #[test]
    fn extract_title_falls_back_to_session_id() {
        // Array content with only tool_result blocks — no text block to use.
        let content = serde_json::json!([
            { "type": "tool_result", "content": "..." },
        ]);
        let turns = vec![user_array_turn(content)];
        assert_eq!(extract_title(&turns, "abcd1234"), "abcd1234");
    }

    // --- aggregation ---

    #[test]
    fn aggregate_filters_zero_billable() {
        let turns = vec![
            user_string_turn("hello"),
            assistant_turn_with_usage(0, 0, 0),
        ];
        let result = aggregate(
            Path::new("/tmp/fake-project"),
            "abc".to_string(),
            turns,
            &pricing::PricingCatalog::empty(),
        );
        assert!(result.is_none());
    }

    #[test]
    fn aggregate_sums_all_assistant_turns() {
        let turns = vec![
            user_string_turn("q1"),
            assistant_turn_with_usage(100, 0, 0),
            assistant_turn_with_usage(200, 0, 0),
        ];
        let session = aggregate(
            Path::new("/tmp/fake-project"),
            "abc".to_string(),
            turns,
            &pricing::PricingCatalog::empty(),
        )
        .expect("zero-billable filter should not fire");
        assert_eq!(session.total_billable, 300);
        // Empty catalog can't price the assistant turns; total_cost
        // collapses to None per the strict-propagation rule.
        assert_eq!(session.total_cost, None);
    }

    #[test]
    fn aggregate_keeps_unknown_model_zero_billable_session() {
        // Regression: previously the zero-billable filter used
        // `total_cost.unwrap_or(0.0) == 0.0`, which silently dropped
        // sessions whose cost was unresolved (`None`). Such sessions
        // must stay visible — the row will render `—` in the cost
        // cell, prompting the user to investigate.
        //
        // Construct a session with one zero-billable assistant turn
        // (so `total_billable == 0`) on an unknown model (so
        // `total_cost == None`). The filter must not drop it.
        let turns = vec![
            user_string_turn("hi"),
            Turn {
                timestamp: Some("2026-04-01T10:00:00Z".parse().unwrap()),
                role: Role::Assistant,
                model: Some("claude-fake-9-9".to_string()),
                message_id: None,
                request_id: None,
                usage: Some(Usage {
                    input: 0,
                    output: 0,
                    cache_creation: CacheCreation::default(),
                    // Non-zero cache_read makes cost_for_components
                    // bypass the all-zero short-circuit; with an empty
                    // catalog the lookup misses and total_cost is None.
                    cache_read: 100,
                }),
                content: None,
                cwd: None,
            },
        ];
        let session = aggregate(
            Path::new("/tmp/fake-project"),
            "abc".to_string(),
            turns,
            &pricing::PricingCatalog::empty(),
        )
        .expect("unknown-model zero-billable session must NOT be filtered out");
        assert_eq!(session.total_billable, 0);
        assert_eq!(session.total_cost, None);
    }

    // --- parser ---

    #[test]
    fn parse_jsonl_skips_malformed_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("session.jsonl");
        let mut f = stdfs::File::create(&path).unwrap();
        writeln!(
            f,
            "{{\"type\":\"assistant\",\"timestamp\":\"2026-04-01T10:00:00Z\",\"message\":{{\"usage\":{{\"input_tokens\":10,\"output_tokens\":5,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}}}}}",
        )
        .unwrap();
        writeln!(f, "not valid json").unwrap();
        writeln!(
            f,
            "{{\"type\":\"user\",\"timestamp\":\"2026-04-01T10:01:00Z\",\"message\":{{\"content\":\"hello\"}}}}",
        )
        .unwrap();
        drop(f);

        let turns = parse_jsonl(&path).unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn into_usage_prefers_bucketed_over_legacy_scalar() {
        // Modern wire shape only — structured cache_creation present,
        // legacy scalar absent.
        let bucketed: RawUsage = serde_json::from_str(
            r#"{"input_tokens":0,"output_tokens":0,
                "cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_1h_input_tokens":200},
                "cache_read_input_tokens":0}"#,
        )
        .expect("parse");
        assert_eq!(
            into_usage(&bucketed).cache_creation,
            CacheCreation {
                ephemeral_5m: 100,
                ephemeral_1h: 200,
            },
        );

        // Legacy wire shape only — flat scalar treated as 5m.
        let legacy: RawUsage = serde_json::from_str(
            r#"{"input_tokens":0,"output_tokens":0,
                "cache_creation_input_tokens":300,"cache_read_input_tokens":0}"#,
        )
        .expect("parse");
        assert_eq!(
            into_usage(&legacy).cache_creation,
            CacheCreation {
                ephemeral_5m: 300,
                ephemeral_1h: 0,
            },
        );

        // Both present — bucketed wins (structured is the modern,
        // higher-fidelity shape).
        let both: RawUsage = serde_json::from_str(
            r#"{"input_tokens":0,"output_tokens":0,
                "cache_creation_input_tokens":999,
                "cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_1h_input_tokens":200},
                "cache_read_input_tokens":0}"#,
        )
        .expect("parse");
        assert_eq!(
            into_usage(&both).cache_creation,
            CacheCreation {
                ephemeral_5m: 100,
                ephemeral_1h: 200,
            },
        );

        // Edge: bucketed object explicitly present with both buckets
        // zero AND the legacy scalar non-zero. The "either bucket
        // non-zero" predicate routes this to the legacy branch — a
        // future refactor that flipped to "object present at all"
        // would silently lose the legacy 300 here.
        let zeros_with_legacy: RawUsage = serde_json::from_str(
            r#"{"input_tokens":0,"output_tokens":0,
                "cache_creation_input_tokens":300,
                "cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},
                "cache_read_input_tokens":0}"#,
        )
        .expect("parse");
        assert_eq!(
            into_usage(&zeros_with_legacy).cache_creation,
            CacheCreation {
                ephemeral_5m: 300,
                ephemeral_1h: 0,
            },
        );
    }

    #[test]
    fn raw_to_turn_populates_message_id_and_request_id() {
        let raw_json = r#"{
            "type": "assistant",
            "timestamp": "2026-04-01T10:00:00Z",
            "requestId": "req_xyz",
            "message": {
                "id": "msg_abc",
                "model": "claude-opus-4-7",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0
                }
            }
        }"#;
        let raw: RawLine = serde_json::from_str(raw_json).expect("parse");
        let turn = raw_to_turn(raw).expect("turn");
        assert_eq!(turn.message_id.as_deref(), Some("msg_abc"));
        assert_eq!(turn.request_id.as_deref(), Some("req_xyz"));
    }

    #[test]
    fn raw_to_turn_leaves_ids_none_when_wire_fields_absent() {
        // Pre-existing fixtures don't carry message.id or requestId;
        // the dedup filter must see them as None and pass them through.
        let raw_json = r#"{
            "type": "assistant",
            "timestamp": "2026-04-01T10:00:00Z",
            "message": {"model": "claude-opus-4-7"}
        }"#;
        let raw: RawLine = serde_json::from_str(raw_json).expect("parse");
        let turn = raw_to_turn(raw).expect("turn");
        assert!(turn.message_id.is_none());
        assert!(turn.request_id.is_none());
    }

    #[test]
    fn dedup_drops_duplicate_assistant_turn_within_run() {
        // Build two assistants with the same (mid, rid); a third with
        // a different pair; a fourth with a missing key (must pass);
        // a user turn (must pass regardless).
        let mut first_dup = assistant_turn_with_usage(1, 1, 0);
        first_dup.message_id = Some("m1".into());
        first_dup.request_id = Some("r1".into());
        let mut second_dup = assistant_turn_with_usage(2, 2, 0);
        second_dup.message_id = Some("m1".into());
        second_dup.request_id = Some("r1".into());
        let mut other = assistant_turn_with_usage(3, 3, 0);
        other.message_id = Some("m2".into());
        other.request_id = Some("r2".into());
        let mut partial = assistant_turn_with_usage(4, 4, 0);
        // Partial key — must pass through.
        partial.message_id = Some("m3".into());
        partial.request_id = None;
        let user = user_string_turn("hi");

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let kept =
            dedup_assistant_turns(vec![first_dup, second_dup, other, partial, user], &mut seen);

        // first_dup (kept), second_dup (dropped), other (kept),
        // partial (kept, missing key), user (kept)
        assert_eq!(kept.len(), 4);
        // Verify the key cache observed the two complete pairs.
        assert!(seen.contains(&("m1".to_string(), "r1".to_string())));
        assert!(seen.contains(&("m2".to_string(), "r2".to_string())));
        // Surviving assistant turns retain their hash keys.
        assert_eq!(kept[0].message_id.as_deref(), Some("m1"));
        assert_eq!(kept[0].request_id.as_deref(), Some("r1"));
    }

    #[test]
    fn dedup_drops_duplicate_assistant_across_separate_calls() {
        // The orchestration layer reuses one HashSet across multiple
        // parse_jsonl results — confirm the second call picks up the
        // first call's state.
        let mut first_call_turn = assistant_turn_with_usage(1, 1, 0);
        first_call_turn.message_id = Some("m1".into());
        first_call_turn.request_id = Some("r1".into());
        let mut second_call_turn = assistant_turn_with_usage(2, 2, 0);
        second_call_turn.message_id = Some("m1".into());
        second_call_turn.request_id = Some("r1".into());

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let first = dedup_assistant_turns(vec![first_call_turn], &mut seen);
        let second = dedup_assistant_turns(vec![second_call_turn], &mut seen);
        assert_eq!(first.len(), 1);
        assert!(second.is_empty(), "duplicate from second call must drop");
    }

    #[test]
    fn dedup_passes_through_sidechain_assistant_with_unique_key() {
        // Sidechain turns aren't special-cased — they're deduped on
        // their own keys like any other assistant turn. This guards
        // against a regression that filtered them by role flag.
        let mut sidechain = assistant_turn_with_usage(50, 0, 0);
        sidechain.message_id = Some("msc".into());
        sidechain.request_id = Some("rsc".into());

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let kept = dedup_assistant_turns(vec![sidechain], &mut seen);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn parse_jsonl_skips_unknown_types() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("session.jsonl");
        let mut f = stdfs::File::create(&path).unwrap();
        writeln!(
            f,
            "{{\"type\":\"some-future-type\",\"timestamp\":\"2026-04-01T10:00:00Z\"}}"
        )
        .unwrap();
        drop(f);

        let turns = parse_jsonl(&path).unwrap();
        assert_eq!(turns.len(), 1);
        assert!(matches!(&turns[0].role, Role::Other(s) if s == "some-future-type"));
    }

    // --- short-name derivation ---

    #[test]
    fn aggregate_short_name_from_first_cwd() {
        let turn_with_cwd = Turn {
            timestamp: Some("2026-04-01T10:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: None,
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input: 1,
                output: 1,
                cache_creation: CacheCreation::default(),
                cache_read: 0,
            }),
            content: None,
            cwd: Some(PathBuf::from("/Users/jasonr/Projects/redis-tui")),
        };
        let session = aggregate(
            Path::new("/tmp/encoded-dir-name"),
            "abc".to_string(),
            vec![turn_with_cwd],
            &pricing::PricingCatalog::empty(),
        )
        .unwrap();
        assert_eq!(session.project_short_name, "redis-tui");
    }

    #[test]
    fn aggregate_short_name_falls_back_to_dir_name() {
        let turn_no_cwd = Turn {
            timestamp: Some("2026-04-01T10:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: None,
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input: 1,
                output: 1,
                cache_creation: CacheCreation::default(),
                cache_read: 0,
            }),
            content: None,
            cwd: None,
        };
        let session = aggregate(
            Path::new("/tmp/-Users-jasonr-encoded"),
            "abc".to_string(),
            vec![turn_no_cwd],
            &pricing::PricingCatalog::empty(),
        )
        .unwrap();
        assert_eq!(session.project_short_name, "-Users-jasonr-encoded");
    }

    // --- discovery ---

    #[test]
    fn discover_finds_jsonl_files_two_levels_deep() {
        let tmp = tempfile::tempdir().unwrap();
        let project_a = tmp.path().join("-Users-alpha");
        let project_b = tmp.path().join("-Users-beta");
        stdfs::create_dir(&project_a).unwrap();
        stdfs::create_dir(&project_b).unwrap();

        for dir in [&project_a, &project_b] {
            stdfs::File::create(dir.join("session1.jsonl")).unwrap();
            stdfs::File::create(dir.join("session2.jsonl")).unwrap();
            // Non-jsonl file — should be skipped.
            stdfs::File::create(dir.join("sessions-index.json")).unwrap();
            // Nested directory — should be skipped at this level.
            stdfs::create_dir(dir.join("subagents")).unwrap();
        }

        let mut result = discover(tmp.path()).unwrap();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(result.len(), 2);
        for (_, jsonl_paths) in &result {
            assert_eq!(jsonl_paths.len(), 2);
            for p in jsonl_paths {
                assert_eq!(p.extension().unwrap(), "jsonl");
            }
        }
    }

    #[test]
    fn discover_handles_empty_projects_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = discover(tmp.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn discover_handles_project_dir_with_no_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("-Users-only-index");
        stdfs::create_dir(&project).unwrap();
        stdfs::File::create(project.join("sessions-index.json")).unwrap();

        let result = discover(tmp.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].1.is_empty());
    }

    // --- rendering ---

    #[test]
    fn truncate_title_under_limit_returns_unchanged() {
        assert_eq!(truncate_title("hello", 80), "hello");
        assert_eq!(truncate_title("", 80), "");
        // Boundary: len == max is also considered "under limit" per the `<=`
        // check, so an exact-80-char title passes through unchanged.
        let exactly_80 = "a".repeat(80);
        assert_eq!(truncate_title(&exactly_80, 80), exactly_80);
    }

    #[test]
    fn truncate_title_over_limit_appends_ellipsis() {
        let long: String = "a".repeat(81);
        let truncated = truncate_title(&long, 80);
        assert_eq!(truncated.chars().count(), 80);
        assert!(truncated.ends_with('…'));
        // First 79 chars should be 'a's.
        assert_eq!(
            truncated.chars().take(79).collect::<String>(),
            "a".repeat(79)
        );
    }

    #[test]
    fn truncate_title_handles_multibyte_chars() {
        // "日本語" is 3 scalars but 9 UTF-8 bytes; truncating by scalar is
        // correct and must not panic on a byte boundary.
        let s = "日本語の説明".to_string() + &"あ".repeat(80);
        let truncated = truncate_title(&s, 10);
        assert_eq!(truncated.chars().count(), 10);
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn truncate_title_collapses_embedded_newlines() {
        // Real JSONL content (e.g. skill preambles) has embedded newlines;
        // unnormalized, comfy-table would render one cell across multiple
        // visual rows.
        assert_eq!(truncate_title("hello\nworld", 80), "hello world");
        assert_eq!(truncate_title("line1\n\nline2", 80), "line1 line2");
    }

    #[test]
    fn truncate_title_collapses_whitespace_runs() {
        assert_eq!(truncate_title("hello  \t  world", 80), "hello world");
        assert_eq!(
            truncate_title("  leading and trailing  ", 80),
            "leading and trailing"
        );
    }

    #[test]
    fn format_local_matches_explicit_chrono_composition() {
        let ts: DateTime<Utc> = "2026-04-01T10:30:00Z".parse().unwrap();
        let expected = ts
            .with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M")
            .to_string();
        assert_eq!(format_local(ts), expected);
    }

    fn session_for_render(
        project: &str,
        title: &str,
        total_billable: u64,
        started_at: &str,
    ) -> Session {
        session_for_render_with_cost(project, title, total_billable, None, started_at)
    }

    fn session_for_render_with_cost(
        project: &str,
        title: &str,
        total_billable: u64,
        total_cost: Option<f64>,
        started_at: &str,
    ) -> Session {
        let ts: DateTime<Utc> = started_at.parse().unwrap();
        Session {
            id: "sid".to_string(),
            project_short_name: project.to_string(),
            started_at: ts,
            last_activity: ts,
            title: title.to_string(),
            turns: Vec::new(),
            total_billable,
            total_cost,
        }
    }

    #[test]
    fn render_table_includes_header_and_rows() {
        let sessions = vec![
            session_for_render("alpha", "first title", 100, "2026-04-01T10:00:00Z"),
            session_for_render("beta", "second title", 250, "2026-04-02T10:00:00Z"),
        ];
        let out = render_table(&sessions);
        assert!(out.contains("datetime"));
        assert!(out.contains("project"));
        assert!(out.contains("title"));
        assert!(out.contains("tokens"));
        assert!(out.contains("id"));
        assert!(out.contains("alpha"));
        assert!(out.contains("beta"));
        assert!(out.contains("first title"));
        assert!(out.contains("second title"));
        assert!(out.contains("100"));
        assert!(out.contains("250"));
        assert!(out.contains("sid"));
    }

    #[test]
    fn render_table_truncates_long_titles_with_ellipsis() {
        // Title of 100 chars — longer than TITLE_MAX_CHARS (80) — should be
        // truncated with a trailing `…` in the rendered output.
        let long_title = "x".repeat(100);
        let sessions = vec![session_for_render(
            "p",
            &long_title,
            1,
            "2026-04-01T10:00:00Z",
        )];
        let out = render_table(&sessions);
        assert!(
            out.contains('…'),
            "expected ellipsis in output, got:\n{out}"
        );
        // Full 100-x run must NOT appear verbatim.
        assert!(!out.contains(&"x".repeat(100)));
    }

    #[test]
    fn render_table_right_aligns_tokens_and_cost_columns() {
        // After Phase 3 the column order is:
        //   datetime | project | title | tokens | cost | id
        // Both rows below have unknown-model totals (None catalog), so
        // every cost cell renders `—`. Stripping `sid` then `—` exposes
        // the tokens column as the new right edge — same alignment
        // verification as before, just two strip steps instead of one.
        let sessions = vec![
            session_for_render("p1", "t1", 9, "2026-04-01T10:00:00Z"),
            session_for_render("p2", "t2", 123_456, "2026-04-02T10:00:00Z"),
        ];
        let out = render_table(&sessions);
        let data_lines: Vec<&str> = out
            .lines()
            .filter(|l| l.contains("p1") || l.contains("p2"))
            .collect();
        assert_eq!(data_lines.len(), 2);
        let strip_trailing = |l: &str| {
            let after_id = l
                .trim_end()
                .strip_suffix("sid")
                .expect("row should end with the hardcoded id column value")
                .trim_end();
            after_id
                .strip_suffix('—')
                .expect("cost cell should be em-dash for unknown-model session")
                .trim_end()
                .to_string()
        };
        let end_of_9 = strip_trailing(data_lines.iter().find(|l| l.contains("p1")).unwrap());
        let end_of_123456 = strip_trailing(data_lines.iter().find(|l| l.contains("p2")).unwrap());
        assert!(end_of_9.ends_with('9'), "got: {end_of_9}");
        assert!(end_of_123456.ends_with("123456"), "got: {end_of_123456}");
        // Right-alignment: shorter value has leading whitespace padding,
        // so the prefix-up-to-tokens has the same scalar count for both.
        assert_eq!(end_of_9.chars().count(), end_of_123456.chars().count());
    }

    // --- show: additional helpers ---

    fn assistant_turn_with_content(
        content: Value,
        input: u64,
        output: u64,
        cache_creation: u64,
    ) -> Turn {
        Turn {
            timestamp: Some("2026-04-22T09:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input,
                output,
                cache_creation: CacheCreation {
                    ephemeral_5m: cache_creation,
                    ephemeral_1h: 0,
                },
                cache_read: 0,
            }),
            content: Some(content),
            cwd: None,
        }
    }

    fn user_turn_role_only(role: Role) -> Turn {
        Turn {
            timestamp: None,
            role,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::String("whatever".to_string())),
            cwd: None,
        }
    }

    // --- is_substantive_user_turn ---

    #[test]
    fn is_substantive_user_turn_accepts_slash_command() {
        let turn = user_string_turn("<command-name>/foo</command-name>");
        assert!(is_substantive_user_turn(&turn));
    }

    #[test]
    fn is_substantive_user_turn_accepts_plain_prose() {
        let turn = user_string_turn("hello");
        assert!(is_substantive_user_turn(&turn));
    }

    #[test]
    fn is_substantive_user_turn_rejects_local_command_caveat() {
        let turn = user_string_turn("<local-command-caveat>heads up</local-command-caveat>");
        assert!(!is_substantive_user_turn(&turn));
    }

    #[test]
    fn is_substantive_user_turn_rejects_local_command_stdout() {
        let turn = user_string_turn("<local-command-stdout>output</local-command-stdout>");
        assert!(!is_substantive_user_turn(&turn));
    }

    #[test]
    fn is_substantive_user_turn_rejects_local_command_stderr() {
        let turn = user_string_turn("<local-command-stderr>oops</local-command-stderr>");
        assert!(!is_substantive_user_turn(&turn));
    }

    #[test]
    fn is_substantive_user_turn_rejects_tool_result_only_array() {
        let content = serde_json::json!([
            { "type": "tool_result", "content": "..." },
        ]);
        assert!(!is_substantive_user_turn(&user_array_turn(content)));
    }

    #[test]
    fn is_substantive_user_turn_accepts_array_with_text_block() {
        let content = serde_json::json!([
            { "type": "tool_result", "content": "..." },
            { "type": "text", "text": "real prose" },
        ]);
        assert!(is_substantive_user_turn(&user_array_turn(content)));
    }

    #[test]
    fn is_substantive_user_turn_rejects_empty_text_block() {
        let content = serde_json::json!([
            { "type": "tool_result", "content": "..." },
            { "type": "text", "text": "" },
        ]);
        assert!(!is_substantive_user_turn(&user_array_turn(content)));
    }

    #[test]
    fn is_substantive_user_turn_rejects_assistant_role() {
        assert!(!is_substantive_user_turn(&user_turn_role_only(
            Role::Assistant,
        )));
    }

    #[test]
    fn is_substantive_user_turn_rejects_skill_injection_array() {
        // Real Claude Code stores skill content as a user turn whose content
        // array has one text block starting with `Base directory for this
        // skill: `. That prefix is in SYNTHETIC_USER_CONTENT_PREFIXES, so
        // the turn must not start a new exchange.
        let content = serde_json::json!([
            {
                "type": "text",
                "text": "Base directory for this skill: /Users/x/.claude/skills/foo\n\n# Foo",
            },
        ]);
        assert!(!is_substantive_user_turn(&user_array_turn(content)));
    }

    #[test]
    fn is_substantive_user_turn_rejects_skill_injection_string() {
        // Defensive: same prefix but in string content (shape not observed
        // in real data yet, but the predicate should still classify it as
        // synthetic).
        let turn = user_string_turn("Base directory for this skill: /Users/x/.claude/skills/foo");
        assert!(!is_substantive_user_turn(&turn));
    }

    // --- group_into_exchanges ---

    #[test]
    fn group_into_exchanges_collapses_tool_loop() {
        let tool_result = serde_json::json!([
            { "type": "tool_result", "tool_use_id": "x", "content": "ok" },
        ]);
        let tool_use = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "x", "input": {} },
        ]);
        let text = serde_json::json!([
            { "type": "text", "text": "done" },
        ]);
        let turns = vec![
            user_string_turn("hello"),
            assistant_turn_with_content(tool_use, 1, 1, 0),
            user_array_turn(tool_result),
            assistant_turn_with_content(text, 1, 1, 0),
        ];
        let exchanges = group_into_exchanges(&turns);
        assert_eq!(exchanges.len(), 1);
        assert_eq!(exchanges[0].assistants.len(), 2);
    }

    #[test]
    fn group_into_exchanges_orphan_final_user() {
        let text = serde_json::json!([{ "type": "text", "text": "reply" }]);
        let turns = vec![
            user_string_turn("first"),
            assistant_turn_with_content(text, 1, 1, 0),
            user_string_turn("second"),
        ];
        let exchanges = group_into_exchanges(&turns);
        assert_eq!(exchanges.len(), 2);
        assert!(exchanges[1].assistants.is_empty());
    }

    #[test]
    fn group_into_exchanges_drops_assistant_before_any_user() {
        let text = serde_json::json!([{ "type": "text", "text": "stray" }]);
        let turns = vec![
            assistant_turn_with_content(text.clone(), 1, 1, 0),
            user_string_turn("hello"),
            assistant_turn_with_content(text, 1, 1, 0),
        ];
        let exchanges = group_into_exchanges(&turns);
        assert_eq!(exchanges.len(), 1);
        assert_eq!(exchanges[0].assistants.len(), 1);
    }

    #[test]
    fn group_into_exchanges_skips_local_command_caveat() {
        let text = serde_json::json!([{ "type": "text", "text": "reply" }]);
        let turns = vec![
            user_string_turn("<local-command-caveat>noise</local-command-caveat>"),
            user_string_turn("real question"),
            assistant_turn_with_content(text, 1, 1, 0),
        ];
        let exchanges = group_into_exchanges(&turns);
        assert_eq!(exchanges.len(), 1);
        assert_eq!(exchanges[0].assistants.len(), 1);
    }

    #[test]
    fn group_into_exchanges_handles_empty_input() {
        let exchanges = group_into_exchanges(&[]);
        assert!(exchanges.is_empty());
    }

    #[test]
    fn group_into_exchanges_absorbs_skill_injection() {
        // Regression for real ~/.claude/projects/ data: slash-command string
        // turn → skill-injection array-text turn → assistant. The skill
        // injection must be absorbed so the slash command stays the user
        // turn of one exchange (not orphaned by the skill injection starting
        // its own).
        let skill_inj = serde_json::json!([
            {
                "type": "text",
                "text": "Base directory for this skill: /x/y\n\nbody",
            },
        ]);
        let reply = serde_json::json!([{ "type": "text", "text": "ok" }]);
        let turns = vec![
            user_string_turn(
                "<command-name>/tw-implement-plan</command-name>\n\
                 <command-args>plan.md</command-args>",
            ),
            user_array_turn(skill_inj),
            assistant_turn_with_content(reply, 100, 50, 0),
        ];
        let exchanges = group_into_exchanges(&turns);
        assert_eq!(exchanges.len(), 1);
        assert_eq!(exchanges[0].assistants.len(), 1);
    }

    // --- user_display_string ---

    #[test]
    fn user_display_string_slash_command_roundtrip() {
        let v = Value::String(
            "<command-name>/foo</command-name><command-args>bar</command-args>".to_string(),
        );
        assert_eq!(user_display_string(&v), Some("/foo bar".to_string()));
    }

    #[test]
    fn user_display_string_plain_prose_returns_as_is() {
        let v = Value::String("hello world".to_string());
        assert_eq!(user_display_string(&v), Some("hello world".to_string()));
    }

    #[test]
    fn user_display_string_array_first_text_block() {
        let v = serde_json::json!([
            { "type": "tool_result", "content": "..." },
            { "type": "text", "text": "hello" },
        ]);
        assert_eq!(user_display_string(&v), Some("hello".to_string()));
    }

    #[test]
    fn user_display_string_returns_none_for_tool_result_only_array() {
        let v = serde_json::json!([
            { "type": "tool_result", "content": "..." },
        ]);
        assert_eq!(user_display_string(&v), None);
    }

    #[test]
    fn user_display_string_returns_none_for_unsupported_value_shapes() {
        assert_eq!(user_display_string(&Value::Null), None);
        assert_eq!(user_display_string(&Value::Bool(true)), None);
        assert_eq!(user_display_string(&serde_json::json!(42)), None);
        assert_eq!(user_display_string(&serde_json::json!({ "k": "v" })), None);
    }

    #[test]
    fn user_display_string_returns_none_for_synthetic_string_content() {
        let v = Value::String(
            "<local-command-caveat>heads up: local</local-command-caveat>".to_string(),
        );
        assert_eq!(user_display_string(&v), None);
    }

    #[test]
    fn user_display_string_returns_none_for_synthetic_array_text_block() {
        let v = serde_json::json!([
            { "type": "text", "text": "Base directory for this skill: /x/y\n\nbody" },
        ]);
        assert_eq!(user_display_string(&v), None);
    }

    #[test]
    fn user_display_string_skips_synthetic_block_and_returns_next() {
        // Hypothetical but defensible: if an array ever contains a synthetic
        // text block followed by a real one, the extractor should skip the
        // synthetic and return the real prose — matching how the show grouper
        // and title extractor agree on "what the user said".
        let v = serde_json::json!([
            { "type": "text", "text": "Base directory for this skill: /x/y" },
            { "type": "text", "text": "real prose" },
        ]);
        assert_eq!(user_display_string(&v), Some("real prose".to_string()));
    }

    #[test]
    fn extract_title_still_passes_after_refactor() {
        // Redundancy guard: confirms the refactor to delegate through
        // `user_display_string` didn't break the existing slash-command case.
        let content = "<command-name>/tw-create-idea</command-name>\n\
                       <command-args>build a CLI</command-args>";
        let turns = vec![user_string_turn(content)];
        assert_eq!(
            extract_title(&turns, "fallback"),
            "/tw-create-idea build a CLI",
        );
    }

    // --- user_content_preview ---

    #[test]
    fn user_content_preview_delegates_to_user_display_string() {
        let content = serde_json::json!([
            { "type": "tool_result", "content": "..." },
            { "type": "text", "text": "hello" },
        ]);
        let turn = user_array_turn(content);
        assert_eq!(user_content_preview(&turn), "hello");
    }

    #[test]
    fn user_content_preview_returns_empty_string_on_unreachable_none() {
        let turn = Turn {
            timestamp: None,
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::Null),
            cwd: None,
        };
        assert_eq!(user_content_preview(&turn), "");
    }

    // --- assistant_cluster_preview ---

    #[test]
    fn assistant_cluster_preview_first_text_block() {
        // First assistant has only a tool_use; second opens with a text block.
        let tu = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "x", "input": {} },
        ]);
        let tx = serde_json::json!([
            { "type": "text", "text": "found it" },
        ]);
        let a1 = assistant_turn_with_content(tu, 0, 0, 0);
        let a2 = assistant_turn_with_content(tx, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a1, &a2];
        assert_eq!(assistant_cluster_preview(&cluster), "found it");
    }

    #[test]
    fn assistant_cluster_preview_falls_back_to_deduped_tool_names() {
        let tu_a = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "1", "input": {} },
            { "type": "tool_use", "name": "Bash", "id": "2", "input": {} },
        ]);
        let tu_b = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "3", "input": {} },
            { "type": "tool_use", "name": "Edit", "id": "4", "input": {} },
        ]);
        let a1 = assistant_turn_with_content(tu_a, 0, 0, 0);
        let a2 = assistant_turn_with_content(tu_b, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a1, &a2];
        assert_eq!(assistant_cluster_preview(&cluster), "Read, Bash, Edit");
    }

    #[test]
    fn assistant_cluster_preview_empty_cluster_returns_empty_string() {
        let cluster: Vec<&Turn> = Vec::new();
        assert_eq!(assistant_cluster_preview(&cluster), "");
    }

    // --- count_tool_uses ---

    #[test]
    fn count_tool_uses_counts_across_cluster() {
        let a_content = serde_json::json!([
            { "type": "tool_use", "name": "Read", "id": "1", "input": {} },
        ]);
        let b_content = serde_json::json!([
            { "type": "text", "text": "..." },
            { "type": "tool_use", "name": "Bash", "id": "2", "input": {} },
            { "type": "tool_use", "name": "Edit", "id": "3", "input": {} },
        ]);
        let a1 = assistant_turn_with_content(a_content, 0, 0, 0);
        let a2 = assistant_turn_with_content(b_content, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a1, &a2];
        assert_eq!(count_tool_uses(&cluster), 3);
    }

    #[test]
    fn count_tool_uses_zero_for_text_only_cluster() {
        let tx = serde_json::json!([{ "type": "text", "text": "hi" }]);
        let a = assistant_turn_with_content(tx, 0, 0, 0);
        let cluster: Vec<&Turn> = vec![&a];
        assert_eq!(count_tool_uses(&cluster), 0);
    }

    // --- render_session ---

    fn show_user_turn(content: &str, ts: &str) -> Turn {
        Turn {
            timestamp: Some(ts.parse().unwrap()),
            role: Role::User,
            model: None,
            message_id: None,
            request_id: None,
            usage: None,
            content: Some(Value::String(content.to_string())),
            cwd: None,
        }
    }

    fn show_assistant_turn(
        content: Value,
        input: u64,
        output: u64,
        cache_creation: u64,
        ts: &str,
    ) -> Turn {
        Turn {
            timestamp: Some(ts.parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input,
                output,
                cache_creation: CacheCreation {
                    ephemeral_5m: cache_creation,
                    ephemeral_1h: 0,
                },
                cache_read: 0,
            }),
            content: Some(content),
            cwd: None,
        }
    }

    #[test]
    fn render_session_header_includes_all_seven_columns() {
        let (out, rows_shown) = render_session(
            &[],
            &pricing::PricingCatalog::empty(),
            &FilterArgs::default(),
        );
        assert_eq!(rows_shown, 0);
        assert!(out.contains("datetime"));
        assert!(out.contains("role"));
        assert!(out.contains("tokens"));
        assert!(out.contains("cost"));
        assert!(out.contains("cumulative"));
        assert!(out.contains("cum_cost"));
        assert!(out.contains("content"));
    }

    #[test]
    fn render_session_orphan_user_shows_em_dash_and_preserves_cumulative() {
        let u1 = show_user_turn("first", "2026-04-01T10:00:00Z");
        let a1 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "reply" }]),
            12000,
            345,
            0,
            "2026-04-01T10:01:00Z",
        );
        let u2 = show_user_turn("orphan", "2026-04-01T10:02:00Z");
        // Assistant row tokens column = 345 (output), cumulative after it =
        // 12000 + 345 = 12345.
        let exchanges = vec![
            Exchange {
                user: &u1,
                assistants: vec![&a1],
            },
            Exchange {
                user: &u2,
                assistants: Vec::new(),
            },
        ];
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            &FilterArgs::default(),
        );
        let lines: Vec<&str> = out.lines().collect();
        let assistant_line = lines
            .iter()
            .find(|l| l.contains("reply"))
            .expect("assistant row missing");
        let orphan_line = lines
            .iter()
            .find(|l| l.contains("orphan"))
            .expect("orphan row missing");

        // Orphan row's tokens column shows the em-dash and cumulative is
        // preserved from the prior row.
        assert!(orphan_line.contains('—'));
        assert!(orphan_line.contains("12345"));

        // Strip the trailing content column AND the cum_cost column so
        // the cumulative column sits at the new right edge. Both rows
        // here use an empty catalog, so every cum_cost cell renders `—`.
        // Pop content first (variable text), then pop the trailing `—`,
        // then we can pin on the numeric `cumulative` value.
        let strip_to_cumulative = |l: &str, content_marker: &str| {
            let trimmed = l.trim_end();
            let cut = trimmed
                .rfind(content_marker)
                .expect("content marker should be in the content column");
            let after_content = trimmed[..cut].trim_end();
            after_content
                .strip_suffix('—')
                .expect("cum_cost should be em-dash for unknown-model session")
                .trim_end()
                .to_string()
        };
        let a_cols = strip_to_cumulative(assistant_line, "reply");
        let o_cols = strip_to_cumulative(orphan_line, "orphan");
        assert!(a_cols.ends_with("12345"), "got: {a_cols}");
        assert!(o_cols.ends_with("12345"), "got: {o_cols}");
        // Right-aligned: the trailing cumulative columns end at the same
        // character position, so the preceding lines (through tokens + spaces +
        // cumulative) have the same scalar count. Compare by chars, not bytes,
        // because `—` is 1 scalar but 3 UTF-8 bytes — `.len()` would report a
        // spurious 2-byte difference even when columns are visually aligned.
        assert_eq!(a_cols.chars().count(), o_cols.chars().count());
    }

    #[test]
    fn render_session_cumulative_reaches_sum_of_billable() {
        let u1 = show_user_turn("q1", "2026-04-01T10:00:00Z");
        let a1 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "r1" }]),
            100,
            50,
            200,
            "2026-04-01T10:01:00Z",
        );
        let u2 = show_user_turn("q2", "2026-04-01T10:02:00Z");
        let a2 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "r2" }]),
            10,
            20,
            0,
            "2026-04-01T10:03:00Z",
        );
        let exchanges = vec![
            Exchange {
                user: &u1,
                assistants: vec![&a1],
            },
            Exchange {
                user: &u2,
                assistants: vec![&a2],
            },
        ];
        // Expected billable total = (100+50+200) + (10+20+0) = 380.
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            &FilterArgs::default(),
        );
        let last_line = out
            .lines()
            .rfind(|l| l.contains("r2"))
            .expect("final assistant row missing");
        assert!(
            last_line.contains(" 380"),
            "expected final cumulative 380 on last assistant row; got: {last_line}",
        );
    }

    #[test]
    fn render_session_right_aligns_numeric_columns() {
        let u1 = show_user_turn("small", "2026-04-01T10:00:00Z");
        let a1 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "s" }]),
            0,
            9,
            0,
            "2026-04-01T10:01:00Z",
        );
        let u2 = show_user_turn("big", "2026-04-01T10:02:00Z");
        let a2 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "b" }]),
            0,
            123_456,
            0,
            "2026-04-01T10:03:00Z",
        );
        let exchanges = vec![
            Exchange {
                user: &u1,
                assistants: vec![&a1],
            },
            Exchange {
                user: &u2,
                assistants: vec![&a2],
            },
        ];
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            &FilterArgs::default(),
        );
        // After Phase 3 the column order is:
        //   datetime | role | tokens | cost | cumulative | cum_cost | content
        // With an empty catalog, both `cost` and `cum_cost` render as
        // `—`. Strip content, then the trailing `—` (cum_cost) so the
        // cumulative column sits at the right edge.
        let a_small = out
            .lines()
            .find(|l| l.trim_end().ends_with(" s"))
            .expect("small assistant row missing");
        let a_big = out
            .lines()
            .find(|l| l.trim_end().ends_with(" b"))
            .expect("big assistant row missing");
        let strip_to_cumulative = |l: &str, tail: &str| {
            let after_content = l.trim_end().strip_suffix(tail).unwrap_or(l).trim_end();
            after_content
                .strip_suffix('—')
                .expect("cum_cost should be em-dash with empty catalog")
                .trim_end()
                .to_string()
        };
        let small_cols = strip_to_cumulative(a_small, "s");
        let big_cols = strip_to_cumulative(a_big, "b");
        // Cumulative column values: 9 for small row, 9 + 123456 = 123465 for
        // big row. Widths of tokens + cost + cumulative parts must
        // match (all three are right-aligned to the same boundaries).
        assert!(small_cols.ends_with('9'), "got: {small_cols}");
        assert!(big_cols.ends_with("123465"), "got: {big_cols}");
        assert_eq!(small_cols.chars().count(), big_cols.chars().count());
    }

    #[test]
    fn render_session_tool_use_suffix_appears_on_assistant_row() {
        let u = show_user_turn("q", "2026-04-01T10:00:00Z");
        let a = show_assistant_turn(
            serde_json::json!([
                { "type": "text", "text": "reading" },
                { "type": "tool_use", "name": "Read", "id": "1", "input": {} },
                { "type": "tool_use", "name": "Bash", "id": "2", "input": {} },
            ]),
            0,
            1,
            0,
            "2026-04-01T10:01:00Z",
        );
        let exchanges = vec![Exchange {
            user: &u,
            assistants: vec![&a],
        }];
        let (out, _) = render_session(
            &exchanges,
            &pricing::PricingCatalog::empty(),
            &FilterArgs::default(),
        );
        assert!(
            out.contains("reading +2 tool uses"),
            "expected tool-use suffix; got:\n{out}",
        );
    }

    // --- FilterArgs ---

    #[test]
    fn filter_args_matches_no_filters_passes_everything() {
        let f = FilterArgs::default();
        assert!(f.matches(0, None));
        assert!(f.matches(0, Some(0.0)));
        assert!(f.matches(1_000_000, Some(1.0)));
    }

    #[test]
    fn filter_args_matches_min_tokens_only() {
        let f = FilterArgs {
            min_tokens: Some(100),
            min_cost: None,
        };
        // Boundary: at threshold passes (>=).
        assert!(f.matches(100, None));
        assert!(f.matches(100, Some(0.0)));
        // Above threshold passes regardless of cost.
        assert!(f.matches(101, None));
        // Below threshold fails regardless of cost.
        assert!(!f.matches(99, Some(1.0)));
        assert!(!f.matches(0, None));
    }

    #[test]
    fn filter_args_matches_min_cost_only() {
        let f = FilterArgs {
            min_tokens: None,
            min_cost: Some(0.50),
        };
        // Boundary: at threshold passes.
        assert!(f.matches(0, Some(0.50)));
        assert!(f.matches(1_000_000, Some(0.50)));
        // Above threshold passes.
        assert!(f.matches(0, Some(0.51)));
        // Below threshold fails.
        assert!(!f.matches(1_000_000, Some(0.49)));
        // None cost is excluded by any active --min-cost.
        assert!(!f.matches(1_000_000, None));
        // Some(0.0) is excluded by any positive threshold.
        assert!(!f.matches(1_000_000, Some(0.0)));
        // ...but accepted by a 0.0 threshold.
        let zero = FilterArgs {
            min_tokens: None,
            min_cost: Some(0.0),
        };
        assert!(zero.matches(0, Some(0.0)));
        // None still excluded even by 0.0 threshold (None means
        // "unpriceable", which can't clear a cost gate).
        assert!(!zero.matches(0, None));
    }

    #[test]
    fn filter_args_matches_both_is_logical_and() {
        let f = FilterArgs {
            min_tokens: Some(100),
            min_cost: Some(0.10),
        };
        // Both clear → pass.
        assert!(f.matches(100, Some(0.10)));
        assert!(f.matches(500, Some(1.00)));
        // Tokens fail, cost clears → fail.
        assert!(!f.matches(50, Some(1.00)));
        // Tokens clear, cost fails → fail.
        assert!(!f.matches(500, Some(0.05)));
        // Tokens clear, cost is None → fail.
        assert!(!f.matches(500, None));
        // Both fail → fail.
        assert!(!f.matches(0, None));
    }

    #[test]
    fn filter_args_describe_active_formats_each_combination() {
        let tokens_only = FilterArgs {
            min_tokens: Some(50_000),
            min_cost: None,
        };
        assert_eq!(tokens_only.describe_active(), "--min-tokens 50000");

        let cost_only = FilterArgs {
            min_tokens: None,
            min_cost: Some(0.50),
        };
        assert_eq!(cost_only.describe_active(), "--min-cost 0.5");

        let both = FilterArgs {
            min_tokens: Some(50_000),
            min_cost: Some(0.50),
        };
        assert_eq!(both.describe_active(), "--min-tokens 50000 --min-cost 0.5",);

        // Regression guard: the default `{}` formatter must round-trip
        // small values faithfully — `{:.2}` would truncate this to
        // `--min-cost 0.00`.
        let small = FilterArgs {
            min_tokens: None,
            min_cost: Some(0.0001),
        };
        assert!(
            small.describe_active().contains("--min-cost 0.0001"),
            "expected 0.0001 to round-trip; got: {}",
            small.describe_active(),
        );
    }

    // --- exchange_filter_totals ---

    #[test]
    fn exchange_filter_totals_orphan_is_zero_and_none() {
        let u = show_user_turn("orphan", "2026-04-01T10:00:00Z");
        let exchange = Exchange {
            user: &u,
            assistants: Vec::new(),
        };
        let (tokens, cost) = exchange_filter_totals(&exchange, &pricing::PricingCatalog::empty());
        assert_eq!(tokens, 0);
        assert_eq!(cost, None);
    }

    #[test]
    fn exchange_filter_totals_sums_billable_across_cluster() {
        // Two assistant turns on a known model. Use the litellm-mini
        // fixture's claude-opus-4-7 rates: input=15e-6, output=75e-6,
        // cache_creation=18.75e-6, cache_read=1.5e-6.
        let u = show_user_turn("q", "2026-04-01T10:00:00Z");
        let a1 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "a1" }]),
            10,
            20,
            0,
            "2026-04-01T10:01:00Z",
        );
        let a2 = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "a2" }]),
            5,
            15,
            100,
            "2026-04-01T10:02:00Z",
        );
        let exchange = Exchange {
            user: &u,
            assistants: vec![&a1, &a2],
        };
        let catalog = sample_catalog();
        let (tokens, cost) = exchange_filter_totals(&exchange, &catalog);
        // Tokens = (10+20+0) + (5+15+100) = 30 + 120 = 150.
        assert_eq!(tokens, 150);
        // Cost = a1 (10*15e-6 + 20*75e-6) + a2 (5*15e-6 + 15*75e-6 + 100*18.75e-6)
        //      = (0.00015 + 0.0015) + (0.000075 + 0.001125 + 0.001875)
        //      = 0.00165 + 0.003075
        //      = 0.004725
        let actual = cost.expect("cost should be Some");
        assert!(
            (actual - 0.004_725).abs() < 1e-9,
            "expected ~0.004725, got {actual}",
        );
    }

    #[test]
    fn exchange_filter_totals_unknown_model_collapses_cost() {
        // Mix one known-model assistant with one unknown-model one;
        // tokens still sum but cost collapses to None.
        let u = show_user_turn("q", "2026-04-01T10:00:00Z");
        let known = show_assistant_turn(
            serde_json::json!([{ "type": "text", "text": "known" }]),
            10,
            20,
            0,
            "2026-04-01T10:01:00Z",
        );
        let unknown = Turn {
            timestamp: Some("2026-04-01T10:02:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-fake-9-9".to_string()),
            message_id: None,
            request_id: None,
            usage: Some(Usage {
                input: 5,
                output: 15,
                cache_creation: CacheCreation::default(),
                cache_read: 0,
            }),
            content: Some(serde_json::json!([{ "type": "text", "text": "?" }])),
            cwd: None,
        };
        let exchange = Exchange {
            user: &u,
            assistants: vec![&known, &unknown],
        };
        let catalog = sample_catalog();
        let (tokens, cost) = exchange_filter_totals(&exchange, &catalog);
        // Tokens = (input+output+cache_creation) per turn = (10+20)+(5+15);
        // both clusters have zero cache_creation here.
        assert_eq!(tokens, (10 + 20) + (5 + 15));
        assert_eq!(cost, None);
    }

    /// Build a small in-memory pricing catalog matching the
    /// `litellm-mini.json` fixture's claude-opus-4-7 rates. Avoids a
    /// disk fetch in unit tests.
    fn sample_catalog() -> pricing::PricingCatalog {
        let raw = r#"{
            "claude-opus-4-7": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075,
                "cache_creation_input_token_cost": 0.00001875,
                "cache_read_input_token_cost": 0.0000015
            }
        }"#;
        pricing::PricingCatalog::from_raw_json(raw).expect("test catalog parse")
    }
}
