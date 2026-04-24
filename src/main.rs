use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use comfy_table::presets::NOTHING;
use comfy_table::{CellAlignment, Table};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---- cli ----

#[derive(Parser)]
#[command(name = "cclens", about = "Browse Claude Code conversations")]
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
    List,
    /// Show per-exchange token breakdown for one session.
    Show {
        /// Full session UUID (matches a .jsonl filename stem under --projects-dir).
        session_id: String,
    },
}

fn default_projects_dir() -> PathBuf {
    dirs::home_dir().map_or_else(
        || PathBuf::from(".claude/projects"),
        |h| h.join(".claude/projects"),
    )
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::List) {
        Command::List => run_list(&cli.projects_dir),
        Command::Show { session_id } => run_show(&cli.projects_dir, &session_id),
    }
}

fn run_list(projects_dir: &Path) -> anyhow::Result<()> {
    let project_entries = discover(projects_dir)?;
    let mut sessions = Vec::new();
    for (project_dir, jsonl_paths) in project_entries {
        for jsonl_path in jsonl_paths {
            // A single unreadable file should not abort the whole listing.
            let Ok(turns) = parse_jsonl(&jsonl_path) else {
                continue;
            };
            let session_id = jsonl_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let Some(session) = aggregate(&project_dir, session_id, turns) {
                sessions.push(session);
            }
        }
    }
    sessions.sort_by_key(|s| s.started_at);
    println!("{}", render_table(&sessions));
    Ok(())
}

fn run_show(projects_dir: &Path, session_id: &str) -> anyhow::Result<()> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        anyhow::bail!("session id must not be empty");
    }
    let project_entries = discover(projects_dir)?;
    let mut matches: Vec<PathBuf> = Vec::new();
    for (_project_dir, jsonl_paths) in project_entries {
        for jsonl_path in jsonl_paths {
            let stem = jsonl_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if stem == session_id {
                matches.push(jsonl_path);
            }
        }
    }
    match matches.as_slice() {
        [] => anyhow::bail!("no session matches id {session_id}"),
        [path] => {
            let turns = parse_jsonl(path)?;
            let exchanges = group_into_exchanges(&turns);
            println!("{}", render_session(&exchanges));
            Ok(())
        }
        paths => {
            let joined = paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join("\n  ");
            anyhow::bail!("multiple sessions match id {session_id}:\n  {joined}")
        }
    }
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
    cache_creation: u64,
    cache_read: u64,
}

impl Usage {
    fn billable(&self) -> u64 {
        self.input + self.output + self.cache_creation
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
}

#[derive(Deserialize)]
struct RawMessage {
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
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

fn into_usage(raw: &RawUsage) -> Usage {
    Usage {
        input: raw.input_tokens,
        output: raw.output_tokens,
        cache_creation: raw.cache_creation_input_tokens,
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
    let (model, usage, content) = raw.message.map_or((None, None, None), |msg| {
        (msg.model, msg.usage.as_ref().map(into_usage), msg.content)
    });
    Some(Turn {
        timestamp: raw.timestamp,
        role,
        model,
        usage,
        content,
        cwd: raw.cwd,
    })
}

// ---- aggregation ----

fn aggregate(project_dir: &Path, session_id: String, turns: Vec<Turn>) -> Option<Session> {
    let total_billable: u64 = turns.iter().filter_map(billable_from_turn).sum();
    if total_billable == 0 {
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
    })
}

fn billable_from_turn(turn: &Turn) -> Option<u64> {
    match &turn.role {
        Role::Assistant => turn.usage.as_ref().map(Usage::billable),
        Role::User | Role::Attachment | Role::System | Role::Other(_) => None,
    }
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
        result.push((path, jsonl_paths));
    }
    Ok(result)
}

// ---- rendering ----

const TITLE_MAX_CHARS: usize = 80;

// Index of the tokens column in the header vector below. Kept as a const so
// reordering columns requires updating one place, not two.
const TOKENS_COL_INDEX: usize = 3;

// Indices of the numeric columns in `render_session`'s header:
//   vec!["datetime", "role", "tokens", "cumulative", "content"]
//                             idx 2    idx 3
// Matches the `TOKENS_COL_INDEX` precedent above.
const SHOW_TOKENS_COL_INDEX: usize = 2;
const SHOW_CUMULATIVE_COL_INDEX: usize = 3;

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
    table.set_header(vec!["datetime", "project", "title", "tokens", "id"]);
    for session in sessions {
        table.add_row(vec![
            format_local(session.started_at),
            session.project_short_name.clone(),
            truncate_title(&session.title, TITLE_MAX_CHARS),
            session.total_billable.to_string(),
            session.id.clone(),
        ]);
    }
    // column_mut returns Option; the column is guaranteed present because the
    // header above defines the tokens column at TOKENS_COL_INDEX.
    if let Some(col) = table.column_mut(TOKENS_COL_INDEX) {
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

fn render_session(exchanges: &[Exchange<'_>]) -> String {
    let mut table = Table::new();
    table.load_preset(NOTHING);
    table.set_header(vec!["datetime", "role", "tokens", "cumulative", "content"]);
    let mut cumulative: u64 = 0;

    for exchange in exchanges {
        let user_tokens_opt: Option<u64> = if exchange.assistants.is_empty() {
            None
        } else {
            let sum = exchange
                .assistants
                .iter()
                .filter_map(|t| t.usage.as_ref())
                .map(|u| u.input + u.cache_creation)
                .sum();
            Some(sum)
        };
        let output_tokens: u64 = exchange
            .assistants
            .iter()
            .filter_map(|t| t.usage.as_ref())
            .map(|u| u.output)
            .sum();

        cumulative += user_tokens_opt.unwrap_or(0);
        table.add_row(vec![
            format_local_or_empty(exchange.user.timestamp),
            "user".to_string(),
            user_tokens_opt.map_or_else(|| "—".to_string(), |n| n.to_string()),
            cumulative.to_string(),
            truncate_title(&user_content_preview(exchange.user), TITLE_MAX_CHARS),
        ]);

        if let Some(first_assistant) = exchange.assistants.first() {
            cumulative += output_tokens;
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
                cumulative.to_string(),
                truncate_title(&content, TITLE_MAX_CHARS),
            ]);
        }
    }

    if let Some(col) = table.column_mut(SHOW_TOKENS_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    if let Some(col) = table.column_mut(SHOW_CUMULATIVE_COL_INDEX) {
        col.set_cell_alignment(CellAlignment::Right);
    }
    format!("{table}")
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
            usage: None,
            content: Some(content),
            cwd: None,
        }
    }

    fn assistant_turn_with_usage(input: u64, output: u64, cache_creation: u64) -> Turn {
        Turn {
            timestamp: Some("2026-04-01T10:00:00Z".parse().unwrap()),
            role: Role::Assistant,
            model: Some("claude-opus-4-7".to_string()),
            usage: Some(Usage {
                input,
                output,
                cache_creation,
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
        assert!(matches!(cli.command, Some(Command::List)));
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
            cache_creation: 18998,
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
        let result = aggregate(Path::new("/tmp/fake-project"), "abc".to_string(), turns);
        assert!(result.is_none());
    }

    #[test]
    fn aggregate_sums_all_assistant_turns() {
        let turns = vec![
            user_string_turn("q1"),
            assistant_turn_with_usage(100, 0, 0),
            assistant_turn_with_usage(200, 0, 0),
        ];
        let session = aggregate(Path::new("/tmp/fake-project"), "abc".to_string(), turns)
            .expect("zero-billable filter should not fire");
        assert_eq!(session.total_billable, 300);
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
            usage: Some(Usage {
                input: 1,
                output: 1,
                cache_creation: 0,
                cache_read: 0,
            }),
            content: None,
            cwd: Some(PathBuf::from("/Users/jasonr/Projects/redis-tui")),
        };
        let session = aggregate(
            Path::new("/tmp/encoded-dir-name"),
            "abc".to_string(),
            vec![turn_with_cwd],
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
            usage: Some(Usage {
                input: 1,
                output: 1,
                cache_creation: 0,
                cache_read: 0,
            }),
            content: None,
            cwd: None,
        };
        let session = aggregate(
            Path::new("/tmp/-Users-jasonr-encoded"),
            "abc".to_string(),
            vec![turn_no_cwd],
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
        let ts: DateTime<Utc> = started_at.parse().unwrap();
        Session {
            id: "sid".to_string(),
            project_short_name: project.to_string(),
            started_at: ts,
            last_activity: ts,
            title: title.to_string(),
            turns: Vec::new(),
            total_billable,
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
    fn render_table_right_aligns_tokens_column() {
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
        // Both rows share the same trailing `id` column value ("sid" from
        // session_for_render), so stripping it leaves the tokens column as the
        // new right edge — which is where right-alignment is observable.
        let strip_trailing_id = |l: &str| {
            l.trim_end()
                .strip_suffix("sid")
                .expect("row should end with the hardcoded id column value")
                .trim_end()
                .to_string()
        };
        let end_of_9 = strip_trailing_id(data_lines.iter().find(|l| l.contains("p1")).unwrap());
        let end_of_123456 =
            strip_trailing_id(data_lines.iter().find(|l| l.contains("p2")).unwrap());
        assert!(end_of_9.ends_with('9'));
        assert!(end_of_123456.ends_with("123456"));
        // Right-alignment: the shorter value has leading whitespace padding
        // inside its column, so the line is the same length as the longer one.
        assert_eq!(end_of_9.len(), end_of_123456.len());
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
            usage: Some(Usage {
                input,
                output,
                cache_creation,
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
            usage: Some(Usage {
                input,
                output,
                cache_creation,
                cache_read: 0,
            }),
            content: Some(content),
            cwd: None,
        }
    }

    #[test]
    fn render_session_header_includes_all_five_columns() {
        let out = render_session(&[]);
        assert!(out.contains("datetime"));
        assert!(out.contains("role"));
        assert!(out.contains("tokens"));
        assert!(out.contains("cumulative"));
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
        let out = render_session(&exchanges);
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

        // Strip the trailing content column so the numeric columns sit at the
        // right edge, then verify em-dash and numeric strings right-align to
        // the same character position.
        let strip_content = |l: &str, marker: &str| {
            let trimmed = l.trim_end();
            let cut = trimmed
                .rfind(marker)
                .expect("marker should be in the content column");
            trimmed[..cut].trim_end().to_string()
        };
        let a_cols = strip_content(assistant_line, "reply");
        let o_cols = strip_content(orphan_line, "orphan");
        assert!(a_cols.ends_with("12345"));
        assert!(o_cols.ends_with("12345"));
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
        let out = render_session(&exchanges);
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
        let out = render_session(&exchanges);
        // The two assistant rows have content 's' and 'b'. Strip the content
        // column so the cumulative column sits at the right edge.
        let a_small = out
            .lines()
            .find(|l| l.trim_end().ends_with(" s"))
            .expect("small assistant row missing");
        let a_big = out
            .lines()
            .find(|l| l.trim_end().ends_with(" b"))
            .expect("big assistant row missing");
        let strip_tail = |l: &str, tail: &str| {
            l.trim_end()
                .strip_suffix(tail)
                .unwrap_or(l)
                .trim_end()
                .to_string()
        };
        let small_cols = strip_tail(a_small, "s");
        let big_cols = strip_tail(a_big, "b");
        // Cumulative column values: 9 for small row, 9 + 123456 = 123465 for
        // big row. Widths of tokens + cumulative parts must match (right-
        // aligned to the same column boundaries).
        assert!(small_cols.ends_with('9'));
        assert!(big_cols.ends_with("123465"));
        assert_eq!(small_cols.len(), big_cols.len());
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
        let out = render_session(&exchanges);
        assert!(
            out.contains("reading +2 tool uses"),
            "expected tool-use suffix; got:\n{out}",
        );
    }
}
