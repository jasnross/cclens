use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
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
    eprintln!("parsed {} sessions", sessions.len());
    Ok(())
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

// Harness-synthetic user-content wrappers that should never be treated as the
// user's intent. Real data has at least these three; the plan spec'd only the
// caveat, but the other two occur on every slash-command session.
const SYNTHETIC_USER_CONTENT_PREFIXES: &[&str] = &[
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
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

        if let Some(s) = content.as_str() {
            if SYNTHETIC_USER_CONTENT_PREFIXES
                .iter()
                .any(|p| s.starts_with(p))
            {
                continue;
            }
            if let Some(title) = extract_slash_command_title(s) {
                return title;
            }
            return s.to_string();
        }

        if let Some(arr) = content.as_array() {
            for block in arr {
                if block.get("type").and_then(Value::as_str) != Some("text") {
                    continue;
                }
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    return text.to_string();
                }
            }
        }
    }
    session_id.to_string()
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

// ---- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs as stdfs;
    use std::io::Write;

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
}
