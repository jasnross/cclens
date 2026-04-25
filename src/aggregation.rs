//! Fold parsed `Turn`s into typed `Session`s and `Exchange`s.
//!
//! Public API:
//! - `aggregate(&Path, String, Vec<Turn>, &PricingCatalog) -> Option<Session>`
//!   — turn list → session, with title extraction, project-short-name
//!   derivation, billable summation, strict cost folding, and the
//!   zero-billable + zero-cost filter.
//! - `Exchange<'a>` — a substantive user turn plus its assistant
//!   cluster.
//! - `group_into_exchanges(&[Turn]) -> Vec<Exchange<'_>>`.
//! - `exchange_filter_totals(&Exchange<'_>, &PricingCatalog) -> (u64, Option<f64>)`.
//! - `user_display_string(&Value) -> Option<String>` — the canonical
//!   "what the user said" extractor; called by `extract_title` here and
//!   by the rendering layer's `user_content_preview` (currently in
//!   `main.rs`; promoted to its own library module in Phase 7).

use std::path::Path;

use serde_json::Value;

use crate::domain::{Role, Session, Turn, Usage};
use crate::pricing::PricingCatalog;

// ---- session aggregation ----

#[must_use]
pub fn aggregate(
    project_dir: &Path,
    session_id: String,
    turns: Vec<Turn>,
    catalog: &PricingCatalog,
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
fn total_session_cost(turns: &[Turn], catalog: &PricingCatalog) -> Option<f64> {
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

// ---- title extraction ----

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
#[must_use]
pub fn user_display_string(content: &Value) -> Option<String> {
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

// ---- exchanges ----

/// A user turn plus the cluster of consecutive assistant turns that follow it
/// (up to the next substantive user turn). The `assistants` vec is empty when
/// the user turn is orphaned at the end of a session with no response.
#[derive(Debug)]
pub struct Exchange<'a> {
    pub user: &'a Turn,
    pub assistants: Vec<&'a Turn>,
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

#[must_use]
pub fn group_into_exchanges(turns: &[Turn]) -> Vec<Exchange<'_>> {
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
#[must_use]
pub fn exchange_filter_totals(
    exchange: &Exchange<'_>,
    catalog: &PricingCatalog,
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::domain::CacheCreation;

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

    /// Build a small in-memory pricing catalog matching the
    /// `litellm-mini.json` fixture's claude-opus-4-7 rates. Avoids a
    /// disk fetch in unit tests.
    fn sample_catalog() -> PricingCatalog {
        let raw = r#"{
            "claude-opus-4-7": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075,
                "cache_creation_input_token_cost": 0.00001875,
                "cache_read_input_token_cost": 0.0000015
            }
        }"#;
        PricingCatalog::from_raw_json(raw).expect("test catalog parse")
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
            &PricingCatalog::empty(),
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
            &PricingCatalog::empty(),
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
            &PricingCatalog::empty(),
        )
        .expect("unknown-model zero-billable session must NOT be filtered out");
        assert_eq!(session.total_billable, 0);
        assert_eq!(session.total_cost, None);
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
            &PricingCatalog::empty(),
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
            &PricingCatalog::empty(),
        )
        .unwrap();
        assert_eq!(session.project_short_name, "-Users-jasonr-encoded");
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

    // --- exchange_filter_totals ---

    #[test]
    fn exchange_filter_totals_orphan_is_zero_and_none() {
        let u = show_user_turn("orphan", "2026-04-01T10:00:00Z");
        let exchange = Exchange {
            user: &u,
            assistants: Vec::new(),
        };
        let (tokens, cost) = exchange_filter_totals(&exchange, &PricingCatalog::empty());
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
}
