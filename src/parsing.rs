//! On-disk JSONL schema deserialization and translation to typed `Turn`s.
//!
//! Public API:
//! - `ParsedTranscript` — one file's turns plus `continued_in`, the
//!   session ids its `continued-in` records name, in file order.
//! - `parse_jsonl(&Path) -> anyhow::Result<ParsedTranscript>` —
//!   streaming per-line parser; silently skips malformed lines. Returns
//!   one turn per assistant message: lines sharing
//!   `(message_id, request_id)` merge at the first line's position,
//!   with the last usage-bearing line's usage and every line's content
//!   blocks in file order. A line repeating a `uuid` already merged
//!   into the message is skipped: it adds neither blocks nor usage.
//!   No two assistant turns in one result share that key. Every
//!   non-assistant line stays its own turn.
//!
//! All `Raw*` types and the `into_usage` / `raw_to_turn` adapters are
//! module-private — the public surface is just the entry point and its
//! result type. If a consumer wants to hand-build a `Turn` it should
//! construct one from `domain` directly rather than reaching into the
//! deserialization layer.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::domain::{CacheCreation, Role, Turn, TurnOrigin, Usage};

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
    /// Top-level line key, not a field of `message` — the harness
    /// records the reasoning effort alongside `type` and `timestamp`.
    #[serde(default)]
    effort: Option<String>,
    #[serde(default)]
    uuid: Option<String>,
    /// Set only on a `continued-in` record: the session this file's
    /// conversation continued in.
    #[serde(rename = "continuedInSessionId", default)]
    continued_in_session_id: Option<String>,
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

/// One transcript file's parsed contents.
#[derive(Debug)]
pub struct ParsedTranscript {
    /// One turn per assistant message and per non-assistant line, in
    /// file order.
    pub turns: Vec<Turn>,
    /// Session ids this file's `continued-in` records name, in file
    /// order.
    pub continued_in: Vec<String>,
}

/// # Errors
///
/// Returns an error if the file at `path` cannot be opened, or if
/// reading a line fails with any error other than invalid UTF-8; the
/// turns parsed before it are discarded. An invalid-UTF-8 line, a
/// JSON-parse failure, and a line that translates to no `Turn` (e.g.
/// missing `type`) are silently skipped.
pub fn parse_jsonl(path: &Path) -> anyhow::Result<ParsedTranscript> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut turns: Vec<Turn> = Vec::new();
    let mut continued_in = Vec::new();
    let mut messages: HashMap<(String, String), MergedMessage> = HashMap::new();
    for line_result in reader.lines() {
        let text = match line_result {
            Ok(text) => text,
            // The reader has already consumed an invalid-UTF-8 line,
            // so skipping it moves on to the next one.
            Err(e) if e.kind() == io::ErrorKind::InvalidData => continue,
            // Any other read error makes the file unreadable: retrying
            // can loop forever (a directory yields `IsADirectory` on
            // every read).
            Err(e) => return Err(e.into()),
        };
        if text.trim().is_empty() {
            continue;
        }
        let Ok(raw) = serde_json::from_str::<RawLine>(&text) else {
            continue;
        };
        if raw.line_type.as_deref() == Some("continued-in")
            && let Some(id) = &raw.continued_in_session_id
        {
            continued_in.push(id.clone());
        }
        let Some(turn) = raw_to_turn(raw) else {
            continue;
        };
        if let Some(key) = message_key(&turn) {
            if let Some(message) = messages.get_mut(&key)
                && let Some(merged) = turns.get_mut(message.position)
            {
                // A line whose `uuid` already folded into this message
                // repeats it: its blocks are already there, and its
                // usage may predate a later line's.
                let repeated = turn
                    .uuid
                    .as_ref()
                    .is_some_and(|uuid| !message.line_uuids.insert(uuid.clone()));
                if !repeated {
                    merge_line(merged, turn);
                }
                continue;
            }
            messages.insert(
                key,
                MergedMessage {
                    position: turns.len(),
                    line_uuids: turn.uuid.iter().cloned().collect(),
                },
            );
        }
        turns.push(turn);
    }
    Ok(ParsedTranscript {
        turns,
        continued_in,
    })
}

/// Where one assistant message's merged turn sits in `turns`, and the
/// line `uuid`s already folded into it.
struct MergedMessage {
    position: usize,
    line_uuids: HashSet<String>,
}

/// The `(message_id, request_id)` pair that ties an assistant
/// message's lines together. `None` for any other role, and for an
/// assistant line missing either id, which therefore never merges.
fn message_key(turn: &Turn) -> Option<(String, String)> {
    match turn.role {
        Role::Assistant => Some((turn.message_id.clone()?, turn.request_id.clone()?)),
        Role::User | Role::Attachment | Role::System | Role::Other(_) => None,
    }
}

/// Fold a later line of one assistant message into the turn its first
/// line produced. Content blocks append in file order; usage is the
/// latest line that carries any, since each line repeats the message's
/// usage and only the last holds the final output count. Every other
/// field keeps the first line's value.
fn merge_line(merged: &mut Turn, line: Turn) {
    let mut blocks = content_blocks(merged.content.take());
    blocks.extend(content_blocks(line.content));
    merged.content = (!blocks.is_empty()).then_some(Value::Array(blocks));
    if line.usage.is_some() {
        merged.usage = line.usage;
    }
}

/// A line's `content` as a list of blocks, so lines of one message
/// can be concatenated: a string becomes one text block, an absent or
/// null value contributes nothing, and any other non-array value
/// becomes a single element.
fn content_blocks(content: Option<Value>) -> Vec<Value> {
    match content {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(blocks)) => blocks,
        Some(Value::String(text)) => vec![json!({"type": "text", "text": text})],
        Some(other) => vec![other],
    }
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
        effort: raw.effort,
        message_id,
        request_id: raw.request_id,
        uuid: raw.uuid,
        usage,
        content,
        cwd: raw.cwd,
        // Parser produces parent-origin turns; the loading walk
        // overrides this for subagent transcripts after parsing.
        origin: TurnOrigin::default(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs as stdfs;
    use std::io::Write;

    use super::*;

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

        let turns = parse_jsonl(&path).unwrap().turns;
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
    fn parse_jsonl_reads_top_level_effort() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("session.jsonl");
        let mut f = stdfs::File::create(&path).unwrap();
        writeln!(
            f,
            "{{\"type\":\"assistant\",\"timestamp\":\"2026-04-01T10:00:00Z\",\"effort\":\"high\",\"message\":{{\"model\":\"claude-opus-5\"}}}}",
        )
        .unwrap();
        drop(f);

        let turns = parse_jsonl(&path).unwrap().turns;
        assert_eq!(turns[0].effort.as_deref(), Some("high"));
    }

    #[test]
    fn parse_jsonl_tolerates_absent_effort() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("session.jsonl");
        let mut f = stdfs::File::create(&path).unwrap();
        writeln!(
            f,
            "{{\"type\":\"assistant\",\"timestamp\":\"2026-04-01T10:00:00Z\",\"message\":{{\"model\":\"claude-opus-5\"}}}}",
        )
        .unwrap();
        drop(f);

        let turns = parse_jsonl(&path).unwrap().turns;
        assert!(turns[0].effort.is_none());
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

        let turns = parse_jsonl(&path).unwrap().turns;
        assert_eq!(turns.len(), 1);
        assert!(matches!(&turns[0].role, Role::Other(s) if s == "some-future-type"));
    }

    /// Write `lines` to `<tmp>/session.jsonl`, one per line, and parse
    /// the result.
    fn parse_lines(lines: &[&str]) -> ParsedTranscript {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("session.jsonl");
        let mut f = stdfs::File::create(&path).unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        drop(f);
        parse_jsonl(&path).unwrap()
    }

    fn block_types(turn: &Turn) -> Vec<&str> {
        turn.content
            .as_ref()
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect()
    }

    #[test]
    fn parse_jsonl_merges_a_split_message() {
        let parsed = parse_lines(&[
            r#"{"type":"assistant","uuid":"a-1","timestamp":"2026-04-01T10:00:00Z","cwd":"/w","effort":"high","requestId":"req_1","message":{"id":"msg_1","model":"m-first","content":[{"type":"thinking","thinking":"hmm"}],"usage":{"input_tokens":100,"output_tokens":1}}}"#,
            r#"{"type":"assistant","uuid":"a-2","timestamp":"2026-04-01T10:00:01Z","requestId":"req_1","message":{"id":"msg_1","model":"m-later","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":100,"output_tokens":5}}}"#,
            r#"{"type":"user","uuid":"u-1","timestamp":"2026-04-01T10:00:02Z","message":{"content":[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]}}"#,
            r#"{"type":"assistant","uuid":"a-3","timestamp":"2026-04-01T10:00:03Z","requestId":"req_1","message":{"id":"msg_1","model":"m-later","content":[{"type":"tool_use","id":"tu_1","name":"Read","input":{}}],"usage":{"input_tokens":100,"output_tokens":40}}}"#,
        ]);
        let turns = parsed.turns;
        assert_eq!(turns.len(), 2);

        let merged = &turns[0];
        assert!(matches!(merged.role, Role::Assistant));
        assert_eq!(block_types(merged), vec!["thinking", "text", "tool_use"]);
        let usage = merged.usage.unwrap();
        assert_eq!(usage.output, 40);
        assert_eq!(usage.input, 100);
        assert_eq!(
            merged.timestamp,
            "2026-04-01T10:00:00Z".parse::<DateTime<Utc>>().ok(),
        );
        assert_eq!(merged.uuid.as_deref(), Some("a-1"));
        assert_eq!(merged.model.as_deref(), Some("m-first"));
        assert_eq!(merged.effort.as_deref(), Some("high"));
        assert_eq!(merged.cwd, Some(PathBuf::from("/w")));

        assert!(matches!(turns[1].role, Role::User));
        assert_eq!(turns[1].uuid.as_deref(), Some("u-1"));
    }

    #[test]
    fn parse_jsonl_merge_wraps_string_content() {
        let turns = parse_lines(&[
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":"plain"}}"#,
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"tool_use","id":"tu_1","name":"Read","input":{}}]}}"#,
        ])
        .turns;
        assert_eq!(turns.len(), 1);
        assert_eq!(
            turns[0].content,
            Some(json!([
                {"type": "text", "text": "plain"},
                {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}},
            ])),
        );
    }

    #[test]
    fn parse_jsonl_merge_skips_null_and_absent_content() {
        let turns = parse_lines(&[
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":null}}"#,
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"text","text":"a"}]}}"#,
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1"}}"#,
        ])
        .turns;
        assert_eq!(turns.len(), 1);
        assert_eq!(
            turns[0].content,
            Some(json!([{"type": "text", "text": "a"}]))
        );
    }

    #[test]
    fn parse_jsonl_merge_of_contentless_lines_has_no_content() {
        let turns = parse_lines(&[
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1"}}"#,
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":null}}"#,
        ])
        .turns;
        assert_eq!(turns.len(), 1);
        assert!(turns[0].content.is_none());
    }

    #[test]
    fn parse_jsonl_merge_appends_a_non_array_object_as_one_block() {
        let turns = parse_lines(&[
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"text","text":"a"}]}}"#,
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":{"type":"text","text":"b"}}}"#,
        ])
        .turns;
        assert_eq!(
            turns[0].content,
            Some(json!([
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ])),
        );
    }

    #[test]
    fn parse_jsonl_merge_adds_no_blocks_for_a_repeated_line() {
        // The same line written twice carries the same `uuid`: its
        // blocks count once, while a distinct line still appends.
        let line_a = r#"{"type":"assistant","uuid":"a-1","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"tool_use","id":"tu_1","name":"Read","input":{}}],"usage":{"input_tokens":100,"output_tokens":5}}}"#;
        let turns = parse_lines(&[
            line_a,
            line_a,
            r#"{"type":"assistant","uuid":"a-2","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":100,"output_tokens":9}}}"#,
        ])
        .turns;
        assert_eq!(turns.len(), 1);
        assert_eq!(block_types(&turns[0]), vec!["tool_use", "text"]);
        assert_eq!(turns[0].usage.unwrap().output, 9);
    }

    #[test]
    fn parse_jsonl_merge_ignores_usage_on_a_repeated_line() {
        // A repeat written after a later line must not roll the
        // message's usage back to the repeat's older value.
        let line_a = r#"{"type":"assistant","uuid":"a-1","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"tool_use","id":"tu_1","name":"Read","input":{}}],"usage":{"input_tokens":100,"output_tokens":5}}}"#;
        let turns = parse_lines(&[
            line_a,
            r#"{"type":"assistant","uuid":"a-2","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":100,"output_tokens":40}}}"#,
            line_a,
        ])
        .turns;
        assert_eq!(turns.len(), 1);
        assert_eq!(block_types(&turns[0]), vec!["tool_use", "text"]);
        assert_eq!(turns[0].usage.unwrap().output, 40);
    }

    #[test]
    fn parse_jsonl_merge_keeps_usage_when_last_line_has_none() {
        let turns = parse_lines(&[
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"text","text":"a"}],"usage":{"input_tokens":100,"output_tokens":30}}}"#,
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"text","text":"b"}]}}"#,
        ])
        .turns;
        assert_eq!(turns.len(), 1);
        let usage = turns[0].usage.unwrap();
        assert_eq!(usage.input, 100);
        assert_eq!(usage.output, 30);
    }

    #[test]
    fn parse_jsonl_does_not_merge_partial_keys() {
        let turns = parse_lines(&[
            r#"{"type":"assistant","message":{"id":"msg_1","content":[{"type":"text","text":"a"}]}}"#,
            r#"{"type":"assistant","message":{"id":"msg_1","content":[{"type":"text","text":"b"}]}}"#,
        ])
        .turns;
        assert_eq!(turns.len(), 2);
    }

    #[test]
    fn parse_jsonl_keeps_distinct_messages_separate() {
        let turns = parse_lines(&[
            r#"{"type":"assistant","requestId":"req_1","message":{"id":"msg_1","content":[{"type":"text","text":"first"}]}}"#,
            r#"{"type":"assistant","requestId":"req_2","message":{"id":"msg_2","content":[{"type":"text","text":"second"}]}}"#,
        ])
        .turns;
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].message_id.as_deref(), Some("msg_1"));
        assert_eq!(turns[1].message_id.as_deref(), Some("msg_2"));
    }

    #[test]
    fn parse_jsonl_collects_continued_in() {
        let parsed = parse_lines(&[
            r#"{"type":"user","uuid":"u-1","timestamp":"2026-04-01T10:00:00Z","message":{"content":"hello"}}"#,
            r#"{"type":"continued-in","timestamp":"2026-04-01T10:05:00Z","sessionId":"orig-id","continuedInSessionId":"target-id"}"#,
        ]);
        assert_eq!(parsed.continued_in, vec!["target-id".to_string()]);
        assert_eq!(parsed.turns.len(), 2);
        assert!(matches!(&parsed.turns[1].role, Role::Other(s) if s == "continued-in"));
    }

    #[test]
    fn parse_jsonl_reads_line_uuid() {
        let turns = parse_lines(&[
            r#"{"type":"user","uuid":"u-42","timestamp":"2026-04-01T10:00:00Z","message":{"content":"hello"}}"#,
        ])
        .turns;
        assert_eq!(turns[0].uuid.as_deref(), Some("u-42"));
    }

    #[test]
    fn parse_jsonl_errors_on_a_directory() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(parse_jsonl(tmp.path()).is_err());
    }

    #[test]
    fn parse_jsonl_skips_an_invalid_utf8_line() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("session.jsonl");
        let mut f = stdfs::File::create(&path).unwrap();
        let user =
            r#"{"type":"user","timestamp":"2026-04-01T10:00:00Z","message":{"content":"hello"}}"#;
        f.write_all(user.as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
        f.write_all(&[0xff, 0xfe, b'\n']).unwrap();
        f.write_all(user.as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
        drop(f);

        let turns = parse_jsonl(&path).unwrap().turns;
        assert_eq!(turns.len(), 2);
    }
}
