//! On-disk JSONL schema deserialization and translation to typed `Turn`s.
//!
//! Public API:
//! - `parse_jsonl(&Path) -> anyhow::Result<Vec<Turn>>` — streaming
//!   per-line parser; silently skips malformed lines.
//!
//! All `Raw*` types and the `into_usage` / `raw_to_turn` adapters are
//! module-private — the public surface is just the entry point. If a
//! consumer wants to hand-build a `Turn` it should construct one from
//! `domain` directly rather than reaching into the deserialization layer.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

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

/// # Errors
///
/// Returns an error if the file at `path` cannot be opened. Per-line
/// I/O failures, JSON-parse failures, and lines that translate to no
/// `Turn` (e.g. missing `type`) are silently skipped — only top-level
/// open failure propagates.
pub fn parse_jsonl(path: &Path) -> anyhow::Result<Vec<Turn>> {
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
        // Parser produces parent-origin turns; `build_subagent_turns`
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
}
