//! Library half of the `cclens` crate. Holds the modules consumed by
//! both the `cclens` binary (via `use cclens::...`) and the integration
//! test suite. Each promoted module documents its own public API in a
//! `//!` doc comment; this file only declares the modules.
//!
//! Pipeline order (alphabetical declaration below; pipeline order
//! documented for orientation):
//!   `domain → parsing → discovery → inventory → aggregation
//!     → attribution → agents → loading → pricing → rendering
//!     → filter`.
//! `agents` folds subagent transcripts into per-dispatch records and
//! groups them into the rows the `agents` subcommand ranks; it sits
//! beside `attribution` rather than under it, sharing input data and
//! no output.
//! `inventory` walks user-controlled context-file locations
//! (`~/.claude/{CLAUDE.md,rules,skills,agents}` and the plugin cache);
//! `attribution` folds inventory + per-session metadata + pricing into
//! ranked rows for the `inputs` subcommand. `loading` composes the
//! prior stages into the view-ready data the TUI (and, via the same
//! functions, the plain/JSON CLI paths) consumes.
//!
//! `formatting` provides shared per-value format helpers.
//! `views` provides shared per-row view builders consumed by both
//! `rendering` and `tui`.

pub mod agents;
pub mod aggregation;
pub mod attribution;
pub mod discovery;
pub mod domain;
pub mod filter;
pub mod formatting;
pub mod inventory;
pub mod loading;
pub mod parsing;
pub mod pricing;
pub mod rendering;
pub mod tui;
pub mod views;
