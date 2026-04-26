//! Library half of the `cclens` crate. Holds the modules consumed by
//! both the `cclens` binary (via `use cclens::...`) and the integration
//! test suite. Each promoted module documents its own public API in a
//! `//!` doc comment; this file only declares the modules.
//!
//! Pipeline order (alphabetical declaration below; pipeline order
//! documented for orientation):
//!   `domain → parsing → discovery → inventory → aggregation
//!     → attribution → pricing → rendering → filter`.
//! `inventory` walks user-controlled context-file locations
//! (`~/.claude/{CLAUDE.md,rules,skills,agents}` and the plugin cache);
//! `attribution` folds inventory + per-session metadata + pricing into
//! ranked rows for the `inputs` subcommand.

pub mod aggregation;
pub mod attribution;
pub mod discovery;
pub mod domain;
pub mod filter;
pub mod inventory;
pub mod parsing;
pub mod pricing;
pub mod rendering;
