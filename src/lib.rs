//! Library half of the `cclens` crate. Holds the modules consumed by
//! both the `cclens` binary (via `use cclens::...`) and the integration
//! test suite. Each promoted module documents its own public API in a
//! `//!` doc comment; this file only declares the modules.

pub mod aggregation;
pub mod attribution;
pub mod discovery;
pub mod domain;
pub mod filter;
pub mod inventory;
pub mod parsing;
pub mod pricing;
pub mod rendering;
