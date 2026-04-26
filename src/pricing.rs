//! Pricing catalog and cost calculation.
//!
//! Public API:
//! - `load_catalog()` — infallible; degrades to an empty catalog on any
//!   failure, emitting one stderr warning per process. Called by `list`
//!   and `show`.
//! - `refresh_catalog()` — explicit refresh; returns `anyhow::Result` so
//!   `cclens pricing refresh` can exit nonzero on failure.
//! - `cache_info()` — infallible stat-like report for
//!   `cclens pricing info`.
//! - `PricingCatalog::cost_for_components` /
//!   `PricingCatalog::cost_for_turn` — generic cost folds.
//! - `PricingCatalog::cost_for_cache_creation_1h` /
//!   `PricingCatalog::cost_for_cache_creation_5m` — per-tier
//!   cache-creation cost helpers used by the `inputs` subcommand.
//!
//! The catalog comes from `LiteLLM`'s
//! `model_prices_and_context_window.json`. Pricing is computed with
//! Claude's 200k-tier rates and includes `cache_read` tokens (which
//! `Usage::billable()` deliberately excludes).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::SystemTime;

use serde::Deserialize;
use thiserror::Error;

use crate::domain::{CacheCreation, Usage};

// ---- raw schema ----

/// Deserialized directly from `LiteLLM`'s
/// `model_prices_and_context_window.json`. Field names mirror the on-disk
/// schema verbatim so serde does the translation in one place.
///
/// All fields are optional because `LiteLLM` entries omit the tier-split
/// variants for models without a 200k price break, and omit cache-related
/// rates entirely for models without prompt caching.
#[allow(clippy::struct_field_names)]
#[derive(Deserialize)]
struct RawPricingEntry {
    #[serde(default)]
    input_cost_per_token: Option<f64>,
    #[serde(default)]
    output_cost_per_token: Option<f64>,
    #[serde(default)]
    cache_creation_input_token_cost: Option<f64>,
    /// 1-hour ephemeral cache creation rate. Anthropic charges roughly
    /// 1.6× the 5m rate for the 1h bucket; cclens applies it to the
    /// `ephemeral_1h_input_tokens` slice on modern transcripts. Older
    /// catalog snapshots that don't carry this field fall back to the
    /// 5m base rate in `normalize`, preserving today's behavior.
    #[serde(default)]
    cache_creation_input_token_cost_above_1hr: Option<f64>,
    /// 1-hour ephemeral cache creation rate for tokens above the 200k
    /// threshold. Upstream `LiteLLM` publishes this field today only
    /// for two Bedrock-routed Sonnet 3.5 entries
    /// (`anthropic.claude-3-5-sonnet-2024{0620,1022}-v1/v2:0`); for
    /// the Claude 4 family the field is currently absent and the
    /// `tier_from_raw` fallback inherits the 1h base rate. Including
    /// the field now keeps cclens correct if any Bedrock-routed
    /// Sonnet 3.5 transcript flows through the lookup chain.
    #[serde(default)]
    cache_creation_input_token_cost_above_1hr_above_200k_tokens: Option<f64>,
    #[serde(default)]
    cache_read_input_token_cost: Option<f64>,
    #[serde(default)]
    input_cost_per_token_above_200k_tokens: Option<f64>,
    #[serde(default)]
    output_cost_per_token_above_200k_tokens: Option<f64>,
    #[serde(default)]
    cache_creation_input_token_cost_above_200k_tokens: Option<f64>,
    #[serde(default)]
    cache_read_input_token_cost_above_200k_tokens: Option<f64>,
}

// ---- domain ----

/// Per-token rate with Claude's 200k-token tier split.
///
/// `first_200k_rate` applies to the first 200,000 tokens of this type;
/// `above_200k_rate` applies to any tokens beyond that. The threshold
/// (200,000) is fixed by `LiteLLM`'s schema (the `_above_200k_tokens`
/// field-name convention) and does not vary per-entry.
///
/// Fallback rule: if `above_200k_rate` is missing from the raw entry it
/// inherits `first_200k_rate`. If the base is also missing it's 0.0 —
/// which renders as `$0.0000` (effectively "free" for that token type
/// on this model).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TieredRate {
    pub(crate) first_200k_rate: f64,
    pub(crate) above_200k_rate: f64,
}

/// Normalized pricing for a single Claude model.
///
/// Cache creation is split into two tier-aware rates: 5m (the default
/// ephemeral lifetime) and 1h (the longer ephemeral lifetime, billed
/// at roughly 1.6× the 5m rate for opus-4-7). Both legs go through
/// `tiered_cost` with the same 200k-token threshold, so the structural
/// shape stays symmetric. The 1h-above-200k field is published by
/// `LiteLLM` today only for two Bedrock-routed Sonnet 3.5 entries; for
/// other models the `tier_from_raw` fallback inherits the 1h base rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ClaudePricing {
    pub(crate) input: TieredRate,
    pub(crate) output: TieredRate,
    pub(crate) cache_creation_5m: TieredRate,
    pub(crate) cache_creation_1h: TieredRate,
    pub(crate) cache_read: TieredRate,
}

/// Pricing catalog keyed by lower-cased model id.
///
/// Constructed via `from_raw_json` (in production) or `empty` (for
/// degraded startup). Used by `aggregation` to compute per-session totals
/// and by `rendering` to compute per-turn cells in the `show` view.
#[derive(Debug, Default)]
pub struct PricingCatalog {
    entries: HashMap<String, ClaudePricing>,
}

/// Errors raised by Phase 2 internals (fetch, cache I/O, parse). Used
/// only within this module to thread failures through `?` before the
/// public entry point folds them into an empty catalog + single stderr
/// warning.
#[derive(Debug, Error)]
pub enum PricingError {
    #[error("failed to fetch pricing catalog: {0}")]
    Fetch(String),
    #[error("failed to read pricing cache: {0}")]
    CacheRead(#[from] std::io::Error),
    #[error("failed to write pricing cache: {0}")]
    CacheWrite(String),
    #[error("failed to parse pricing catalog: {0}")]
    Parse(#[from] serde_json::Error),
}

// ---- lookup ----

impl PricingCatalog {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Number of Claude entries in the catalog. Used by `pricing info` and
    /// the `pricing refresh` success report.
    pub(crate) fn claude_entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Parse a raw `LiteLLM` JSON payload, filter to Claude entries, and
    /// normalize each to a `ClaudePricing`.
    ///
    /// Filter: retain keys starting with `claude` or `anthropic/`, or
    /// containing `claude-`. This mirrors ccusage's behavior and is
    /// deliberately permissive — any Claude-ish key is kept because the
    /// lookup chain below further narrows by exact / prefixed / longest-
    /// substring matching.
    ///
    /// # Errors
    ///
    /// Returns `PricingError::Parse` if `json` is not a valid `LiteLLM`
    /// catalog payload.
    pub fn from_raw_json(json: &str) -> Result<Self, PricingError> {
        let raw: HashMap<String, RawPricingEntry> = serde_json::from_str(json)?;
        let mut entries = HashMap::new();
        for (key, entry) in raw {
            let key_lower = key.to_lowercase();
            if !is_claude_entry(&key_lower) {
                continue;
            }
            entries.insert(key_lower, normalize(&entry));
        }
        Ok(Self { entries })
    }

    /// Look up pricing for `model` using a staged fallback chain. Each
    /// stage is deterministic — no stage relies on `HashMap` iteration
    /// order, because Rust's `HashMap` iterates in nondeterministic order
    /// and two runs on identical input could otherwise return different
    /// pricing.
    ///
    /// 1. Exact match on the lower-cased input.
    /// 2. Prefix-augmented variants: `claude-`, `anthropic/`,
    ///    `claude-3-5-`, `claude-3-`, `openrouter/openai/` — each
    ///    prepended one at a time to the lower-cased input, in the order
    ///    listed; first hit wins.
    /// 3. One-directional longest-substring fallback: iterate catalog
    ///    keys in lexicographic order and return the **longest** key `k`
    ///    for which `lower_cased_input.contains(k)` holds. Tie-break on
    ///    equal length by lexicographic order (first in the sorted list).
    /// 4. `None`.
    #[must_use]
    pub(crate) fn lookup(&self, model: &str) -> Option<&ClaudePricing> {
        let q = model.to_lowercase();

        if let Some(hit) = self.entries.get(&q) {
            return Some(hit);
        }

        for prefix in LOOKUP_PREFIXES {
            let candidate = format!("{prefix}{q}");
            if let Some(hit) = self.entries.get(&candidate) {
                return Some(hit);
            }
        }

        self.longest_substring_match(&q)
    }

    fn longest_substring_match(&self, query_lower: &str) -> Option<&ClaudePricing> {
        let mut keys: Vec<&String> = self.entries.keys().collect();
        keys.sort();
        let mut best: Option<&String> = None;
        for key in keys {
            if !query_lower.contains(key.as_str()) {
                continue;
            }
            let replace = match best {
                None => true,
                Some(current) => key.chars().count() > current.chars().count(),
            };
            if replace {
                best = Some(key);
            }
        }
        best.and_then(|k| self.entries.get(k))
    }
}

// Ordered prefixes tried during lookup when an exact match misses. Each
// prefix is prepended to the lower-cased query; the first variant that
// hits a catalog key wins. The ordering matches ccusage — `claude-` and
// `anthropic/` cover the common "model id dropped the provider / family
// prefix" case; `claude-3-5-` / `claude-3-` cover queries like `sonnet`
// or `haiku` that dropped the version prefix; `openrouter/openai/` covers
// third-party proxies that route Claude models under their own namespace.
const LOOKUP_PREFIXES: &[&str] = &[
    "claude-",
    "anthropic/",
    "claude-3-5-",
    "claude-3-",
    "openrouter/openai/",
];

fn is_claude_entry(key_lower: &str) -> bool {
    key_lower.starts_with("claude")
        || key_lower.starts_with("anthropic/")
        || key_lower.contains("claude-")
}

// `cache_creation_5m` and `cache_creation_1h` differ by one character
// only — the entire point of these names is the parallel structure
// (each binding maps to its own catalog field and pricing leg).
#[allow(clippy::similar_names)]
fn normalize(raw: &RawPricingEntry) -> ClaudePricing {
    let cache_creation_5m = tier_from_raw(
        raw.cache_creation_input_token_cost,
        raw.cache_creation_input_token_cost_above_200k_tokens,
    );
    // Two-level fallback for the 1h rate:
    //   1. Above-200k missing → inherit from 1h base (handled inside
    //      `tier_from_raw`).
    //   2. 1h base missing → fall back to the 5m base rate; this
    //      preserves today's behavior on older catalog snapshots that
    //      predate the 1h rate. Modern catalogs (post-2026 Anthropic
    //      pricing change) carry the 1h fields and they win.
    let cache_creation_1h = tier_from_raw(
        Some(
            raw.cache_creation_input_token_cost_above_1hr
                .unwrap_or(cache_creation_5m.first_200k_rate),
        ),
        raw.cache_creation_input_token_cost_above_1hr_above_200k_tokens,
    );
    ClaudePricing {
        input: tier_from_raw(
            raw.input_cost_per_token,
            raw.input_cost_per_token_above_200k_tokens,
        ),
        output: tier_from_raw(
            raw.output_cost_per_token,
            raw.output_cost_per_token_above_200k_tokens,
        ),
        cache_creation_5m,
        cache_creation_1h,
        cache_read: tier_from_raw(
            raw.cache_read_input_token_cost,
            raw.cache_read_input_token_cost_above_200k_tokens,
        ),
    }
}

/// Build a `TieredRate` from a (base, `above_200k`) pair applying the
/// documented fallback:
/// - missing `above_200k` inherits `base`
/// - missing `base` falls through to 0.0 (which also cascades into the
///   missing `above_200k` case)
fn tier_from_raw(base: Option<f64>, above: Option<f64>) -> TieredRate {
    let first_200k_rate = base.unwrap_or(0.0);
    let above_200k_rate = above.unwrap_or(first_200k_rate);
    TieredRate {
        first_200k_rate,
        above_200k_rate,
    }
}

// ---- cost ----

/// Tier threshold pinned by `LiteLLM`'s `_above_200k_tokens` field-name
/// convention. Cannot vary per-entry.
const TIER_THRESHOLD: u64 = 200_000;

/// Apply the 200k tier split to a single token count.
///
/// The first 200,000 tokens of this type are priced at `first_200k_rate`;
/// any excess is priced at `above_200k_rate`. `saturating_sub` avoids
/// underflow when `tokens < TIER_THRESHOLD`.
#[allow(clippy::cast_precision_loss)]
fn tiered_cost(tokens: u64, rate: &TieredRate) -> f64 {
    let first = tokens.min(TIER_THRESHOLD) as f64;
    let excess = tokens.saturating_sub(TIER_THRESHOLD) as f64;
    first * rate.first_200k_rate + excess * rate.above_200k_rate
}

impl PricingCatalog {
    /// Compute the total cost for a set of token components.
    ///
    /// Diverges from `Usage::billable()` by including `cache_read` —
    /// cache reads are billed (at a heavily discounted rate) and cost
    /// should reflect that.
    ///
    /// Returns `Some(0.0)` without any lookup if all four counts are
    /// zero (covers synthetic / zero-usage turns on any model, including
    /// `None`). Otherwise returns `Some(sum)` on model hit or `None` on
    /// miss — strict `None` propagation is the contract that Phase 3's
    /// session and running-total folds rely on.
    #[must_use]
    pub fn cost_for_components(
        &self,
        input: u64,
        output: u64,
        cache_creation: CacheCreation,
        cache_read: u64,
        model: Option<&str>,
    ) -> Option<f64> {
        if input == 0 && output == 0 && cache_creation.total() == 0 && cache_read == 0 {
            return Some(0.0);
        }
        let model = model?;
        let pricing = self.lookup(model)?;
        Some(
            tiered_cost(input, &pricing.input)
                + tiered_cost(output, &pricing.output)
                + tiered_cost(cache_creation.ephemeral_5m, &pricing.cache_creation_5m)
                + tiered_cost(cache_creation.ephemeral_1h, &pricing.cache_creation_1h)
                + tiered_cost(cache_read, &pricing.cache_read),
        )
    }

    /// Thin wrapper over `cost_for_components` for callers that already
    /// hold a `Usage`.
    #[must_use]
    pub fn cost_for_turn(&self, usage: &Usage, model: Option<&str>) -> Option<f64> {
        self.cost_for_components(
            usage.input,
            usage.output,
            usage.cache_creation,
            usage.cache_read,
            model,
        )
    }

    /// Cost of `tokens` priced at the 1-hour cache-creation tier for `model`.
    /// Returns `Some(0.0)` when `tokens == 0`; `None` when `model` is
    /// `Some(unknown)` and `tokens > 0`. Used by the `inputs` subcommand
    /// to attribute system-prompt-region content (`CLAUDE.md`, rules, agents).
    #[must_use]
    pub fn cost_for_cache_creation_1h(&self, tokens: u64, model: Option<&str>) -> Option<f64> {
        self.cost_for_components(
            0,
            0,
            CacheCreation {
                ephemeral_1h: tokens,
                ephemeral_5m: 0,
            },
            0,
            model,
        )
    }

    /// Cost of `tokens` priced at the 5-minute cache-creation tier for
    /// `model`. Same null/zero contracts as the 1h variant. Used by the
    /// `inputs` subcommand to attribute on-demand content (skills,
    /// commands).
    #[must_use]
    pub fn cost_for_cache_creation_5m(&self, tokens: u64, model: Option<&str>) -> Option<f64> {
        self.cost_for_components(
            0,
            0,
            CacheCreation {
                ephemeral_5m: tokens,
                ephemeral_1h: 0,
            },
            0,
            model,
        )
    }
}

// ---- fetch ----

/// `LiteLLM`'s canonical pricing catalog URL. Used unless the caller
/// overrides via `CCLENS_PRICING_URL` (both for integration tests and
/// as a power-user escape hatch if GitHub is unreachable).
const LITELLM_CATALOG_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

/// Env var that overrides `LITELLM_CATALOG_URL`. Accepts `http(s)://`,
/// `file://`, or a plain filesystem path.
const CCLENS_PRICING_URL_ENV: &str = "CCLENS_PRICING_URL";

/// Env var that overrides the XDG cache directory (useful for hermetic
/// integration tests and for users who want the cache somewhere other
/// than `dirs::cache_dir()/cclens/`).
const CCLENS_CACHE_DIR_ENV: &str = "CCLENS_CACHE_DIR";

/// Cache filename within whichever directory `resolve_cache_file` picks.
const CACHE_FILENAME: &str = "litellm-pricing.json";

/// HTTP timeouts for the catalog fetch. `load_catalog` is on the hot
/// path for `cclens list` — an unbounded fetch would let a slow or
/// unreachable mirror hang an interactive CLI indefinitely. 5s to
/// establish the connection is generous for a single-hop GitHub CDN;
/// 30s total read covers the ~1.3 MB payload on slow connections.
const FETCH_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const FETCH_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn resolve_catalog_url() -> String {
    match std::env::var(CCLENS_PRICING_URL_ENV) {
        Ok(v) if !v.is_empty() => v,
        _ => LITELLM_CATALOG_URL.to_owned(),
    }
}

/// Fetch the catalog body from `url`.
///
/// Dispatches on scheme so tests can point at a `file://` fixture
/// without a trait abstraction:
/// - `file://` — strip the two-slash prefix and read from disk.
///   Only `file://<absolute-path>` is supported; `file://host/path`
///   is **not** (the "host" component would be silently misread as
///   the start of the path).
/// - `http(s)://` — synchronous GET via `ureq` with hard timeouts
///   (see `FETCH_CONNECT_TIMEOUT` / `FETCH_READ_TIMEOUT`).
/// - anything else — treat as a filesystem path.
fn fetch_catalog_body(url: &str) -> Result<String, PricingError> {
    if let Some(path) = url.strip_prefix("file://") {
        return fs::read_to_string(path).map_err(|e| PricingError::Fetch(e.to_string()));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(FETCH_CONNECT_TIMEOUT)
            .timeout_read(FETCH_READ_TIMEOUT)
            .build();
        return agent
            .get(url)
            .call()
            .map_err(|e| PricingError::Fetch(e.to_string()))?
            .into_string()
            .map_err(|e| PricingError::Fetch(e.to_string()));
    }
    fs::read_to_string(url).map_err(|e| PricingError::Fetch(e.to_string()))
}

// ---- cache ----

/// Resolve the cache file path. `None` means no cache is available for
/// this process (neither `CCLENS_CACHE_DIR` nor `dirs::cache_dir()`
/// returned a usable path — extremely rare, and degrades to "fetch on
/// every run").
///
/// An empty `CCLENS_CACHE_DIR=""` falls through to `dirs::cache_dir()`
/// rather than being interpreted as "use CWD" — a user setting the env
/// var to empty almost certainly means "unset" rather than "use the
/// current directory" (which would silently write pricing caches into
/// arbitrary project dirs).
fn resolve_cache_file() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var(CCLENS_CACHE_DIR_ENV)
        && !custom.is_empty()
    {
        return Some(PathBuf::from(custom).join(CACHE_FILENAME));
    }
    dirs::cache_dir().map(|p| p.join("cclens").join(CACHE_FILENAME))
}

fn read_cache(path: &Path) -> Result<String, PricingError> {
    Ok(fs::read_to_string(path)?)
}

/// Atomically persist `body` at `path` via a same-directory tempfile
/// plus rename. This protects against a Ctrl-C during `pricing refresh`
/// leaving a partially-written, unparseable cache file on disk.
fn write_cache_atomic(path: &Path, body: &str) -> Result<(), PricingError> {
    let parent = path.parent().ok_or_else(|| {
        PricingError::CacheWrite(format!("cache path has no parent: {}", path.display()))
    })?;
    fs::create_dir_all(parent).map_err(|e| PricingError::CacheWrite(e.to_string()))?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .map_err(|e| PricingError::CacheWrite(e.to_string()))?;
    tmp.as_file_mut()
        .write_all(body.as_bytes())
        .map_err(|e| PricingError::CacheWrite(e.to_string()))?;
    tmp.persist(path)
        .map_err(|e| PricingError::CacheWrite(e.to_string()))?;
    Ok(())
}

// ---- load ----

/// Gate for the first-run warning so we print it at most once per
/// process — even if both `run_list` and `run_show` were to end up
/// calling `load_catalog` on the same invocation.
static WARN_ONCE: Once = Once::new();

fn warn_once(message: &str) {
    WARN_ONCE.call_once(|| {
        eprintln!("cclens: {message}");
    });
}

/// Load the pricing catalog, fetching on first run and degrading to an
/// empty catalog on any failure.
///
/// Contract: **infallible**. A missing cache triggers a one-time fetch
/// from `resolve_catalog_url()`; a successful fetch is persisted
/// atomically. Failures (no cache path, fetch failure, parse failure)
/// fold into an empty catalog and emit a single stderr warning per
/// process. Callers need not branch on success/failure — lookups against
/// an empty catalog return `None` and rendering naturally prints `—`.
#[must_use]
pub fn load_catalog() -> PricingCatalog {
    let Some(path) = resolve_cache_file() else {
        warn_once("no cache directory available; cost columns will show —");
        return PricingCatalog::empty();
    };

    if path.exists() {
        match read_cache(&path).and_then(|body| PricingCatalog::from_raw_json(&body)) {
            Ok(catalog) => return catalog,
            Err(e) => {
                warn_once(&format!(
                    "failed to load pricing cache at {}: {e}; cost columns will show —",
                    path.display(),
                ));
                return PricingCatalog::empty();
            }
        }
    }

    let url = resolve_catalog_url();
    let body = match fetch_catalog_body(&url) {
        Ok(body) => body,
        Err(e) => {
            warn_once(&format!(
                "failed to fetch pricing catalog from {url}: {e}; \
                 cost columns will show —",
            ));
            return PricingCatalog::empty();
        }
    };
    // Parse before writing. A bad remote payload (404 HTML, captive
    // portal, gzip artifact) would otherwise clobber any pre-existing
    // cache and leave every subsequent run unable to recover until
    // `pricing refresh` succeeds again.
    let catalog = match PricingCatalog::from_raw_json(&body) {
        Ok(catalog) => catalog,
        Err(e) => {
            warn_once(&format!(
                "fetched pricing catalog but failed to parse: {e}; \
                 cost columns will show —",
            ));
            return PricingCatalog::empty();
        }
    };
    if let Err(e) = write_cache_atomic(&path, &body) {
        warn_once(&format!(
            "fetched pricing catalog but failed to cache at {}: {e}",
            path.display(),
        ));
    }
    catalog
}

/// Result of an explicit `cclens pricing refresh`.
pub struct RefreshReport {
    pub path: PathBuf,
    pub previous_size: u64,
    pub new_size: u64,
    pub entry_count: usize,
}

/// Explicit refresh. Unlike `load_catalog`, this is fallible so the CLI
/// can exit nonzero on a failed user-requested refresh.
///
/// The sequence is fetch → parse → write: on a bad fetch or an
/// unparseable response, the pre-existing cache is left intact so a
/// transient remote hiccup never destroys a known-good catalog.
///
/// # Errors
///
/// Returns an error if no cache directory is available, the catalog
/// fetch fails (network, HTTP status), the response body fails to
/// parse as a `LiteLLM` catalog, or the cache file write fails.
pub fn refresh_catalog() -> anyhow::Result<RefreshReport> {
    let path = resolve_cache_file()
        .ok_or_else(|| anyhow::anyhow!("no cache directory available for pricing catalog"))?;
    let previous_size = fs::metadata(&path).map_or(0, |m| m.len());
    let url = resolve_catalog_url();
    let body =
        fetch_catalog_body(&url).map_err(|e| anyhow::anyhow!("fetch failed from {url}: {e}"))?;
    let catalog =
        PricingCatalog::from_raw_json(&body).map_err(|e| anyhow::anyhow!("parse failed: {e}"))?;
    write_cache_atomic(&path, &body).map_err(|e| anyhow::anyhow!("cache write failed: {e}"))?;
    let new_size = fs::metadata(&path).map_or(0, |m| m.len());
    Ok(RefreshReport {
        path,
        previous_size,
        new_size,
        entry_count: catalog.claude_entry_count(),
    })
}

/// Diagnostic report for `cclens pricing info`.
///
/// Infallible by design: `pricing info` is *the* tool the user runs
/// when something is wrong with the cache, so it must still report
/// path / size / mtime even when the JSON is corrupt. `entry_count ==
/// None` signals "file exists but couldn't be parsed".
pub struct CacheInfo {
    pub path: Option<PathBuf>,
    pub exists: bool,
    pub last_modified: Option<SystemTime>,
    pub size: u64,
    pub entry_count: Option<usize>,
}

#[must_use]
pub fn cache_info() -> CacheInfo {
    let Some(path) = resolve_cache_file() else {
        return CacheInfo {
            path: None,
            exists: false,
            last_modified: None,
            size: 0,
            entry_count: None,
        };
    };
    let meta = fs::metadata(&path).ok();
    let exists = meta.is_some();
    let size = meta.as_ref().map_or(0, fs::Metadata::len);
    let last_modified = meta.and_then(|m| m.modified().ok());
    let entry_count = if exists {
        read_cache(&path)
            .ok()
            .and_then(|body| PricingCatalog::from_raw_json(&body).ok())
            .map(|c| c.claude_entry_count())
    } else {
        None
    };
    CacheInfo {
        path: Some(path),
        exists,
        last_modified,
        size,
        entry_count,
    }
}

// ---- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tier(first: f64, above: f64) -> TieredRate {
        TieredRate {
            first_200k_rate: first,
            above_200k_rate: above,
        }
    }

    fn sample_pricing() -> ClaudePricing {
        ClaudePricing {
            input: sample_tier(3e-6, 6e-6),
            output: sample_tier(15e-6, 22.5e-6),
            cache_creation_5m: sample_tier(3.75e-6, 7.5e-6),
            // Distinct rates so tests that exercise the 1h leg can
            // tell which side priced their tokens; 1h base differs
            // from 5m base, and 1h above-200k differs from both.
            cache_creation_1h: sample_tier(6e-6, 12e-6),
            cache_read: sample_tier(0.3e-6, 0.6e-6),
        }
    }

    fn catalog_with(keys_and_pricing: &[(&str, ClaudePricing)]) -> PricingCatalog {
        let mut entries = HashMap::new();
        for (k, v) in keys_and_pricing {
            entries.insert((*k).to_string(), *v);
        }
        PricingCatalog { entries }
    }

    #[test]
    fn zero_usage_short_circuits_without_model_lookup() {
        let empty = PricingCatalog::empty();
        assert_eq!(
            empty.cost_for_components(0, 0, CacheCreation::default(), 0, Some("any")),
            Some(0.0),
        );
        assert_eq!(
            empty.cost_for_components(0, 0, CacheCreation::default(), 0, None),
            Some(0.0),
        );
    }

    #[test]
    fn unknown_model_with_nonzero_usage_returns_none() {
        let empty = PricingCatalog::empty();
        assert_eq!(
            empty.cost_for_components(1, 0, CacheCreation::default(), 0, Some("any")),
            None,
        );
        assert_eq!(
            empty.cost_for_components(1, 0, CacheCreation::default(), 0, None),
            None,
        );
    }

    #[test]
    fn exact_match_lookup_succeeds() {
        let c = catalog_with(&[("claude-opus-4-7", sample_pricing())]);
        assert!(c.lookup("claude-opus-4-7").is_some());
        assert!(c.lookup("Claude-Opus-4-7").is_some()); // case-insensitive
    }

    #[test]
    fn prefix_augmented_lookup_succeeds() {
        let c = catalog_with(&[("claude-opus-4-7", sample_pricing())]);
        // `opus-4-7` + `claude-` prefix matches the catalog entry.
        assert!(c.lookup("opus-4-7").is_some());
    }

    #[test]
    fn substring_fallback_is_one_directional_longest_match() {
        // Three Claude entries of increasing specificity. Each uses a
        // distinct `first_200k_rate` so the returned entry is
        // identifiable — otherwise a regression that returns the
        // shortest match (or any match) would silently pass.
        let short = ClaudePricing {
            input: sample_tier(1.0, 1.0),
            output: sample_tier(1.0, 1.0),
            cache_creation_5m: sample_tier(1.0, 1.0),
            cache_creation_1h: sample_tier(1.0, 1.0),
            cache_read: sample_tier(1.0, 1.0),
        };
        let mid = ClaudePricing {
            input: sample_tier(2.0, 2.0),
            output: sample_tier(2.0, 2.0),
            cache_creation_5m: sample_tier(2.0, 2.0),
            cache_creation_1h: sample_tier(2.0, 2.0),
            cache_read: sample_tier(2.0, 2.0),
        };
        let long = ClaudePricing {
            input: sample_tier(3.0, 3.0),
            output: sample_tier(3.0, 3.0),
            cache_creation_5m: sample_tier(3.0, 3.0),
            cache_creation_1h: sample_tier(3.0, 3.0),
            cache_read: sample_tier(3.0, 3.0),
        };
        let c = catalog_with(&[
            ("claude", short),
            ("claude-sonnet", mid),
            ("claude-sonnet-4-6", long),
        ]);
        let query = "some-prefix/claude-sonnet-4-6-extra";
        // Repeat the lookup to confirm deterministic output — a regression
        // that iterated over the HashMap directly would flake here.
        for _ in 0..50 {
            let hit = c.lookup(query).expect("longest substring must hit");
            assert!(
                (hit.input.first_200k_rate - 3.0).abs() < 1e-12,
                "expected longest entry (claude-sonnet-4-6) with rate 3.0; \
                 got {}",
                hit.input.first_200k_rate,
            );
        }
    }

    #[test]
    fn substring_fallback_case_insensitive() {
        let c = catalog_with(&[("claude-sonnet-4-6", sample_pricing())]);
        assert!(c.lookup("PREFIX/CLAUDE-SONNET-4-6/SUFFIX").is_some());
    }

    #[test]
    fn tiered_pricing_split_at_200k() {
        let rate = sample_tier(10.0, 1.0);
        // 200k at 10 + 50k at 1 = 2_000_000 + 50_000 = 2_050_000
        assert!((tiered_cost(250_000, &rate) - 2_050_000.0).abs() < 1e-9);
    }

    #[test]
    fn tiered_pricing_below_200k_uses_base_only() {
        let rate = sample_tier(10.0, 1.0);
        // 100k at 10 = 1_000_000; above_200k rate must not contribute.
        assert!((tiered_cost(100_000, &rate) - 1_000_000.0).abs() < 1e-9);
    }

    #[test]
    fn cost_for_turn_sums_all_four_token_types() {
        let c = catalog_with(&[("claude-opus-4-7", sample_pricing())]);
        let usage = Usage {
            input: 1000,
            output: 100,
            cache_creation: CacheCreation {
                ephemeral_5m: 500,
                ephemeral_1h: 0,
            },
            cache_read: 10_000,
        };
        // Using sample_pricing first-tier rates:
        // input:             1000 * 3e-6    = 0.003
        // output:             100 * 15e-6   = 0.0015
        // cache_creation_5m:  500 * 3.75e-6 = 0.001875
        // cache_read:       10000 * 0.3e-6  = 0.003
        // total                             = 0.009375
        let cost = c
            .cost_for_turn(&usage, Some("claude-opus-4-7"))
            .expect("pricing present");
        assert!((cost - 0.009_375).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn cost_for_components_counts_cache_read_even_when_other_fields_zero() {
        // Regression: cache_read-only usage is non-zero; a session with
        // only cache reads must not appear free.
        let c = catalog_with(&[("claude-opus-4-7", sample_pricing())]);
        let cost = c
            .cost_for_components(
                0,
                0,
                CacheCreation::default(),
                1000,
                Some("claude-opus-4-7"),
            )
            .expect("pricing present");
        // 1000 * 0.3e-6 = 0.0003
        assert!((cost - 0.0003).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn cost_for_components_splits_5m_and_1h() {
        // Pricing with distinct 5m and 1h rates:
        // 5m base: 3.75e-6, 1h: 6e-6 (from sample_pricing).
        let c = catalog_with(&[("claude-opus-4-7", sample_pricing())]);
        let only_5m = c
            .cost_for_components(
                0,
                0,
                CacheCreation {
                    ephemeral_5m: 1000,
                    ephemeral_1h: 0,
                },
                0,
                Some("claude-opus-4-7"),
            )
            .expect("pricing present");
        // 1000 * 3.75e-6 = 0.00375
        assert!((only_5m - 0.003_75).abs() < 1e-12, "got {only_5m}");

        let only_1h = c
            .cost_for_components(
                0,
                0,
                CacheCreation {
                    ephemeral_5m: 0,
                    ephemeral_1h: 1000,
                },
                0,
                Some("claude-opus-4-7"),
            )
            .expect("pricing present");
        // 1000 * 6e-6 = 0.006
        assert!((only_1h - 0.006).abs() < 1e-12, "got {only_1h}");

        let both = c
            .cost_for_components(
                0,
                0,
                CacheCreation {
                    ephemeral_5m: 500,
                    ephemeral_1h: 500,
                },
                0,
                Some("claude-opus-4-7"),
            )
            .expect("pricing present");
        // 500 * 3.75e-6 + 500 * 6e-6 = 0.001875 + 0.003 = 0.004875
        assert!((both - 0.004_875).abs() < 1e-12, "got {both}");
    }

    #[test]
    fn cost_for_cache_creation_1h_zero_short_circuits_unknown_model() {
        let empty = PricingCatalog::empty();
        assert_eq!(empty.cost_for_cache_creation_1h(0, Some("any")), Some(0.0));
        assert_eq!(empty.cost_for_cache_creation_1h(0, None), Some(0.0));
    }

    #[test]
    fn cost_for_cache_creation_5m_zero_short_circuits_unknown_model() {
        let empty = PricingCatalog::empty();
        assert_eq!(empty.cost_for_cache_creation_5m(0, Some("any")), Some(0.0));
        assert_eq!(empty.cost_for_cache_creation_5m(0, None), Some(0.0));
    }

    #[test]
    fn cost_for_cache_creation_1h_unknown_model_returns_none() {
        let empty = PricingCatalog::empty();
        assert_eq!(empty.cost_for_cache_creation_1h(1, Some("any")), None);
        assert_eq!(empty.cost_for_cache_creation_1h(1, None), None);
    }

    #[test]
    fn cost_for_cache_creation_5m_unknown_model_returns_none() {
        let empty = PricingCatalog::empty();
        assert_eq!(empty.cost_for_cache_creation_5m(1, Some("any")), None);
        assert_eq!(empty.cost_for_cache_creation_5m(1, None), None);
    }

    #[test]
    fn cost_for_cache_creation_1h_uses_1h_rate_not_5m() {
        // Regression guard against an ephemeral_1h / ephemeral_5m field
        // swap inside the helper. sample_pricing() sets distinct rates
        // (1h base 6e-6, 5m base 3.75e-6); the 1h helper must price at
        // the 1h rate.
        let c = catalog_with(&[("claude-opus-4-7", sample_pricing())]);
        let cost = c
            .cost_for_cache_creation_1h(1000, Some("claude-opus-4-7"))
            .expect("pricing present");
        // 1000 * 6e-6 = 0.006 (1h rate), not 1000 * 3.75e-6 = 0.00375 (5m rate).
        assert!((cost - 0.006).abs() < 1e-12, "got {cost}");
    }

    #[test]
    fn cost_for_cache_creation_5m_uses_5m_rate_not_1h() {
        // Symmetric regression guard: the 5m helper must price at the
        // 5m rate, not the 1h rate.
        let c = catalog_with(&[("claude-opus-4-7", sample_pricing())]);
        let cost = c
            .cost_for_cache_creation_5m(1000, Some("claude-opus-4-7"))
            .expect("pricing present");
        // 1000 * 3.75e-6 = 0.00375 (5m rate), not 1000 * 6e-6 = 0.006 (1h rate).
        assert!((cost - 0.003_75).abs() < 1e-12, "got {cost}");
    }

    #[test]
    fn missing_above_200k_rate_falls_back_to_first_200k_rate() {
        let raw = RawPricingEntry {
            input_cost_per_token: Some(5.0),
            output_cost_per_token: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            cache_creation_input_token_cost_above_1hr_above_200k_tokens: None,
            cache_read_input_token_cost: None,
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_token_cost_above_200k_tokens: None,
        };
        let pricing = normalize(&raw);
        // Above-200k inherits the base rate.
        assert!((pricing.input.above_200k_rate - 5.0).abs() < 1e-12);
        // 300k at 5.0 = 1_500_000.
        assert!((tiered_cost(300_000, &pricing.input) - 1_500_000.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_falls_back_to_5m_for_missing_1h() {
        // Catalog snapshots predating the 1h pricing rollout omit
        // cache_creation_input_token_cost_above_1hr; the 1h leg must
        // fall back to the 5m base rate so older snapshots price
        // identically to today (both first_200k and above_200k inherit
        // the 5m base via the two-level fallback).
        let raw = RawPricingEntry {
            input_cost_per_token: Some(1e-6),
            output_cost_per_token: Some(2e-6),
            cache_creation_input_token_cost: Some(7e-6),
            cache_creation_input_token_cost_above_1hr: None,
            cache_creation_input_token_cost_above_1hr_above_200k_tokens: None,
            cache_read_input_token_cost: Some(0.5e-6),
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_token_cost_above_200k_tokens: None,
        };
        let pricing = normalize(&raw);
        assert!((pricing.cache_creation_1h.first_200k_rate - 7e-6).abs() < 1e-15);
        assert!((pricing.cache_creation_1h.above_200k_rate - 7e-6).abs() < 1e-15);
    }

    #[test]
    fn normalize_uses_explicit_1h_rate_when_present() {
        // Modern catalog: 1h base present, 1h above-200k absent →
        // above-200k inherits the 1h base.
        let raw = RawPricingEntry {
            input_cost_per_token: Some(1e-6),
            output_cost_per_token: Some(2e-6),
            cache_creation_input_token_cost: Some(7e-6),
            cache_creation_input_token_cost_above_1hr: Some(11e-6),
            cache_creation_input_token_cost_above_1hr_above_200k_tokens: None,
            cache_read_input_token_cost: Some(0.5e-6),
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_token_cost_above_200k_tokens: None,
        };
        let pricing = normalize(&raw);
        assert!((pricing.cache_creation_1h.first_200k_rate - 11e-6).abs() < 1e-15);
        assert!((pricing.cache_creation_1h.above_200k_rate - 11e-6).abs() < 1e-15);
    }

    #[test]
    fn normalize_picks_up_1h_above_200k_when_present() {
        // Bedrock-routed Sonnet 3.5 entries publish the
        // cache_creation_input_token_cost_above_1hr_above_200k_tokens
        // field; cclens must apply it to 1h tokens above the 200k
        // threshold rather than reusing the 1h base rate.
        let raw = RawPricingEntry {
            input_cost_per_token: Some(1e-6),
            output_cost_per_token: Some(2e-6),
            cache_creation_input_token_cost: Some(7e-6),
            cache_creation_input_token_cost_above_1hr: Some(11e-6),
            cache_creation_input_token_cost_above_1hr_above_200k_tokens: Some(13e-6),
            cache_read_input_token_cost: Some(0.5e-6),
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_token_cost_above_200k_tokens: None,
        };
        let pricing = normalize(&raw);
        assert!((pricing.cache_creation_1h.first_200k_rate - 11e-6).abs() < 1e-15);
        assert!((pricing.cache_creation_1h.above_200k_rate - 13e-6).abs() < 1e-15);
    }

    #[test]
    fn missing_base_rate_is_zero() {
        let raw = RawPricingEntry {
            input_cost_per_token: None,
            output_cost_per_token: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            cache_creation_input_token_cost_above_1hr_above_200k_tokens: None,
            cache_read_input_token_cost: None,
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_token_cost_above_200k_tokens: None,
        };
        let pricing = normalize(&raw);
        assert!((tiered_cost(1_000_000, &pricing.input)).abs() < 1e-12);
    }

    #[test]
    fn parses_real_litellm_fixture() {
        let json = r#"{
            "claude-opus-4-7": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075,
                "cache_creation_input_token_cost": 0.00001875,
                "cache_read_input_token_cost": 0.0000015,
                "input_cost_per_token_above_200k_tokens": 0.00003,
                "output_cost_per_token_above_200k_tokens": 0.0001125,
                "cache_creation_input_token_cost_above_200k_tokens": 0.0000375,
                "cache_read_input_token_cost_above_200k_tokens": 0.000003
            },
            "claude-sonnet-4-6": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015
            },
            "claude-haiku-4-5": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000005
            },
            "gpt-4": {
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006
            }
        }"#;
        let catalog = PricingCatalog::from_raw_json(json).unwrap();
        // Claude filter keeps 3, drops gpt-4.
        assert_eq!(catalog.claude_entry_count(), 3);

        // Opus rates round-trip with tier split preserved.
        let opus = catalog.lookup("claude-opus-4-7").unwrap();
        assert!((opus.input.first_200k_rate - 0.000_015).abs() < 1e-12);
        assert!((opus.input.above_200k_rate - 0.00003).abs() < 1e-12);

        // Sonnet has no above-200k entries — fallback to base rate.
        let sonnet = catalog.lookup("claude-sonnet-4-6").unwrap();
        assert!((sonnet.input.above_200k_rate - 0.000_003).abs() < 1e-12);
    }
}
