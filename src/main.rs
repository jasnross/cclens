mod cli;

use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;

use cclens::agents::{AgentRow, PinningFilter, RepricedDelta, repriced_delta};
use cclens::aggregation::SessionSummary;
use cclens::attribution::{AttributionRow, CoverageStats};
use cclens::inventory::InventoryConfig;
use cclens::loading::{self, DataContext, Query};
use cclens::pricing;
use cclens::rendering::{
    render_agents, render_inputs, render_prices, render_session, render_table,
};
use cclens::tui::{Tab, run_tui, run_tui_with_compare_model};
use clap::{CommandFactory, Parser};
use clap_complete::CompleteEnv;
use cli::{
    AgentsArgs, Cli, Command, InputsArgs, OutputFormat, PinningFilterArgs, PricingAction,
    SessionFilterArgs, ThresholdsFilterArgs, emit_agents_empty_hint, emit_inputs_empty_hint,
    emit_list_empty_hint, emit_show_empty_hint,
};
use serde::Serialize;

/// Resolved output mode. Computed once from `OutputFormat` (CLI surface)
/// and TTY detection; `Tui` is the auto-detected case when no `--format`
/// is given and stdout is a terminal.
#[derive(Clone, Copy)]
enum RenderMode {
    Tui,
    Plain,
    Json,
}

#[derive(Serialize)]
struct ShowOutput<'a> {
    session_id: &'a str,
    exchanges: &'a [cclens::aggregation::PreparedExchange],
}

#[derive(Serialize)]
struct InputsOutput<'a> {
    rows: &'a [AttributionRow],
    coverage: &'a CoverageStats,
}

#[derive(Serialize)]
struct AgentsOutput<'a> {
    rows: &'a [AgentRow],
    /// Present only under `--compare-model`. Carries the skip counts
    /// alongside the delta, so a consumer cannot read a total summed
    /// over a shrunken subset as a complete one.
    repriced: Option<RepricedDelta>,
}

#[derive(Serialize)]
struct PricingListEntry<'a> {
    model: &'a str,
    #[serde(flatten)]
    rates: &'a cclens::pricing::ClaudePricing,
}

fn main() -> anyhow::Result<()> {
    CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();
    let mode = match cli.format {
        Some(OutputFormat::Json) => RenderMode::Json,
        None if std::io::stdout().is_terminal() => RenderMode::Tui,
        // Explicit --format plain, or no flag with non-TTY stdout.
        Some(OutputFormat::Plain) | None => RenderMode::Plain,
    };
    match cli.command.unwrap_or(Command::List {
        scope: SessionFilterArgs::default(),
        thresholds: ThresholdsFilterArgs::default(),
    }) {
        Command::List { scope, thresholds } => {
            run_list(mode, &cli.projects_dir, &scope, thresholds)
        }
        Command::Show {
            session_id,
            thresholds,
        } => run_show(mode, &cli.projects_dir, &session_id, thresholds),
        Command::Pricing { action } => run_pricing(mode, action),
        Command::Inputs {
            scope,
            inputs,
            thresholds,
        } => run_inputs(mode, &cli.projects_dir, &scope, &inputs, thresholds),
        Command::Agents {
            agents,
            pinning,
            scope,
            thresholds,
        } => run_agents(
            mode,
            &cli.projects_dir,
            &agents,
            &pinning,
            &scope,
            thresholds,
        ),
    }
}

/// **Deviation 1 applies here too.** The `DataContext` built below
/// carries real `--min-tokens` / `--min-cost` thresholds, and it is
/// the same context every TUI view — including a drilled-into Show —
/// loads against. So `cclens list --min-tokens N` followed by `Enter`
/// now hides below-threshold exchange rows in the detail view, where
/// it previously always showed the full session regardless of list
/// thresholds. This is deliberate uniform-threshold behavior, not a
/// bug: see the plan's Deviation 1 and `Migration Notes`.
fn run_list(
    mode: RenderMode,
    projects_dir: &Path,
    scope: &SessionFilterArgs,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let catalog = Arc::new(pricing::load_catalog());
    let inventory = Arc::new(InventoryConfig::default());
    let query = Query {
        sessions: scope.session_filter(),
        thresholds: thresholds.thresholds_filter(),
        inputs_session_id: None,
        pinning: PinningFilter::default(),
    };
    let ctx = DataContext {
        projects_dir: projects_dir.to_path_buf(),
        catalog: Arc::clone(&catalog),
        inventory: Arc::clone(&inventory),
        query,
    };
    let sessions = loading::load_sessions(&ctx)?;
    match mode {
        RenderMode::Tui if !sessions.is_empty() => {
            let pricing_data = loading::pricing_data(&catalog);
            let initial_fingerprint = loading::build_fingerprint(projects_dir).unwrap_or_default();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let result = rt.block_on(run_tui(
                ctx,
                sessions,
                initial_fingerprint,
                pricing_data,
                Tab::Sessions,
            ));
            rt.shutdown_timeout(std::time::Duration::from_millis(100));
            result?;
        }
        RenderMode::Json => {
            let summaries: Vec<SessionSummary> =
                sessions.iter().map(SessionSummary::from).collect();
            println!("{}", serde_json::to_string_pretty(&summaries)?);
        }
        RenderMode::Tui | RenderMode::Plain => {
            println!("{}", render_table(&sessions));
            if sessions.is_empty() {
                emit_list_empty_hint(scope, &thresholds);
            }
        }
    }
    Ok(())
}

/// **CLI behavior must not change.** The plain/JSON paths below keep
/// today's per-loader-slot threshold scoping (sessions unfiltered by
/// thresholds, inputs rows filtered) by constructing a `Query` per
/// call rather than sharing the TUI's single, uniformly-thresholded
/// one. See the plan's Deviation 1 for why the TUI's `Query.thresholds`
/// is uniform while these paths intentionally are not.
fn run_inputs(
    mode: RenderMode,
    projects_dir: &Path,
    scope: &SessionFilterArgs,
    inputs: &InputsArgs,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let catalog = Arc::new(pricing::load_catalog());
    let inventory = Arc::new(InventoryConfig::default());
    let thresholds_filter = thresholds.thresholds_filter();
    let inputs_query = Query {
        sessions: scope.session_filter(),
        thresholds: thresholds_filter,
        inputs_session_id: inputs.session_id(),
        pinning: PinningFilter::default(),
    };
    let inputs_ctx = DataContext {
        projects_dir: projects_dir.to_path_buf(),
        catalog: Arc::clone(&catalog),
        inventory: Arc::clone(&inventory),
        query: inputs_query,
    };
    match mode {
        RenderMode::Tui => {
            // The Sessions tab is reachable from this entry point (key
            // `1`), and per Deviation 1 the TUI applies `ctx.query`
            // uniformly to every view — so the *initial* Sessions-tab
            // render must be seeded from the same `inputs_ctx` that
            // every subsequent timer/forced refresh runs against.
            // Building it from a separate default-thresholds context
            // would show an unfiltered list that snaps to filtered on
            // the first refresh.
            let sessions = loading::load_sessions(&inputs_ctx)?;
            let pricing_data = loading::pricing_data(&catalog);
            let initial_fingerprint = loading::build_fingerprint(projects_dir).unwrap_or_default();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let result = rt.block_on(run_tui(
                inputs_ctx,
                sessions,
                initial_fingerprint,
                pricing_data,
                Tab::Inputs,
            ));
            rt.shutdown_timeout(std::time::Duration::from_millis(100));
            result?;
        }
        RenderMode::Json => {
            let (visible_rows, coverage) = loading::load_inputs(&inputs_ctx)?;
            let output = InputsOutput {
                rows: &visible_rows,
                coverage: &coverage,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        RenderMode::Plain => {
            let (visible_rows, coverage) = loading::load_inputs(&inputs_ctx)?;
            println!("{}", render_inputs(&visible_rows, &coverage));
            if visible_rows.is_empty() {
                emit_inputs_empty_hint(scope, inputs, &thresholds);
            }
        }
    }
    Ok(())
}

fn run_agents(
    mode: RenderMode,
    projects_dir: &Path,
    agents: &AgentsArgs,
    pinning: &PinningFilterArgs,
    scope: &SessionFilterArgs,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let catalog = Arc::new(pricing::load_catalog());
    let inventory = Arc::new(InventoryConfig::default());

    // Resolved before the walk, so a typo fails immediately rather
    // than after a full scan of every transcript. Exact-key only:
    // `PricingCatalog::lookup`'s fallbacks would resolve a near miss
    // to some unrelated entry and reprice against a model the user
    // never named.
    let compare_model = agents.compare_model();
    if let Some(target) = compare_model
        && catalog.lookup_exact(target).is_none()
    {
        anyhow::bail!(
            "unknown model {target:?} — --compare-model takes an exact pricing-catalog key; \
             run `cclens pricing list` to see them"
        );
    }

    let pinning_filter = pinning.pinning_filter();
    let ctx = DataContext {
        projects_dir: projects_dir.to_path_buf(),
        catalog: Arc::clone(&catalog),
        inventory: Arc::clone(&inventory),
        query: Query {
            sessions: scope.session_filter(),
            thresholds: thresholds.thresholds_filter(),
            inputs_session_id: None,
            pinning: pinning_filter.clone(),
        },
    };
    match mode {
        RenderMode::Json => {
            let rows = loading::load_agents(&ctx)?;
            let output = AgentsOutput {
                rows: &rows,
                repriced: compare_model.map(|t| repriced_delta(&rows, t, &catalog)),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        RenderMode::Tui => {
            // The Sessions tab is reachable from this entry point (key
            // `1`), and the TUI applies `ctx.query` uniformly to every
            // view — so the initial Sessions-tab render must be seeded
            // from the same context every subsequent refresh runs
            // against, exactly as `run_inputs` does.
            // No `load_agents` here: the TUI dispatches its own
            // `AgentsRefresh` at startup, so a scan run now would be
            // discarded — and its failure would abort startup rather
            // than surface as `AgentsData::Error`.
            let sessions = loading::load_sessions(&ctx)?;
            let pricing_data = loading::pricing_data(&catalog);
            let initial_fingerprint = loading::build_fingerprint(projects_dir).unwrap_or_default();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let result = rt.block_on(run_tui_with_compare_model(
                ctx,
                sessions,
                initial_fingerprint,
                pricing_data,
                Tab::Agents,
                compare_model.map(str::to_string),
            ));
            rt.shutdown_timeout(std::time::Duration::from_millis(100));
            result?;
        }
        RenderMode::Plain => {
            let rows = loading::load_agents(&ctx)?;
            let compare = compare_model.map(|t| (t, catalog.as_ref()));
            println!("{}", render_agents(&rows, &pinning_filter, compare));
            if rows.is_empty() {
                emit_agents_empty_hint(scope, pinning, &thresholds);
            }
        }
    }
    Ok(())
}

fn run_pricing(mode: RenderMode, action: PricingAction) -> anyhow::Result<()> {
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
        PricingAction::List { all } => {
            let catalog = pricing::load_catalog();
            let entries = catalog.sorted_entries(all);
            match mode {
                RenderMode::Json => {
                    let json_entries: Vec<PricingListEntry<'_>> = entries
                        .iter()
                        .map(|(model, rates)| PricingListEntry { model, rates })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&json_entries)?);
                }
                RenderMode::Tui | RenderMode::Plain => {
                    if entries.is_empty() {
                        eprintln!("note: pricing catalog is empty — try `cclens pricing refresh`");
                    } else {
                        println!("{}", render_prices(&entries));
                    }
                }
            }
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

/// Render one session's per-exchange table.
///
/// Delegates to `loading::load_show`, which walks the parent JSONL
/// (with the same per-project cross-file dedup pass `run_list` runs)
/// plus every subagent transcript discovered under
/// `<stem>/subagents/`, tags subagent turns with `TurnOrigin::Subagent`,
/// and sorts by user-turn timestamp. The renderer dispatches on origin
/// to render subagent exchanges as single rows (role `subagent`)
/// inline with the parent's exchanges. The body's `cumulative` column
/// at the bottom equals what `cclens list` reports for the same
/// session — list/show consistency by construction.
fn run_show(
    mode: RenderMode,
    projects_dir: &Path,
    session_id: &str,
    thresholds: ThresholdsFilterArgs,
) -> anyhow::Result<()> {
    let session_id = session_id.trim();
    if session_id.is_empty() {
        anyhow::bail!("session id must not be empty");
    }
    let ctx = DataContext {
        projects_dir: projects_dir.to_path_buf(),
        catalog: Arc::new(pricing::load_catalog()),
        inventory: Arc::new(InventoryConfig::default()),
        query: Query {
            thresholds: thresholds.thresholds_filter(),
            ..Query::default()
        },
    };
    let prepared = loading::load_show(&ctx, session_id)?;
    match mode {
        RenderMode::Json => {
            let output = ShowOutput {
                session_id,
                exchanges: &prepared,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        RenderMode::Tui | RenderMode::Plain => {
            let (rendered, _rows_shown) = render_session(&prepared);
            println!("{rendered}");
            if prepared.is_empty() {
                emit_show_empty_hint(&thresholds);
            }
        }
    }
    Ok(())
}
