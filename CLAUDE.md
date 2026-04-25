# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# cclens

A tiny Rust CLI that lists Claude Code conversations: when they happened, what
project they were in, what they were about, and how many tokens they consumed.
It reads `~/.claude/projects/` and prints a plain aligned table to stdout.

## Project Status

**cclens is pre-1.0.** The CLI surface (subcommands, flag names, defaults), the
rendered output format, and the library API exposed by `src/lib.rs` are all
expected to evolve. This is the cheapest time to make foundational changes —
once we ship 1.0 and users start depending on stable flags, output, and import
paths, every assumption hardens and refactors get expensive.

When weighing design decisions:

- Refactorings and breaking changes are on the table. Weigh the tradeoffs, but
  don't reflexively defer hard changes to "later" — later is when they're
  harder to make.
- If components are difficult to fit together, abstractions are trending toward
  leaky, or a change is awkward to implement, treat that friction as a signal
  to reshape the surrounding code rather than work around it.
- **"First make the change easy, then make the easy change."** When the next
  feature feels awkward, the right first step is often to refactor so the
  feature drops in cleanly — then add it.

This bias toward refactoring does not override scope discipline. Improve what
you touch in service of the current task; surface larger structural changes as
their own work rather than smuggling them into unrelated commits.

## Commands

```sh
just check                      # fmt + lint + build + test (the full suite)
just fmt                        # cargo fmt
just lint                       # clippy on all targets, warnings-as-errors
just test                       # run tests
just build                      # build only
just install                    # cargo install --path .

# Or without just:
cargo build
cargo test
cargo fmt --check               # verify formatting without writing (what CI runs)
cargo clippy --all-targets -- -D warnings

# Run a single unit test (inline module in src/main.rs):
cargo test extract_title_from_slash_command_with_args

# Run a single integration test (tests/listing.rs):
cargo test --test listing list_renders_sessions_oldest_first_with_correct_totals
```

## Git Commits

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <description>

feat: add json output mode
fix(parser): tolerate missing timestamp field
refactor(render): extract truncate_title helper
test(discover): cover unreadable subdir case
chore: bump chrono to 0.4.41
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `style`.
Scope is optional but helpful for section-level changes.

## Clippy

Five lint groups (`complexity`, `pedantic`, `perf`, `style`, `suspicious`) are
denied at `priority = -1` in `Cargo.toml`. Four `restriction` lints are
additionally opted into: `expect_used`, `panic`, `unwrap_used`,
`wildcard_enum_match_arm`.

Notable lints that affect everyday coding:

- `unwrap_used` — use `?`, `match`, `let else`, or an explicit fallback instead
  of `.unwrap()`
- `expect_used` / `panic` — avoid in non-test code; tests are allowed via
  `clippy.toml` (`allow-expect-in-tests`, `allow-panic-in-tests`,
  `allow-unwrap-in-tests`)
- `wildcard_enum_match_arm` — enumerate `Role` variants explicitly; do not
  match with `_`, so new variants force a compile-time decision at every
  call site
- `uninlined_format_args` — write `format!("{x}")` not `format!("{}", x)`
- `doc_markdown` — wrap identifiers like `RawLine`, `serde`, `comfy-table`
  in backticks in doc comments

Run `cargo fmt && cargo clippy --all-targets -- -D warnings` before committing.

## Source Layout

The crate compiles as two crates from the same `src/` tree: a library
(`src/lib.rs`, root for the seven promoted modules) and a binary
(`src/main.rs`, root for orchestration plus the CLI submodule). Both
are named `cclens`; the binary consumes the library via
`use cclens::...`.

Library modules (declared `pub mod` in `src/lib.rs`, one file each
under `src/<name>.rs`, in pipeline order):

```
domain        ← Session, Turn, Role, Usage, CacheCreation
parsing       ← RawLine / RawMessage / RawUsage + parse_jsonl + raw_to_turn
discovery     ← projects_dir walk → (project_dir, [jsonl_paths])
aggregation   ← turn list → Session; Exchange grouping;
                title extraction; project short-name derivation;
                zero-billable filter
pricing       ← LiteLLM-catalog fetch/cache/lookup, tiered cost math,
                pricing-subcommand handlers
rendering     ← comfy-table render_table (list view) and render_session
                (show view), plus per-row formatters
filter        ← Thresholds value type — the cross-boundary primitive
                that lets rendering accept --min-tokens / --min-cost
                without depending on clap
```

Binary entry point (`src/main.rs`):

- Declares `mod cli;` (binary-only — `src/cli.rs` is **not** in
  `lib.rs`) so library code cannot reach into clap-derived types.
- Imports library modules via `use cclens::...`.
- Holds `main`, `run_list`, `run_show`, `run_pricing`, the per-project
  `dedup_assistant_turns` orchestration helper, and `stem_matches`.

Binary CLI submodule (`src/cli.rs`):

- `Cli` / `Command` / `PricingAction` clap parser types.
- `FilterArgs` — flattened `--min-tokens` / `--min-cost` flags with a
  `.thresholds()` constructor that produces a library `Thresholds`.
- `emit_empty_result_hint`.

Each library module file opens with a `//!` doc comment listing its
public API surface. The gityard repo is the reference for the
split's overall shape (flat `src/<name>.rs` files, `lib.rs` of pure
`pub mod` declarations, `main.rs` with binary-only `mod cli;`).

Integration tests live in `tests/listing.rs` (and sibling files) and
use `assert_cmd` against fixtures under `tests/fixtures/projects/`.
Snapshot regression tests in `tests/snapshots.rs` use `insta` against
a dedicated fixture tree at `tests/fixtures/snapshot-projects/` and
lock byte-for-byte rendering of `list` and `show`.

## Design Principles

### Model the domain with typed structs; never parse strings twice

The on-disk JSONL schema is represented by `RawLine` / `RawMessage` / `RawUsage`
— `#[derive(Deserialize)]` structs whose field names mirror the contract
verbatim (`input_tokens`, `cache_creation_input_tokens`, …). These are
translated into the internal `Turn` / `Usage` types exactly once, in
`raw_to_turn` and `into_usage`. Every downstream caller operates on the
typed domain.

When the on-disk schema changes, only the `Raw*` structs and the `raw_to_turn`
adapter should move. If you find yourself reaching into a `serde_json::Value`
outside `parsing` or `extract_title`, that's a signal to add a typed field to
`RawMessage` instead.

### Let serde do the work

Optional fields are modelled as `Option<T>` with `#[serde(default)]`, and unknown
enum variants fall through to `Role::Other(String)`. Do not preprocess JSON text
before deserialization — if a field shape varies, widen the struct with
`#[serde(untagged)]` or accept `Value` at that one site.

### Separate data from presentation

`discovery` produces paths, `parsing` produces `Turn`s, `aggregation` produces
`Session`s, `rendering` produces a `String`. Each section consumes the previous
section's output and nothing else. This makes a future `--json` or `--format=…`
flag a mechanical addition next to `render_table`, not a rewrite.

Keep formatting concerns (truncation, timezone conversion, alignment,
column order) inside `rendering`. The domain types know nothing about how
they'll be displayed.

### Degrade gracefully, per entry

One unreadable project directory, one unreadable `.jsonl` file, and one
malformed line must not abort the listing. Only a failure to read the
top-level `projects_dir` itself propagates — that's a user-facing config
error (bad `--projects-dir`).

This is enforced by `let Ok(...) = ... else { continue }` at each layer and
by integration fixtures that deliberately include malformed JSONL. When
adding new parsing or I/O paths, match the existing pattern rather than
propagating errors with `?` at the same level.

### Colocate code with its consumer

Place helpers in the section that calls them, not in a generic `util`
bucket. If a function has one caller, it belongs next to that caller. The
banner sections in `main.rs` encode this: `extract_title`,
`extract_slash_command_title`, and the `SYNTHETIC_USER_CONTENT_PREFIXES`
constant all live in `aggregation` because that's the only place they run.

### Count scalars, not bytes

`truncate_title` is scalar-aware (`s.chars().count()`) because session
titles are user-authored text and frequently contain multi-byte characters.
Any new truncation or width logic must operate on `chars()` (or a grapheme
iterator where stricter correctness is needed), never on byte slices.

### Snapshot tests are reviewable assertions, not contracts

`tests/snapshots.rs` locks byte-for-byte rendering of `list` and `show` via
`insta` so that any change to output is visible in review. When output changes
intentionally — a new column, a tweaked label, a different default sort —
accept the new baseline with `cargo insta review` rather than contorting code
to preserve the old bytes. While cclens is pre-1.0 (see Project Status), a
failing snapshot is a prompt to confirm intent, not a regression to avoid.
