# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# cclens

A tiny Rust CLI that lists Claude Code conversations: when they happened, what
project they were in, what they were about, and how many tokens they consumed.
It reads `~/.claude/projects/` and prints a plain aligned table to stdout.

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

The crate is currently a single `src/main.rs` file, sectioned with `// ---- x ----`
banner comments. The sections (in order) are:

```
cli           ← clap parser, main entry, run_list
domain        ← Session, Turn, Role, Usage (the internal types)
parsing       ← RawLine / RawMessage / RawUsage + parse_jsonl + raw_to_turn
aggregation   ← turn list → Session (title extraction, short-name derivation,
                zero-billable filter)
discovery     ← projects_dir walk → (project_dir, [jsonl_paths])
rendering     ← comfy-table setup, truncate_title, format_local, render_table
tests         ← inline #[cfg(test)] mod tests
```

When a section grows large enough that the banners hurt more than help,
promote it to its own module file. Prefer the modern Rust module convention
(`src/parsing.rs`) over `mod.rs`. The gityard repo is the reference for how
the split should look.

Integration tests live in `tests/listing.rs` and use `assert_cmd` against
fixtures under `tests/fixtures/projects/`.

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
