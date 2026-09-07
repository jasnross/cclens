# cclens

A tiny Rust CLI that lists your Claude Code conversations: when they happened, what project they were in, what they were about, and how many tokens they consumed. It reads `~/.claude/projects/` and renders either an interactive TUI or a plain aligned table on stdout, depending on `--format` (see [Commands](#commands)).

## Install

```sh
cargo install --path .
```

## Commands

All subcommands accept a global `--format` flag:

- `--format plain` — plain aligned tables on stdout
- `--format json` — machine-readable JSON

When `--format` is omitted, cclens launches an interactive TUI if stdout is a terminal, and falls back to `plain` otherwise (piped or redirected output).

### list

Bare invocation covers all sessions, sorted oldest-first. With `--format plain` (or piped output) that renders as a table:

```sh
$ cclens
 datetime          project            title                                                                               tokens  id
 2026-03-24 00:53  dotfiles           /clear                                                                               99062  f47ac10b-58cc-4372-a567-0e02b2c3d479
 2026-03-28 10:03  agentspec          Could you try running the build and help me fix the error?                        15158013  7c3e8f2a-1d9b-4856-8c20-1f6e4a7b9d33
 2026-03-30 05:29  nvim               I'm having an issue with my fold config in neovim which I'm wondering if you co…    662469  b2f41d0e-5a3c-4721-9e8d-ef5b6a9c1d27
 ...
```

Column widths adapt to the data in each column; the example above is illustrative.

`cclens list` is the explicit equivalent of the default.

Point at a non-default location (e.g. a backup or sshfs mount) with `--projects-dir`:

```sh
$ cclens --projects-dir /mnt/backup/claude/projects list
```

#### Columns

- **datetime** — local-timezone start of the session (`YYYY-MM-DD HH:MM`).
- **project** — final path segment of the session's working directory.
- **title** — the user's first substantive prompt, or the slash command they ran, truncated to 80 characters with `…`.
- **tokens** — sum of billable tokens (`input + output + cache_creation`) across every assistant turn, including subagent/sidechain turns.
- **cost** — per-session USD cost (`$X.XXXX`), summed from the same turns, computed via the LiteLLM pricing catalog. Diverges from `tokens` by including `cache_read` tokens (priced at the discounted cache-read rate). Renders `—` if any assistant turn has an unknown model — strict no-partial-sums propagation.
- **id** — the session UUID, taken from the JSONL filename in `~/.claude/projects/`.

Sessions with zero billable tokens **and** zero cost are hidden — sessions that are non-zero-cost only via `cache_read` stay visible. Malformed JSONL lines are silently skipped. One unreadable file or project directory does not abort the listing.

#### Filtering

Two groups of flags shared across subcommands narrow the result set (`inputs` adds one extra flag of its own — see [the inputs section](#inputs)).

**Scope** — apply to `list` and `inputs` (not `show`, which already pins a single session via its `<session-id>` argument):

- `--project <NAME>` — exact match against the short project name shown in the `project` column. Case-sensitive; substring / glob / regex matching is not supported.
- `--since <WHEN>` / `--until <WHEN>` — inclusive bounds on the session's `started_at`. Accepts a full RFC 3339 timestamp with an explicit offset (e.g. `2026-04-15T00:00:00Z`) or a bare `YYYY-MM-DD`, which means the **start of that day in local time** — the same timeline the `datetime` column displays, and the same rule `git log --since` uses.

**Thresholds** — apply to `list`, `show`, and `inputs`:

- `--min-tokens <N>` — show only rows with at least N billable tokens (e.g. `--min-tokens 50000`).
- `--min-cost <USD>` — show only rows costing at least USD (e.g. `--min-cost 0.50`). Must be a finite number at or above zero; negative and non-finite values are rejected.

When multiple flags are passed, all must clear (logical AND) — for example, `cclens list --project beta --min-tokens 100` keeps only sessions in project `beta` whose total billable tokens are also at least 100. Rows whose cost is unknown (renders `—` in the `cost` column — i.e. an unknown-model row) are excluded by any active `--min-cost`. Orphan user exchanges in `show` (whose `tokens` cell renders `—`) are excluded by any `--min-tokens >= 1`.

If the filter drops every row, stdout still prints the table header, stderr prints a one-line hint (`note: no rows matched <flags>`), and the exit code is 0.

### show

Drill into a single session and see a per-exchange token breakdown:

```sh
$ cclens show <session-id>
```

The session ID argument must match a full session UUID exactly (case- sensitive; surrounding whitespace is trimmed). Tool-use round-trips (assistant tool call → user tool result → assistant reply) collapse into a single exchange; orphaned trailing user turns (no assistant response) render with `—` in the tokens column.

Unlike `list`, `show` does not hide zero-billable sessions — it's an inspection tool and will render any valid session ID.

#### Columns

- **datetime** — local-timezone timestamp of the turn (`YYYY-MM-DD HH:MM`).
- **role** — `user` or `assistant`.
- **tokens** — per-row token total. User rows count `input + cache_creation` across the following assistant cluster; assistant rows count `output`. Orphan user rows show `—`.
- **cost** — per-row USD cost (`$X.XXXX`). User rows price `input + cache_creation + cache_read`; assistant rows price `output`. Includes `cache_read` (the `tokens` column doesn't). Orphan user rows show `—`. Any unknown-model row shows `—`.
- **cumulative** — running sum of billable tokens through this row. Matches the session's `list` tokens value at the final assistant row.
- **cum_cost** — running USD cost through this row. Strict propagation: once any row is `—` (unknown model), every subsequent `cum_cost` cell is also `—`.
- **content** — user prose (or slash-command reconstruction) on user rows; assistant reply preview with an optional `+N tool uses` suffix on assistant rows. Truncated to 80 characters with `…`.

The `--min-tokens` / `--min-cost` flags described under [Filtering](#filtering) work on `show` too — the filter unit is a collapsed exchange (so a user row and its assistant row are shown or hidden together). `cumulative` and `cum_cost` continue to fold over **every** exchange so the final visible row's running totals match the session's `list` totals — visible cells may "jump" between rows when the filter drops middle exchanges.

### inputs

Rank user-controlled context files (CLAUDE.md, rules, skills, agents, plugin-shipped bundles, per-project ancestor files) by attributed cache-creation cost. Walks `~/.claude/{CLAUDE.md,rules,skills,agents}`, the plugin cache, and per-session ancestor + project-local context, then attributes each file's tokens to the matching-tier `cache_creation_*` events observed in the JSONL stream.

```sh
$ cclens inputs
$ cclens inputs --project beta
```

The same scope flags described under [Filtering](#filtering) — `--project`, `--since`, `--until` — apply here, plus an `inputs`-only flag:

- `--session <UUID>` — restrict attribution to one session by full session UUID. Exact match.

`--min-tokens` / `--min-cost` apply as row-level filters on the rendered table (the per-tier coverage line below the table reflects every session in scope, not just the rows kept).

#### Columns

- **file** — path to the context file, with the home directory shown as `~`, truncated with `…`.
- **kind** — what the file is: `global` (`~/.claude/CLAUDE.md`), `rule`, `skill`, `agent`, plugin-shipped, or project-local.
- **tier** — the cache-creation tier the file was observed loading at: `1h`, `5m`, `1h+5m` when both (parent and subagent on different tiers), or `—` when the file is in scope but no session loaded it.
- **tokens** — the file's own size in `cl100k_base` tokens.
- **loads** — how many times the file was loaded, summed across both tiers.
- **billed** — estimated tokens actually billed across those loads.
- **attributed_cost** — per-file USD estimate (`$X.XXXX`). Named apart from the `cost` column in `list` and `show` because those are billed totals and this is an estimate. Renders `—` when the cost is unknown.

The empty-result behavior matches `list`: if every row is dropped, stdout still prints the header, stderr prints `note: no rows matched <flags>`, and the exit code is 0.

### agents

Rank subagent dispatches by observed spend. Reads every subagent transcript under `--projects-dir`, deduplicates each one's assistant turns, and prices the survivors at the model that dispatch actually ran on.

```sh
$ cclens agents
$ cclens agents --pinning pinned
$ cclens agents --compare-model claude-sonnet-5
```

Rows are grouped by agent type, resolved model, observed effort, and how the agent's file constrains the model. One agent whose dispatches disagree about any of those appears as several rows — that disagreement is information, not noise to be merged away.

#### Columns

- **agent** — the dispatch's `agentType`, truncated with `…` at 32 characters. A plugin-shipped agent carries its plugin's namespace (`tw:code-reviewer`).
- **model** — the model the dispatches ran on, as a majority vote across each transcript's assistant turns. A run that switched mid-flight reports the model it mostly ran on.
- **effort** — the reasoning effort the transcript recorded, likewise a majority vote. `—` when the transcript records none.
- **declared_effort** — the effort the agent's file declares in its frontmatter, or `—` when it declares none. Never compared against **effort**; both are reported as found.
- **pinning** — how the agent's file constrains the model: `pinned` (frontmatter names a concrete model), `inherit` (the `inherit` sentinel), `unpinned` (a file that names no model), `no-agent-file` (nothing on disk matched — either a built-in agent or a file not installed here, which are indistinguishable), or `fork` (a fork of the parent, which structurally has no model choice).
- **dispatches** — how many dispatches this row covers.
- **tokens** — deduplicated billable tokens summed across those dispatches.
- **cost** — USD across those dispatches. Renders `—` if any contributing dispatch had an unpriceable model; the token figure still reflects all of them.

#### Filtering

The scope flags under [Filtering](#filtering) — `--project`, `--since`, `--until` — apply per dispatch; `--min-tokens` / `--min-cost` apply per accumulated row.

`--pinning <kinds>` takes a comma-separated list of the five pinning values and **defaults to `inherit,unpinned,no-agent-file`** — the agents whose model nobody chose, which is the question the view was built to answer. That default narrows, so the totals line names the active slice on every run, including at the default. Pass all five to see the whole roster.

`--compare-model <id>` adds `repriced` and `delta` columns and a footer reporting what the visible rows would have cost at that model's rates. The model must be an exact pricing-catalog key (see `cclens pricing list`); a near miss is an error rather than a fuzzy match, so a typo can never reprice against a model you did not name.

**The repriced figure is an upper bound.** It prices the exact token bundle these dispatches produced at another model's rates, and the same work on a different model would produce a different — generally smaller — bundle. It is exact arithmetic over a counterfactual it does not model. Fork rows are excluded from it entirely and counted in the footer, a fork having no model choice to make; unpriced rows are excluded and counted too, so a total summed over a shrunken subset never passes as a complete one.

The empty-result behavior matches `list`.

### pricing

`cclens pricing refresh` re-fetches the LiteLLM catalog and overwrites the cache atomically. `cclens pricing info` prints the cache path, size, last-modified time, and Claude-entry count.

`cclens pricing list` shows per-model rates for every Claude entry in the catalog — columns `model`, `tier`, `input`, `output`, `cache_rd`, `cache_5m`, `cache_1h`, in $/MTok. Models whose ≤200k and >200k rates differ render as two rows (`≤200k`, `>200k`); uniform models render as one. `--all` additionally includes provider/region-prefixed keys (`bedrock`, `vertex_ai`, …) alongside the bare `claude-*` ones.

The catalog is fetched on first run (one synchronous HTTPS GET to `raw.githubusercontent.com`). It does not auto-expire; refresh is explicit. If the fetch fails, every cost cell renders `—` and a single stderr warning is printed — `list` and `show` still work.

The cache lives at `dirs::cache_dir()/cclens/litellm-pricing.json` (macOS: `~/Library/Caches/cclens/`; Linux: `~/.cache/cclens/`). Two env vars override defaults:

- `CCLENS_CACHE_DIR` — alternative cache directory
- `CCLENS_PRICING_URL` — alternative catalog URL (accepts `http(s)://`, `file://<absolute-path>`, or a plain filesystem path)

## Development

```sh
just check    # fmt-check + lint + build + test (verifies; never writes)
just fmt      # format the tree in place
just install  # cargo install --path .
```
