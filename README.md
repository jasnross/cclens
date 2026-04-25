# cclens

A tiny Rust CLI that lists your Claude Code conversations: when they happened,
what project they were in, what they were about, and how many tokens they
consumed. It reads `~/.claude/projects/` and prints a plain aligned table to
stdout.

## Install

```sh
cargo install --path .
```

## Commands

### list

Bare invocation lists all sessions sorted oldest-first:

```sh
$ cclens
 datetime          project            title                                                                               tokens  id
 2026-03-24 00:53  dotfiles           /clear                                                                               99062  f47ac10b-58cc-4372-a567-0e02b2c3d479
 2026-03-28 10:03  agentspec          Could you try running the build and help me fix the error?                        15158013  7c3e8f2a-1d9b-4856-8c20-1f6e4a7b9d33
 2026-03-30 05:29  nvim               I'm having an issue with my fold config in neovim which I'm wondering if you co…    662469  b2f41d0e-5a3c-4721-9e8d-ef5b6a9c1d27
 ...
```

Column widths adapt to the data in each column; the example above is
illustrative.

`cclens list` is the explicit equivalent of the default.

Point at a non-default location (e.g. a backup or sshfs mount) with
`--projects-dir`:

```sh
$ cclens --projects-dir /mnt/backup/claude/projects list
```

#### Columns

- **datetime** — local-timezone start of the session (`YYYY-MM-DD HH:MM`).
- **project** — final path segment of the session's working directory.
- **title** — the user's first substantive prompt, or the slash command they
  ran, truncated to 80 characters with `…`.
- **tokens** — sum of billable tokens (`input + output + cache_creation`)
  across every assistant turn, including subagent/sidechain turns.
- **cost** — per-session USD cost (`$X.XXXX`), summed from the same turns,
  computed via the LiteLLM pricing catalog. Diverges from `tokens` by
  including `cache_read` tokens (priced at the discounted cache-read
  rate). Renders `—` if any assistant turn has an unknown model — strict
  no-partial-sums propagation.
- **id** — the session UUID, taken from the JSONL filename in
  `~/.claude/projects/`.

Sessions with zero billable tokens **and** zero cost are hidden — sessions
that are non-zero-cost only via `cache_read` stay visible. Malformed JSONL
lines are silently skipped. One unreadable file or project directory does
not abort the listing.

#### Filtering

Two flags narrow the result set; both apply to `list` and `show`:

- `--min-tokens <N>` — show only rows with at least N billable tokens
  (e.g. `--min-tokens 50000`).
- `--min-cost <USD>` — show only rows costing at least USD (e.g.
  `--min-cost 0.50`).

When both are passed, both must clear (logical AND). Rows whose cost is
unknown (renders `—` in the `cost` column — i.e. an unknown-model row)
are excluded by any active `--min-cost`. Orphan user exchanges in
`show` (whose `tokens` cell renders `—`) are excluded by any
`--min-tokens >= 1`.

If the filter drops every row, stdout still prints the table header,
stderr prints a one-line hint (`note: no rows matched <flags>`), and
the exit code is 0.

### show

Drill into a single session and see a per-exchange token breakdown:

```sh
$ cclens show <session-id>
```

The session ID argument must match a full session UUID exactly (case-
sensitive; surrounding whitespace is trimmed). Tool-use round-trips
(assistant tool call → user tool result → assistant reply) collapse into a
single exchange; orphaned trailing user turns (no assistant response) render
with `—` in the tokens column.

Unlike `list`, `show` does not hide zero-billable sessions — it's an
inspection tool and will render any valid session ID.

#### Columns

- **datetime** — local-timezone timestamp of the turn (`YYYY-MM-DD HH:MM`).
- **role** — `user` or `assistant`.
- **tokens** — per-row token total. User rows count
  `input + cache_creation` across the following assistant cluster; assistant
  rows count `output`. Orphan user rows show `—`.
- **cost** — per-row USD cost (`$X.XXXX`). User rows price `input +
  cache_creation + cache_read`; assistant rows price `output`. Includes
  `cache_read` (the `tokens` column doesn't). Orphan user rows show `—`.
  Any unknown-model row shows `—`.
- **cumulative** — running sum of billable tokens through this row. Matches
  the session's `list` tokens value at the final assistant row.
- **cum_cost** — running USD cost through this row. Strict propagation:
  once any row is `—` (unknown model), every subsequent `cum_cost`
  cell is also `—`.
- **content** — user prose (or slash-command reconstruction) on user rows;
  assistant reply preview with an optional `+N tool uses` suffix on assistant
  rows. Truncated to 80 characters with `…`.

The `--min-tokens` / `--min-cost` flags described under
[Filtering](#filtering) work on `show` too — the filter unit is a
collapsed exchange (so a user row and its assistant row are shown or
hidden together). `cumulative` and `cum_cost` continue to fold over
**every** exchange so the final visible row's running totals match
the session's `list` totals — visible cells may "jump" between rows
when the filter drops middle exchanges.

### pricing

`cclens pricing refresh` re-fetches the LiteLLM catalog and overwrites
the cache atomically. `cclens pricing info` prints the cache path,
size, last-modified time, and Claude-entry count.

The catalog is fetched on first run (one synchronous HTTPS GET to
`raw.githubusercontent.com`). It does not auto-expire; refresh is
explicit. If the fetch fails, every cost cell renders `—` and a single
stderr warning is printed — `list` and `show` still work.

The cache lives at `dirs::cache_dir()/cclens/litellm-pricing.json`
(macOS: `~/Library/Caches/cclens/`; Linux: `~/.cache/cclens/`). Two
env vars override defaults:
- `CCLENS_CACHE_DIR` — alternative cache directory
- `CCLENS_PRICING_URL` — alternative catalog URL (accepts `http(s)://`,
  `file://<absolute-path>`, or a plain filesystem path)

## Development

```sh
just check    # fmt + lint + build + test
just install  # cargo install --path .
```
