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
- **id** — the session UUID, taken from the JSONL filename in
  `~/.claude/projects/`.

Sessions with zero billable tokens are hidden. Malformed JSONL lines are
silently skipped. One unreadable file or project directory does not abort
the listing.

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
- **cumulative** — running sum of billable tokens through this row. Matches
  the session's `list` tokens value at the final assistant row.
- **content** — user prose (or slash-command reconstruction) on user rows;
  assistant reply preview with an optional `+N tool uses` suffix on assistant
  rows. Truncated to 80 characters with `…`.

## Development

```sh
just check    # fmt + lint + build + test
just install  # cargo install --path .
```
