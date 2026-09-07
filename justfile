# List available recipes
default:
    @just --list

# Build the project
build:
    cargo build

run *args:
    cargo run -- {{ args }}

# Run tests
test:
    cargo test

# Format source files
# The glob stays quoted so prettier expands it recursively; just runs each
# line under `sh -cu`, where an unquoted `**` matches one directory level.
fmt:
    just --fmt
    cargo +nightly fmt
    cargo fix --allow-dirty
    prettier -w "./**/*.md"

# Verify formatting without writing
# `cargo fix --allow-dirty` has no check-only counterpart and is deliberately
# omitted, so passing this does not prove `fmt` would leave the tree unchanged
# — `lint` covers most of what fix would rewrite.
fmt-check:
    just --fmt --check
    cargo +nightly fmt --check
    prettier --check "./**/*.md"

# Run clippy on all targets
lint:
    cargo clippy --all-targets -- -D warnings

# Verify formatting + lint + build + test
check: fmt-check lint build test

# Install binary locally
install:
    cargo install --path .
