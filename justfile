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
fmt:
    just --fmt
    cargo +nightly fmt
    cargo fix --allow-dirty
    prettier -w ./**/*.md

# Run clippy on all targets
lint:
    cargo clippy --all-targets -- -D warnings

# Format + lint + build + test
check: fmt lint build test

# Install binary locally
install:
    cargo install --path .
