# List available recipes
default:
    @just --list

# Build the project
build:
    cargo build

# Run tests
test:
    cargo test

# Format source files
fmt:
    cargo +nightly fmt

# Run clippy on all targets
lint:
    cargo clippy --all-targets -- -D warnings

# Format + lint + build + test
check: fmt lint build test

# Install binary locally
install:
    cargo install --path .
