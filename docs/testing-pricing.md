# Testing Pricing Behavior

How to exercise cclens against a doctored pricing catalog — including a catalog missing a model the transcripts use — without touching the real cache or the network.

## The two levers

`resolve_cache_file` honors `CCLENS_CACHE_DIR`, and `resolve_catalog_url` honors `CCLENS_PRICING_URL`. The value can be an `http(s)://` URL, a `file://<absolute-path>`, or a plain filesystem path — `fetch_catalog_body` dispatches on the scheme, which is what lets a fixture stand in for the network without a trait abstraction. Together they make catalog behavior fully controllable and offline. The integration harness uses both for every invocation (`tests/common/mod.rs`), which is what keeps the suite from reading or writing a developer's real cache.

What makes a "missing model" reproducible is that `load_catalog` uses an existing cache file **verbatim** — no staleness check, no auto-refetch — so a doctored cache survives launch. `refresh_catalog` (what `r` calls in the TUI's pricing overlay, and what `cclens pricing refresh` runs) fetches, parses, and writes, so pointing `CCLENS_PRICING_URL` at a full local copy makes the swap instant and network-free.

## Recipe: a model with no price

1. Copy the real cache aside twice — one copy stays untouched, one gets doctored.
2. Delete the target model's key from the doctored copy, and point `CCLENS_CACHE_DIR` at it.
3. Point `CCLENS_PRICING_URL` at the untouched full copy.

Cost cells for that model render `—` until `r` is pressed in the pricing overlay, then repopulate. Re-doctor before each run: a refresh overwrites the doctored cache, which is the behavior under test.

## Choosing which key to delete

Choosing the key matters twice over.

Pick a model the transcripts actually use — grep `"model":"..."` under the projects directory and count — and prefer one that leaves other models priced. Partial blanking is a sharper test than total blanking, because neighbouring cells stay put while the target cells flip.

Then confirm the removal will actually bite. `lookup` falls back through prefixed variants and a longest-substring match over catalog keys, so if any remaining key is a substring of the model string, deleting the exact key changes nothing.
