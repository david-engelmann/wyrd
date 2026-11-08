//! wyrd-train: the candle training substrate for PRE-001.
//!
//! ADR-001 decided this crate's shape: candle is the trainer, the
//! external GPU trainer stays quarantined in reserve behind this
//! crate's boundary and is not used. The crate holds the pinned
//! subject model (the re-spike harness's architecture, so the
//! measured 296 tokens per second applies), checkpointed training
//! runs with deterministic stream construction and a hashed run
//! manifest, and the anchor-protocol evaluation (generation at
//! temperature 1, perplexity gaps, clean-perplexity gate measure).
//!
//! This is the workspace's one crate with external dependencies:
//! candle is a Rust library, so the charter's language ledger is
//! untouched (it governs hand-written code, and every line here is
//! Rust), and the substrate choice itself is ADR-001's decision,
//! named in charter sections 6 and 10.
//!
//! A boundary note, consciously taken: the charter's crate sketch
//! puts case runners in wyrd-eval, but PRE-001's evaluation reads
//! operate directly on this crate's model type (generation and
//! perplexity under the trained weights), so they live here in
//! evalgen. wyrd-eval keeps the provider-based case runners for API
//! subjects, which is the split the code actually has.

pub mod evalgen;
pub mod model;
pub mod train;
