//! wyrd-core: the dependency-free heart of the wyrd benchmark.
//!
//! Charter section 6 assigns this crate the types, manifests, hashing and
//! sealing, and statistics that every other crate builds on. Phase 0
//! lands sealing and statistics with tests, as the charter's Phase 0 exit
//! bar requires: `cargo test` green, and sealing proven by round-trip.

pub mod dose;
pub mod seal;
pub mod sha256;
pub mod stats;
