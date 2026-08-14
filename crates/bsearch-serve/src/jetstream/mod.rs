//! Jetstream event consumption.
//!
//! The daemon talks to Jetstream through this module only; the transport
//! lives in the submodules.

mod events;
mod v1;

pub use v1::{run, IngestHandler};
