//! Jetstream v2 event consumption.
//!
//! The daemon talks to Jetstream through this module only. `runner`
//! orchestrates archive catch-up (`archive`, `jss`, `replay`) and the
//! live tail (`live`) over the shared event model (`events`), feeding
//! everything through one `ingest::IngestHandler`.

mod archive;
mod events;
mod ingest;
mod jss;
mod live;
mod replay;
mod runner;

pub use ingest::IngestHandler;
pub use runner::run;
