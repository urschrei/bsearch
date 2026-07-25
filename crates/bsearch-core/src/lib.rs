//! Shared building blocks for the bsearch tools.
//!
//! The search CLI and the ingest daemon both need the same embedding model,
//! the same database schema and the same configuration resolution, so those
//! live here rather than being duplicated per binary.

pub mod config;
pub mod db;
pub mod embed;
pub mod models;
