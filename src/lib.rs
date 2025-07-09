//! # svm-hash
//!
//! A Rust library providing Solana-compatible hashing and Merkle tree utilities.
//!
//! ## Features
//!
//! - **Double hashing**: Domain-separated double SHA-256 hashing compatible with Solana.
//! - **Merkle trees**: Complete Merkle tree implementation with proof generation and verification.
//! - **Generic support**: Works with any data type implementing `AsRef<[u8]>`.
//! - **`bytemuck` integration**: Zero-copy serialization support with the `bytemuck` feature.

pub mod merkle;
pub mod sha2;
