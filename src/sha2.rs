pub use solana_hash::Hash;
pub use solana_sha256_hasher::{hash, hashv};

/// Computes a domain-separated double SHA-256 hash of a message.
///
/// This function performs two rounds of SHA-256 hashing with domain separation
/// prefixes, which is a common pattern in blockchain applications to prevent
/// hash collision attacks across different contexts.
///
/// # Example
/// ```
/// use svm_hash::sha2::double_hash;
/// use svm_hash::sha2::{LEAF_PREFIX, NODE_PREFIX};
///
/// let hash = double_hash(b"message", LEAF_PREFIX, NODE_PREFIX);
/// ```
pub fn double_hash(message: &[u8], first_prefix: &[u8], second_prefix: &[u8]) -> Hash {
    let first = hashv(&[first_prefix, message]);
    hashv(&[second_prefix, first.as_ref()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_hash_deterministic() {
        let hash1 = double_hash(b"test", b"prefix1", b"prefix2");
        let hash2 = double_hash(b"test", b"prefix1", b"prefix2");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_double_hash_different_inputs() {
        let hash1 = double_hash(b"test1", b"prefix", b"prefix");
        let hash2 = double_hash(b"test2", b"prefix", b"prefix");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_double_hash_different_prefixes() {
        let hash1 = double_hash(b"test", b"prefix1", b"prefix2");
        let hash2 = double_hash(b"test", b"prefix2", b"prefix1");
        assert_ne!(hash1, hash2);
    }
}
