use bytemuck::Pod;
use solana_hash::Hash;

use super::{hash_leaf_internal, hash_leaves_internal, root_from_leaf_hashes, MerkleProof};

impl MerkleProof {
    /// Create a proof from POD (Plain Old Data) items when bytemuck feature is
    /// enabled. The `leaf_prefix` is optional and defaults to
    /// `DEFAULT_LEAF_PREFIX`.
    pub fn from_pod_leaves<T: Pod>(
        items: &[T],
        node_index: u32,
        leaf_prefix: Option<&[u8]>,
    ) -> Option<Self> {
        let hashes = hash_leaves_internal(items, bytemuck::bytes_of, leaf_prefix, false);
        let siblings = Self::from_hashed_leaves_internal(hashes, node_index as usize)?;
        Some(Self {
            siblings,
            leaf_index: None,
        })
    }

    /// Create proof from POD items with index binding.
    pub fn from_indexed_pod_leaves<T: Pod>(
        items: &[T],
        node_index: u32,
        leaf_prefix: Option<&[u8]>,
    ) -> Option<Self> {
        let hashes = hash_leaves_internal(items, bytemuck::bytes_of, leaf_prefix, true);
        let siblings = Self::from_hashed_leaves_internal(hashes, node_index as usize)?;
        Some(Self {
            siblings,
            leaf_index: Some(node_index),
        })
    }

    /// Compute the root from a POD leaf when bytemuck feature is enabled. The
    /// `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
    pub fn root_from_pod_leaf<T: Pod>(&self, item: &T, leaf_prefix: Option<&[u8]>) -> Hash {
        let leaf_hash = hash_leaf_internal(item, self.leaf_index, bytemuck::bytes_of, leaf_prefix);
        self.root_from_hashed_leaf(leaf_hash)
    }
}

/// Compute the Merkle root from POD items when bytemuck feature is enabled. The
/// `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
pub fn merkle_root_from_pod_leaves<T: Pod>(
    items: &[T],
    leaf_prefix: Option<&[u8]>,
) -> Option<Hash> {
    let hashes = hash_leaves_internal(items, bytemuck::bytes_of, leaf_prefix, false);
    root_from_leaf_hashes(hashes)
}

/// Compute the Merkle root from POD leaves with index binding. The
/// `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
pub fn merkle_root_from_indexed_pod_leaves<T: Pod>(
    items: &[T],
    leaf_prefix: Option<&[u8]>,
) -> Option<Hash> {
    let hashes = hash_leaves_internal(items, bytemuck::bytes_of, leaf_prefix, true);
    root_from_leaf_hashes(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_leaves() {
        const LEAF_PREFIX: &[u8] = b"test_pod_leaves";

        use bytemuck::{Pod, Zeroable};

        #[derive(Clone, Copy, Default, Pod, Zeroable)]
        #[repr(C)]
        struct TestData {
            id: Hash,
            value: u64,
        }

        let data = [
            TestData::default(),
            TestData {
                id: Hash::new_unique(),
                value: 100,
            },
            TestData {
                id: Hash::new_unique(),
                value: 200,
            },
            TestData {
                id: Hash::new_unique(),
                value: 300,
            },
            TestData {
                id: Hash::new_unique(),
                value: 400,
            },
        ];

        let proof = MerkleProof::from_pod_leaves(&data, 2, Some(LEAF_PREFIX)).unwrap();
        let root = merkle_root_from_pod_leaves(&data, Some(LEAF_PREFIX)).unwrap();

        assert_eq!(proof.root_from_pod_leaf(&data[2], Some(LEAF_PREFIX)), root);
        assert_ne!(proof.root_from_pod_leaf(&data[0], Some(LEAF_PREFIX)), root);
    }
}
