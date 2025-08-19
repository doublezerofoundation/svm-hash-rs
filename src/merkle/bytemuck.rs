use bytemuck::Pod;
use solana_hash::Hash;

use crate::sha2::double_hash;

use super::{hash_leaves, MerkleProof, DEFAULT_LEAF_PREFIX};

impl MerkleProof {
    /// Create a proof from POD (Plain Old Data) items when bytemuck feature is
    /// enabled. The `leaf_prefix` is optional and defaults to
    /// `DEFAULT_LEAF_PREFIX`.
    pub fn from_pod_leaves<T: Pod>(
        items: &[T],
        node_index: usize,
        leaf_prefix: Option<&[u8]>,
    ) -> Option<Self> {
        Self::from_hashed_leaves(
            hash_leaves(items, bytemuck::bytes_of, leaf_prefix),
            node_index,
        )
    }

    /// Compute the root from a POD leaf when bytemuck feature is enabled. The
    /// `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
    pub fn root_from_pod_leaf<T: Pod>(&self, item: &T, leaf_prefix: Option<&[u8]>) -> Hash {
        let leaf = double_hash(
            bytemuck::bytes_of(item),
            leaf_prefix.unwrap_or(DEFAULT_LEAF_PREFIX),
            DEFAULT_LEAF_PREFIX,
        );
        self.root_from_hashed_leaf(leaf)
    }
}

/// Compute the Merkle root from POD items when bytemuck feature is enabled. The
/// `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
pub fn merkle_root_from_pod_leaves<T: Pod>(
    items: &[T],
    leaf_prefix: Option<&[u8]>,
) -> Option<Hash> {
    super::root_from_leaf_hashes(hash_leaves(items, bytemuck::bytes_of, leaf_prefix))
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
