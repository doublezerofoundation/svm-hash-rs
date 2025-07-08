use solana_hash::Hash;
use solana_sha256_hasher::hashv;

use crate::sha2::double_hash;

#[cfg(feature = "bytemuck")]
use bytemuck::Pod;

/// Prefix used when hashing leaf nodes.
pub const LEAF_PREFIX: &[u8] = &[0x00];

/// Prefix used when hashing internal nodes.
pub const NODE_PREFIX: &[u8] = &[0x01];

/// A sibling node in a Merkle proof, containing the hash and position information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleSibling {
    /// The hash of the sibling node.
    pub hash: Hash,
    /// Whether this sibling is on the left or right side of the tree.
    pub side: LeafSide,
}

/// A Merkle inclusion proof consisting of a path of sibling hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleProof(Vec<MerkleSibling>);

/// Indicates whether a sibling node is on the left or right side of the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafSide {
    Left,
    Right,
}

/// Hash a pair of nodes to create their parent node hash.
fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    hashv(&[NODE_PREFIX, left.as_ref(), right.as_ref()])
}

/// Convert a slice of items to their corresponding leaf hashes.
fn hash_leaves<T>(items: &[T], to_bytes: impl Fn(&T) -> &[u8]) -> Vec<Hash> {
    items
        .iter()
        .map(|item| double_hash(to_bytes(item), LEAF_PREFIX, LEAF_PREFIX))
        .collect()
}

/// Compute the Merkle root from a vector of leaf hashes.
fn root_from_leaf_hashes(mut nodes: Vec<Hash>) -> Option<Hash> {
    if nodes.is_empty() {
        return None;
    }

    while nodes.len() > 1 {
        nodes = nodes
            .chunks(2)
            .map(|pair| {
                let left = pair[0];
                let right = *pair.get(1).unwrap_or(&left);
                hash_pair(&left, &right)
            })
            .collect();
    }

    nodes.first().copied()
}

impl MerkleProof {
    /// Create a `MerkleProof` from a slice of raw byte slices representing Merkle leaves.
    ///
    /// Each item will be double-hashed with a leaf prefix and used to build the tree.
    /// The proof will allow verification that the leaf at `node_index` is included in the tree.
    pub fn from_leaves(items: &[&[u8]], node_index: usize) -> Option<Self> {
        let hashes: Vec<Hash> = items
            .iter()
            .map(|bytes| double_hash(bytes, LEAF_PREFIX, LEAF_PREFIX))
            .collect();
        Self::from_hashed_leaves(hashes, node_index)
    }

    /// Create a `MerkleProof` from pre-hashed leaves.
    fn from_hashed_leaves(mut nodes: Vec<Hash>, node_index: usize) -> Option<Self> {
        if node_index >= nodes.len() {
            return None;
        }

        // TODO: Optimize this by precalculating the number of siblings.
        let mut siblings = Vec::new();
        let mut index = node_index;

        while nodes.len() > 1 {
            let mut next = Vec::new();

            for i in (0..nodes.len()).step_by(2) {
                let left = nodes[i];
                let right = *nodes.get(i + 1).unwrap_or(&left);
                next.push(hash_pair(&left, &right));

                if i == index || i + 1 == index {
                    let sibling = if i == index { right } else { left };
                    let side = if i + 1 == index {
                        LeafSide::Left
                    } else {
                        LeafSide::Right
                    };
                    siblings.push(MerkleSibling {
                        hash: sibling,
                        side,
                    });
                    index = next.len() - 1;
                }
            }

            nodes = next;
        }

        Some(Self(siblings))
    }

    /// Compute the Merkle root from a hashed leaf using this proof.
    fn root_from_hashed_leaf(&self, leaf: Hash) -> Hash {
        self.0.iter().fold(
            leaf,
            |hash,
             MerkleSibling {
                 hash: sibling,
                 side,
             }| {
                match side {
                    LeafSide::Left => hash_pair(sibling, &hash),
                    LeafSide::Right => hash_pair(&hash, sibling),
                }
            },
        )
    }

    /// Get the number of siblings in this proof (equivalent to tree depth).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if this proof is empty (no siblings).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn root_from_leaf(&self, leaf: &[u8]) -> Hash {
        let leaf = double_hash(leaf, LEAF_PREFIX, LEAF_PREFIX);
        self.root_from_hashed_leaf(leaf)
    }

    /// Create a proof from items implementing `AsRef<[u8]>`.
    pub fn from_byte_ref_leaves<T: AsRef<[u8]>>(items: &[T], node_index: usize) -> Option<Self> {
        Self::from_hashed_leaves(hash_leaves(items, AsRef::as_ref), node_index)
    }

    /// Compute the root from a leaf implementing `AsRef<[u8]>`.
    pub fn root_from_byte_ref_leaf<T: AsRef<[u8]>>(&self, item: &T) -> Hash {
        let leaf = double_hash(item.as_ref(), LEAF_PREFIX, LEAF_PREFIX);
        self.root_from_hashed_leaf(leaf)
    }

    /// Create a proof from POD (Plain Old Data) items when bytemuck feature is enabled.
    #[cfg(feature = "bytemuck")]
    pub fn from_pod_leaves<T: Pod>(items: &[T], node_index: usize) -> Option<Self> {
        Self::from_hashed_leaves(hash_leaves(items, bytemuck::bytes_of), node_index)
    }

    /// Compute the root from a POD leaf when bytemuck feature is enabled.
    #[cfg(feature = "bytemuck")]
    pub fn root_from_pod_leaf<T: Pod>(&self, item: &T) -> Hash {
        let leaf = double_hash(bytemuck::bytes_of(item), LEAF_PREFIX, LEAF_PREFIX);
        self.root_from_hashed_leaf(leaf)
    }
}

/// Compute the Merkle root from POD items when bytemuck feature is enabled.
#[cfg(feature = "bytemuck")]
pub fn merkle_root_from_pod_leaves<T: Pod>(items: &[T]) -> Option<Hash> {
    root_from_leaf_hashes(hash_leaves(items, bytemuck::bytes_of))
}

/// Compute the Merkle root from items implementing `AsRef<[u8]>`
pub fn merkle_root_from_byte_ref_leaves<T: AsRef<[u8]>>(items: &[T]) -> Option<Hash> {
    root_from_leaf_hashes(hash_leaves(items, AsRef::as_ref))
}

/// Compute the Merkle root from a slice of raw byte slices.
///
/// Each item is treated as a leaf and double-hashed with a leaf prefix.
pub fn merkle_root_from_leaves(items: &[&[u8]]) -> Option<Hash> {
    let hashes: Vec<Hash> = items
        .iter()
        .map(|bytes| double_hash(bytes, LEAF_PREFIX, LEAF_PREFIX))
        .collect();
    root_from_leaf_hashes(hashes)
}

// Iterator implementations
impl<'a> IntoIterator for &'a MerkleProof {
    type Item = &'a MerkleSibling;
    type IntoIter = core::slice::Iter<'a, MerkleSibling>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl IntoIterator for MerkleProof {
    type Item = MerkleSibling;
    type IntoIter = std::vec::IntoIter<MerkleSibling>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_proof_basic() {
        let leaves: [&[u8]; 26] = [
            b"A", b"B", b"C", b"D", b"E", b"F", b"G", b"H", b"I", b"J", b"K", b"L", b"M", b"N",
            b"O", b"P", b"Q", b"R", b"S", b"T", b"U", b"V", b"W", b"X", b"Y", b"Z",
        ];

        let leaf_index = 1;
        let leaf: &[u8] = b"B";
        assert_eq!(leaf, leaves[leaf_index]);

        let proof = MerkleProof::from_leaves(&leaves, leaf_index).unwrap();
        let root = merkle_root_from_leaves(&leaves).unwrap();

        assert_eq!(proof.root_from_leaf(leaf), root);
        assert_ne!(proof.root_from_leaf(b"nope"), root);

        let mut wrong_leaves = leaves.to_vec();
        wrong_leaves.retain(|l| l != &leaf);

        for wrong_leaf in wrong_leaves.into_iter() {
            assert_ne!(wrong_leaf, leaf);
            assert_ne!(proof.root_from_leaf(wrong_leaf), root);
        }
    }

    #[test]
    fn test_single_leaf_tree() {
        let leaves: [&[u8]; 1] = [b"single"];
        let proof = MerkleProof::from_leaves(&leaves, 0).unwrap();
        let root = merkle_root_from_leaves(&leaves).unwrap();

        assert_eq!(proof.root_from_leaf(b"single"), root);
        assert!(proof.is_empty());
        assert_eq!(proof.len(), 0);

        assert_ne!(proof.root_from_leaf(b"wrong"), root);
    }

    #[test]
    fn test_two_leaf_tree() {
        let leaves: [&[u8]; 2] = [b"left", b"right"];

        // Test proof for left leaf
        let proof_left = MerkleProof::from_leaves(&leaves, 0).unwrap();
        let root = merkle_root_from_leaves(&leaves).unwrap();

        assert_eq!(proof_left.root_from_leaf(b"left"), root);
        assert_eq!(proof_left.len(), 1);
        assert!(!proof_left.is_empty());

        // Test proof for right leaf
        let proof_right = MerkleProof::from_leaves(&leaves, 1).unwrap();
        assert_eq!(proof_right.root_from_leaf(b"right"), root);
        assert_eq!(proof_right.len(), 1);
        assert!(!proof_right.is_empty());

        // Cross-verify proofs don't work with wrong leaves
        assert_ne!(proof_left.root_from_leaf(b"right"), root);
        assert_ne!(proof_right.root_from_leaf(b"left"), root);
    }

    #[test]
    fn test_deterministic_roots() {
        let leaves: [&[u8]; 4] = [b"apple", b"banana", b"cherry", b"date"];

        let root1 = merkle_root_from_leaves(&leaves).unwrap();
        let root2 = merkle_root_from_leaves(&leaves).unwrap();
        assert_eq!(root1, root2);

        let reordered: [&[u8]; 4] = [b"banana", b"apple", b"cherry", b"date"];
        let root3 = merkle_root_from_leaves(&reordered).unwrap();
        assert_ne!(root1, root3);
    }

    #[test]
    fn test_empty_tree() {
        assert!(merkle_root_from_leaves(&[]).is_none());
        assert!(MerkleProof::from_leaves(&[], 0).is_none());
    }

    #[test]
    fn test_invalid_indices() {
        let leaves: [&[u8]; 3] = [b"one", b"two", b"three"];
        assert!(MerkleProof::from_leaves(&leaves, 3).is_none());
    }

    #[test]
    fn test_proof_structure_correctness() {
        // Test with 4 leaves to verify proof structure
        let leaves: [&[u8]; 4] = [b"leaf0", b"leaf1", b"leaf2", b"leaf3"];
        let root = merkle_root_from_leaves(&leaves).unwrap();

        // For a 4-leaf tree, each proof should have exactly 2 siblings
        for i in 0..4 {
            let proof = MerkleProof::from_leaves(&leaves, i).unwrap();
            assert_eq!(proof.len(), 2);
            assert_eq!(proof.root_from_leaf(leaves[i]), root);
        }
    }
}

#[cfg(feature = "bytemuck")]
#[cfg(test)]
mod bytemuck_tests {
    use super::*;

    #[test]
    fn test_pod_leaves() {
        use bytemuck::{Pod, Zeroable};

        #[derive(Clone, Copy, Default, Pod, Zeroable)]
        #[repr(C)]
        struct TestData {
            id: Hash,
            value: u64,
        }

        let data = [
            TestData::default(),
            TestData { id: Hash::new_unique(), value: 100 },
            TestData { id: Hash::new_unique(), value: 200 },
            TestData { id: Hash::new_unique(), value: 300 },
            TestData { id: Hash::new_unique(), value: 400 },
        ];

        let proof = MerkleProof::from_pod_leaves(&data, 2).unwrap();
        let root = merkle_root_from_pod_leaves(&data).unwrap();

        assert_eq!(proof.root_from_pod_leaf(&data[2]), root);
        assert_ne!(proof.root_from_pod_leaf(&data[0]), root);
    }
}
