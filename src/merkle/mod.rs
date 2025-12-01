#[cfg(feature = "bytemuck")]
mod bytemuck;

#[cfg(feature = "bytemuck")]
pub use bytemuck::*;

#[cfg(feature = "borsh")]
use borsh::{BorshDeserialize, BorshSerialize};
use solana_hash::Hash;
use solana_sha256_hasher::hashv;

use crate::sha2::double_hash;

/// Default prefix used when hashing leaf nodes. When double-hashing leaves,
/// this prefix is used for the second hash.
pub const DEFAULT_LEAF_PREFIX: &[u8] = &[0x00];

/// Prefix used when hashing internal nodes.
pub const NODE_PREFIX: &[u8] = &[0x01];

/// Separator used to indicate the end of an index in a Merkle proof.
pub const INDEX_SEPARATOR: u8 = 0xFF;

/// A sibling node in a Merkle proof, containing the hash and position
/// information.
#[cfg_attr(feature = "borsh", derive(BorshSerialize, BorshDeserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleSibling {
    /// The hash of the sibling node.
    pub hash: Hash,
    /// Whether this sibling is on the left or right side of the tree.
    pub side: LeafSide,
}

/// A Merkle inclusion proof consisting of a path of sibling hashes.
#[cfg_attr(feature = "borsh", derive(BorshSerialize, BorshDeserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleProof {
    siblings: Vec<MerkleSibling>,

    /// Index of the leaf in the tree. This value will be `Some` if the merkle
    /// tree enforces the index to be a part of the hashed leaf.
    ///
    /// An example on how to use this index is to be able to set a flag
    /// indicating whether a specific leaf has been used (like for a reward
    /// claim to a particular address) to prevent replay attacks.
    pub leaf_index: Option<u32>,
}

/// Indicates whether a sibling node is on the left or right side of the tree.
#[cfg_attr(feature = "borsh", derive(BorshSerialize, BorshDeserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafSide {
    Left,
    Right,
}

/// Hash a pair of nodes to create their parent node hash.
fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    hashv(&[NODE_PREFIX, left.as_ref(), right.as_ref()])
}

/// For indexed leaves: Creates a 10-byte second_prefix
/// (DEFAULT_LEAF_PREFIX[0] + 8_bytes_index + separator) then applies
/// double_hash(data, original_prefix, second_prefix) For non-indexed leaves:
/// Uses the original double_hash(data, prefix, DEFAULT_LEAF_PREFIX)
///
/// This is a helper function for `hash_leaf` and `hash_leaf_internal`.
fn hash_leaf_internal<T>(
    item: &T,
    index: Option<u32>,
    to_bytes: impl Fn(&T) -> &[u8],
    leaf_prefix: Option<&[u8]>,
) -> Hash {
    let data = to_bytes(item);
    let prefix = leaf_prefix.unwrap_or(DEFAULT_LEAF_PREFIX);

    if let Some(idx) = index {
        let mut second_prefix = [0; 6];
        second_prefix[0] = DEFAULT_LEAF_PREFIX[0];
        second_prefix[1..5].copy_from_slice(&idx.to_le_bytes());
        second_prefix[5] = INDEX_SEPARATOR;

        // Preserve double hash structure: hash(prefix + data) then hash again
        // with second prefix.
        double_hash(data, prefix, &second_prefix)
    } else {
        double_hash(data, prefix, DEFAULT_LEAF_PREFIX)
    }
}

/// NOTE: This method will panic if the number of items is greater than
/// `u32::MAX`.
fn hash_leaves_internal<T>(
    items: &[T],
    to_bytes: impl Fn(&T) -> &[u8],
    leaf_prefix: Option<&[u8]>,
    include_index: bool,
) -> Vec<Hash> {
    assert!(items.len() <= u32::MAX as usize);

    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let index = if include_index { Some(i as u32) } else { None };
            hash_leaf_internal(item, index, &to_bytes, leaf_prefix)
        })
        .collect()
}

/// Compute the Merkle root from a vector of leaf hashes. This method does not
/// check whether the number of nodes is greater than `u32::MAX` because the
/// internal method that generates these nodes asserts that the number of nodes
/// is less than or equal to `u32::MAX`.
fn root_from_leaf_hashes(mut nodes: Vec<Hash>) -> Option<Hash> {
    if nodes.is_empty() {
        return None;
    }

    let mut len = nodes.len();
    while len > 1 {
        let mut write_index = 0;
        for read_index in (0..len).step_by(2) {
            let left = nodes[read_index];

            let right_index = read_index + 1;
            let right = if read_index + 1 < len {
                nodes[read_index + 1]
            } else {
                dummy_right_leaf(right_index as u32, left)
            };
            nodes[write_index] = hash_pair(&left, &right);
            write_index += 1;
        }
        len = write_index;
    }

    nodes.first().copied()
}

impl MerkleProof {
    /// Create a `MerkleProof` from a slice of raw byte slices representing
    /// Merkle leaves.
    ///
    /// Each item will be double-hashed with a leaf prefix and used to build the
    /// tree. The proof will allow verification that the leaf at `node_index` is
    /// included in the tree. The `leaf_prefix` is optional and defaults to
    /// `DEFAULT_LEAF_PREFIX`.
    pub fn from_leaves(
        items: &[&[u8]],
        node_index: u32,
        leaf_prefix: Option<&[u8]>,
    ) -> Option<Self> {
        let hashes = hash_leaves_internal(items, |b| *b, leaf_prefix, false);
        let siblings = Self::from_hashed_leaves_internal(hashes, node_index as usize)?;
        Some(Self {
            siblings,
            leaf_index: None,
        })
    }

    /// Create a `MerkleProof` from a slice of raw byte slices with index
    /// binding. Each item will be hashed with its index to prevent position
    /// confusion attacks.
    pub fn from_indexed_leaves(
        items: &[&[u8]],
        node_index: u32,
        leaf_prefix: Option<&[u8]>,
    ) -> Option<Self> {
        let hashes = hash_leaves_internal(items, |b| *b, leaf_prefix, true);
        let siblings = Self::from_hashed_leaves_internal(hashes, node_index as usize)?;
        Some(Self {
            siblings,
            leaf_index: Some(node_index),
        })
    }

    /// Create a `MerkleProof` from pre-hashed leaves.
    fn from_hashed_leaves_internal(
        mut nodes: Vec<Hash>,
        node_index: usize,
    ) -> Option<Vec<MerkleSibling>> {
        if node_index >= nodes.len() {
            return None;
        }

        // NOTE: Pre-calculate tree depth for siblings.
        let tree_depth = (nodes.len() as f64).log2().ceil() as usize;
        let mut siblings = Vec::with_capacity(tree_depth);
        let mut index = node_index;

        while nodes.len() > 1 {
            // NOTE: Pre-allocate with exact capacity.
            let next_size = nodes.len().div_ceil(2);
            let mut next = Vec::with_capacity(next_size);

            for i in (0..nodes.len()).step_by(2) {
                let left = nodes[i];

                let right_index = i + 1;
                let right = *nodes
                    .get(i + 1)
                    .unwrap_or(&dummy_right_leaf(right_index as u32, left));
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

            debug_assert_eq!(next.len(), next_size, "next layer size mismatch");
            nodes = next;
        }

        debug_assert_eq!(siblings.len(), tree_depth, "tree depth size mismatch");
        Some(siblings)
    }

    /// Compute the Merkle root from a hashed leaf using this proof.
    fn root_from_hashed_leaf(&self, leaf: Hash) -> Hash {
        self.siblings.iter().fold(
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
        self.siblings.len()
    }

    /// Check if this proof is empty (no siblings).
    pub fn is_empty(&self) -> bool {
        self.siblings.is_empty()
    }

    pub fn root_from_leaf(&self, leaf: &[u8], leaf_prefix: Option<&[u8]>) -> Hash {
        let leaf_hash = hash_leaf_internal(&leaf, self.leaf_index, |b| *b, leaf_prefix);
        self.root_from_hashed_leaf(leaf_hash)
    }

    /// Create a proof from items implementing `AsRef<[u8]>`. The `leaf_prefix`
    /// is optional and defaults to `DEFAULT_LEAF_PREFIX`.
    pub fn from_byte_ref_leaves<T: AsRef<[u8]>>(
        items: &[T],
        node_index: u32,
        leaf_prefix: Option<&[u8]>,
    ) -> Option<Self> {
        let hashes = hash_leaves_internal(items, AsRef::as_ref, leaf_prefix, false);
        let siblings = Self::from_hashed_leaves_internal(hashes, node_index as usize)?;
        Some(Self {
            siblings,
            leaf_index: None,
        })
    }

    /// Create a proof from items implementing `AsRef<[u8]>` with index binding.
    pub fn from_indexed_byte_ref_leaves<T: AsRef<[u8]>>(
        items: &[T],
        node_index: u32,
        leaf_prefix: Option<&[u8]>,
    ) -> Option<Self> {
        let hashes = hash_leaves_internal(items, AsRef::as_ref, leaf_prefix, true);
        let siblings = Self::from_hashed_leaves_internal(hashes, node_index as usize)?;
        Some(Self {
            siblings,
            leaf_index: Some(node_index),
        })
    }

    /// Compute the root from a leaf implementing `AsRef<[u8]>`. The
    /// `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
    pub fn root_from_byte_ref_leaf<T: AsRef<[u8]>>(
        &self,
        item: &T,
        leaf_prefix: Option<&[u8]>,
    ) -> Hash {
        let leaf_hash = hash_leaf_internal(item, self.leaf_index, AsRef::as_ref, leaf_prefix);
        self.root_from_hashed_leaf(leaf_hash)
    }

    /// Check if the merkle tree is indexed.
    pub fn is_indexed(&self) -> bool {
        self.leaf_index.is_some()
    }
}

/// Compute the Merkle root from items implementing `AsRef<[u8]>`. The
/// `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
pub fn merkle_root_from_byte_ref_leaves<T: AsRef<[u8]>>(
    items: &[T],
    leaf_prefix: Option<&[u8]>,
) -> Option<Hash> {
    let hashes = hash_leaves_internal(items, AsRef::as_ref, leaf_prefix, false);
    root_from_leaf_hashes(hashes)
}

/// Compute the Merkle root from items implementing `AsRef<[u8]>` with index
/// binding. The `leaf_prefix` is optional and defaults to
/// `DEFAULT_LEAF_PREFIX`.
pub fn merkle_root_from_indexed_byte_ref_leaves<T: AsRef<[u8]>>(
    items: &[T],
    leaf_prefix: Option<&[u8]>,
) -> Option<Hash> {
    let hashes = hash_leaves_internal(items, AsRef::as_ref, leaf_prefix, true);
    root_from_leaf_hashes(hashes)
}

/// Compute the Merkle root from a slice of raw byte slices. The `leaf_prefix`
/// is optional and defaults to `DEFAULT_LEAF_PREFIX`.
///
/// Each item is treated as a leaf and double-hashed with a leaf prefix (either
/// `leaf_prefix` or `DEFAULT_LEAF_PREFIX`).
pub fn merkle_root_from_leaves(items: &[&[u8]], leaf_prefix: Option<&[u8]>) -> Option<Hash> {
    let hashes = hash_leaves_internal(items, |b| *b, leaf_prefix, false);
    root_from_leaf_hashes(hashes)
}

/// Compute the Merkle root from a slice of raw byte slices with index binding.
/// The `leaf_prefix` is optional and defaults to `DEFAULT_LEAF_PREFIX`.
pub fn merkle_root_from_indexed_leaves(
    items: &[&[u8]],
    leaf_prefix: Option<&[u8]>,
) -> Option<Hash> {
    let hashes = hash_leaves_internal(items, |b| *b, leaf_prefix, true);
    root_from_leaf_hashes(hashes)
}

impl<'a> IntoIterator for &'a MerkleProof {
    type Item = &'a MerkleSibling;
    type IntoIter = core::slice::Iter<'a, MerkleSibling>;

    fn into_iter(self) -> Self::IntoIter {
        self.siblings.iter()
    }
}

impl IntoIterator for MerkleProof {
    type Item = MerkleSibling;
    type IntoIter = std::vec::IntoIter<MerkleSibling>;

    fn into_iter(self) -> Self::IntoIter {
        self.siblings.into_iter()
    }
}

/// To avoid duplicating the last leaf, we create a dummy right leaf. This
/// method adds overhead when calculating the root because it needs to hash the
/// dummy leaf. But the cost in SVM runtime is minimal.
fn dummy_right_leaf(right_index: u32, left: Hash) -> Hash {
    hashv(&[&right_index.to_le_bytes(), left.as_ref()])
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
        assert_eq!(leaf, leaves[leaf_index as usize]);

        let proof = MerkleProof::from_leaves(&leaves, leaf_index, None).unwrap();
        let root = merkle_root_from_leaves(&leaves, None).unwrap();

        assert_eq!(proof.root_from_leaf(leaf, None), root);
        assert_ne!(proof.root_from_leaf(b"nope", None), root);

        let mut wrong_leaves = leaves.to_vec();
        wrong_leaves.retain(|l| l != &leaf);

        for wrong_leaf in wrong_leaves.into_iter() {
            assert_ne!(wrong_leaf, leaf);
            assert_ne!(proof.root_from_leaf(wrong_leaf, None), root);
        }
    }

    #[test]
    fn test_single_leaf_tree() {
        const LEAF_PREFIX: &[u8] = b"test_single_leaf_tree";

        let leaves: [&[u8]; 1] = [b"single"];
        let proof = MerkleProof::from_leaves(&leaves, 0, Some(LEAF_PREFIX)).unwrap();
        let root = merkle_root_from_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        assert_eq!(proof.root_from_leaf(b"single", Some(LEAF_PREFIX)), root);
        assert!(proof.is_empty());
        assert_eq!(proof.len(), 0);

        assert_ne!(proof.root_from_leaf(b"wrong", Some(LEAF_PREFIX)), root);
        assert_ne!(proof.root_from_leaf(b"single", None), root);
    }

    #[test]
    fn test_two_leaf_tree() {
        const LEAF_PREFIX: &[u8] = b"test_two_leaf_tree";

        let leaves: [&[u8]; 2] = [b"left", b"right"];

        // Test proof for left leaf.
        let proof_left = MerkleProof::from_leaves(&leaves, 0, Some(LEAF_PREFIX)).unwrap();
        let root = merkle_root_from_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        assert_eq!(proof_left.root_from_leaf(b"left", Some(LEAF_PREFIX)), root);
        assert_eq!(proof_left.len(), 1);
        assert!(!proof_left.is_empty());

        // Test proof for right leaf.
        let proof_right = MerkleProof::from_leaves(&leaves, 1, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(
            proof_right.root_from_leaf(b"right", Some(LEAF_PREFIX)),
            root
        );
        assert_eq!(proof_right.len(), 1);
        assert!(!proof_right.is_empty());

        // Cross-verify proofs do not work with wrong leaves.
        assert_ne!(proof_left.root_from_leaf(b"right", Some(LEAF_PREFIX)), root);
        assert_ne!(proof_right.root_from_leaf(b"left", Some(LEAF_PREFIX)), root);
    }

    #[test]
    fn test_deterministic_roots() {
        const LEAF_PREFIX: &[u8] = b"test_deterministic_roots";

        let leaves: [&[u8]; 4] = [b"apple", b"banana", b"cherry", b"date"];

        let root1 = merkle_root_from_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();
        let root2 = merkle_root_from_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(root1, root2);

        let reordered: [&[u8]; 4] = [b"banana", b"apple", b"cherry", b"date"];
        let root3 = merkle_root_from_leaves(&reordered, Some(LEAF_PREFIX)).unwrap();
        assert_ne!(root1, root3);
    }

    #[test]
    fn test_empty_tree() {
        assert!(merkle_root_from_leaves(&[], None).is_none());
        assert!(MerkleProof::from_leaves(&[], 0, None).is_none());
    }

    #[test]
    fn test_invalid_indices() {
        let leaves: [&[u8]; 3] = [b"one", b"two", b"three"];
        assert!(MerkleProof::from_leaves(&leaves, 3, None).is_none());
    }

    #[test]
    fn test_proof_structure_correctness() {
        // Test with 4 leaves to verify proof structure.
        let leaves: [&[u8]; 4] = [b"leaf0", b"leaf1", b"leaf2", b"leaf3"];
        let root = merkle_root_from_leaves(&leaves, None).unwrap();

        // For a 4-leaf tree, each proof should have exactly 2 siblings.
        for i in 0..4 {
            let proof = MerkleProof::from_leaves(&leaves, i, None).unwrap();
            assert_eq!(proof.len(), 2);
            assert_eq!(proof.root_from_leaf(leaves[i as usize], None), root);
        }
    }

    #[test]
    fn test_odd_number_of_leaves() {
        const LEAF_PREFIX: &[u8] = b"test_odd_number_of_leaves";

        let leaves: [&[u8]; 5] = [b"A", b"B", b"C", b"D", b"E"];
        let root = merkle_root_from_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        // Test proof for a leaf in the middle.
        let leaf_index = 2; // "C"
        let leaf = leaves[leaf_index as usize];

        let proof = MerkleProof::from_leaves(&leaves, leaf_index, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(proof.root_from_leaf(leaf, Some(LEAF_PREFIX)), root);

        // Verify proof length. For 5 leaves, ceil(log2(5)) = 3.
        assert_eq!(proof.len(), 3);

        // Test proof for the last leaf.
        let last_leaf_index = 4; // "E"
        let last_leaf = leaves[last_leaf_index as usize];
        let proof_last =
            MerkleProof::from_leaves(&leaves, last_leaf_index, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(
            proof_last.root_from_leaf(last_leaf, Some(LEAF_PREFIX)),
            root
        );
    }

    #[test]
    fn test_odd_number_of_leaves_cannot_spoof_root() {
        const LEAF_PREFIX: &[u8] = b"test_odd_number_of_leaves_cannot_spoof_root";

        let leaves: [&[u8]; 5] = [b"A", b"B", b"C", b"D", b"E"];
        let root = merkle_root_from_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        // We should not be able to spoof the root by duplicating the last leaf.
        let leaves_duplicated_last: [&[u8]; 6] = [b"A", b"B", b"C", b"D", b"E", b"E"];
        let root_duplicated_last =
            merkle_root_from_leaves(&leaves_duplicated_last, Some(LEAF_PREFIX)).unwrap();
        assert_ne!(root, root_duplicated_last);

        let last_left_index = 4;
        let proof_last_left =
            MerkleProof::from_leaves(&leaves_duplicated_last, last_left_index, Some(LEAF_PREFIX))
                .unwrap();
        assert_ne!(
            proof_last_left.root_from_leaf(
                leaves_duplicated_last[last_left_index as usize],
                Some(LEAF_PREFIX)
            ),
            root
        );

        let last_right_index = 5;
        let proof_last_right =
            MerkleProof::from_leaves(&leaves_duplicated_last, last_right_index, Some(LEAF_PREFIX))
                .unwrap();
        assert_ne!(
            proof_last_right.root_from_leaf(
                leaves_duplicated_last[last_right_index as usize],
                Some(LEAF_PREFIX)
            ),
            root
        );

        // We should expect that these are equal.
        assert_eq!(
            proof_last_left.root_from_leaf(
                leaves_duplicated_last[last_left_index as usize],
                Some(LEAF_PREFIX)
            ),
            proof_last_right.root_from_leaf(
                leaves_duplicated_last[last_right_index as usize],
                Some(LEAF_PREFIX)
            )
        );
    }

    #[test]
    fn test_even_tree_proof_content() {
        const LEAF_PREFIX: &[u8] = b"test_even_tree_proof_content";

        // This test verifies the exact content of a proof for a balanced,
        // power-of-two tree. For 4 leaves (A, B, C, D), the tree structure is
        // perfectly balanced.
        //
        //          root
        //         /    \
        //        /      \
        //     hash_ab    hash_cd
        //      / \        /   \
        //     /   \      /     \
        //   h(A) h(B)   h(C)   h(D)
        //
        // The proof path for h(C) is as follows:
        // 1. To get to `hash_cd`, we need `h(D)`, which is the Right sibling.
        // 2. To get to `root`, we need `hash_ab`, which is the Left sibling.
        // The final proof should be `[ (h(D), Right), (hash_ab, Left) ]`.

        let leaves: [&[u8]; 4] = [b"A", b"B", b"C", b"D"];

        // Manually compute the expected hashes
        let hash_a = double_hash(b"A", LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_b = double_hash(b"B", LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_d = double_hash(b"D", LEAF_PREFIX, DEFAULT_LEAF_PREFIX);

        // Compute the intermediate node hash
        let hash_ab = hash_pair(&hash_a, &hash_b);

        // Generate proof for leaf "C" (index 2)
        let proof = MerkleProof::from_leaves(&leaves, 2, Some(LEAF_PREFIX)).unwrap();

        // Verification
        let root = merkle_root_from_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(proof.root_from_leaf(b"C", Some(LEAF_PREFIX)), root);

        // Assert the exact proof structure as derived from the diagram
        assert_eq!(proof.len(), 2);

        // 1. First sibling is h(D) on the right
        assert_eq!(proof.siblings[0].hash, hash_d);
        assert_eq!(proof.siblings[0].side, LeafSide::Right);

        // 2. Second sibling is hash_ab on the left
        assert_eq!(proof.siblings[1].hash, hash_ab);
        assert_eq!(proof.siblings[1].side, LeafSide::Left);
    }

    #[test]
    fn test_odd_tree_proof_content() {
        // This test verifies the exact content of a proof for a
        // non-power-of-two tree. For 3 leaves (A, B, C), the tree structure is
        // unbalanced. The lone leaf C at the first level is duplicated to
        // create its partner node.
        //
        //          root
        //         /    \
        //        /      \
        //     hash_ab    hash_c_dummy
        //      / \        /   \
        //     /   \      /     \
        //   h(A) h(B)   h(C)   DUMMY
        //
        // The proof path for h(B) is as follows:
        // 1. To get to `hash_ab`, we need `h(A)`, which is the Left sibling.
        // 2. To get to `root`, we need `hash_c_dummy`, which is the Right
        //    sibling.
        // The final proof should be `[ (h(A), Left), (hash_c_dummy, Right) ]`.

        let leaves: [&[u8]; 3] = [b"A", b"B", b"C"];

        // Manually compute the expected hashes
        let hash_a = double_hash(b"A", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_c = double_hash(b"C", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);

        // At the second level, C is paired with itself to create the
        // `hash_c_dummy` node.
        let dummy = dummy_right_leaf(2 + 1, hash_c);
        let hash_c_dummy = hash_pair(&hash_c, &dummy);

        // Generate proof for leaf "B" (index 1)
        let proof = MerkleProof::from_leaves(&leaves, 1, None).unwrap();

        // Verification
        let root = merkle_root_from_leaves(&leaves, None).unwrap();
        assert_eq!(proof.root_from_leaf(b"B", None), root);

        // Assert the exact proof structure as derived from the diagram.
        assert_eq!(proof.len(), 2);

        // 1. First sibling is h(A) on the left.
        assert_eq!(proof.siblings[0].hash, hash_a);
        assert_eq!(proof.siblings[0].side, LeafSide::Left);

        // 2. Second sibling is hash_c_dummy on the right.
        assert_eq!(proof.siblings[1].hash, hash_c_dummy);
        assert_eq!(proof.siblings[1].side, LeafSide::Right);
    }

    #[test]
    fn test_generic_as_ref() {
        const LEAF_PREFIX: &[u8] = b"test_generic_asref";

        // Test various types that implement AsRef<[u8]>.
        let string_leaves = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];
        let vec_leaves = vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()];
        let str_leaves = vec!["apple", "banana", "cherry"];

        // Test from_byte_ref_leaves with different types.
        let proof_string =
            MerkleProof::from_byte_ref_leaves(&string_leaves, 1, Some(LEAF_PREFIX)).unwrap();
        let proof_vec =
            MerkleProof::from_byte_ref_leaves(&vec_leaves, 1, Some(LEAF_PREFIX)).unwrap();
        let proof_str =
            MerkleProof::from_byte_ref_leaves(&str_leaves, 1, Some(LEAF_PREFIX)).unwrap();

        // All proofs should be identical since the underlying bytes are the
        // same.
        assert_eq!(proof_string.len(), proof_vec.len());
        assert_eq!(proof_string.len(), proof_str.len());

        // Test root_from_byte_ref_leaf with different types.
        let root_string =
            proof_string.root_from_byte_ref_leaf(&string_leaves[1], Some(LEAF_PREFIX));
        let root_vec = proof_vec.root_from_byte_ref_leaf(&vec_leaves[1], Some(LEAF_PREFIX));
        let root_str = proof_str.root_from_byte_ref_leaf(&str_leaves[1], Some(LEAF_PREFIX));

        assert_eq!(root_string, root_vec);
        assert_eq!(root_string, root_str);

        // Test merkle_root_from_byte_ref_leaves.
        let merkle_root_string =
            merkle_root_from_byte_ref_leaves(&string_leaves, Some(LEAF_PREFIX)).unwrap();
        let merkle_root_vec =
            merkle_root_from_byte_ref_leaves(&vec_leaves, Some(LEAF_PREFIX)).unwrap();
        let merkle_root_str =
            merkle_root_from_byte_ref_leaves(&str_leaves, Some(LEAF_PREFIX)).unwrap();

        assert_eq!(merkle_root_string, merkle_root_vec);
        assert_eq!(merkle_root_string, merkle_root_str);

        // Verify the roots match.
        assert_eq!(root_string, merkle_root_string);
    }

    #[test]
    fn test_merkle_proof_iterators() {
        let leaves: [&[u8]; 4] = [b"one", b"two", b"three", b"four"];
        let proof = MerkleProof::from_leaves(&leaves, 2, None).unwrap();

        // Test borrowed iterator (&MerkleProof).
        let borrowed_siblings = (&proof).into_iter().collect::<Vec<_>>();
        assert_eq!(borrowed_siblings.len(), 2); // 4 leaves = 2 levels.

        // Verify we can iterate multiple times with borrowed iterator.
        let borrowed_count = (&proof).into_iter().count();
        assert_eq!(borrowed_count, 2);

        // Test that the proof still exists after borrowed iteration.
        assert_eq!(proof.len(), 2);

        // Test owned iterator (MerkleProof) - this consumes the proof.
        let proof_for_owned = MerkleProof::from_leaves(&leaves, 2, None).unwrap();
        let owned_siblings = proof_for_owned.into_iter().collect::<Vec<_>>();
        assert_eq!(owned_siblings.len(), 2);

        // Verify the siblings are the same between borrowed and owned
        for (borrowed, owned) in borrowed_siblings.iter().zip(owned_siblings.iter()) {
            assert_eq!(borrowed.hash, owned.hash);
            assert_eq!(borrowed.side, owned.side);
        }
    }

    #[test]
    fn test_indexed_merkle_proof_basic() {
        let leaves: [&[u8]; 26] = [
            b"A", b"B", b"C", b"D", b"E", b"F", b"G", b"H", b"I", b"J", b"K", b"L", b"M", b"N",
            b"O", b"P", b"Q", b"R", b"S", b"T", b"U", b"V", b"W", b"X", b"Y", b"Z",
        ];

        let leaf_index = 1;
        let leaf: &[u8] = b"B";
        assert_eq!(leaf, leaves[leaf_index as usize]);

        let proof = MerkleProof::from_indexed_leaves(&leaves, leaf_index, None).unwrap();
        let root = merkle_root_from_indexed_leaves(&leaves, None).unwrap();

        assert_eq!(proof.root_from_leaf(leaf, None), root);
        assert_ne!(proof.root_from_leaf(b"nope", None), root);

        let mut wrong_leaves = leaves.to_vec();
        wrong_leaves.retain(|l| l != &leaf);

        for wrong_leaf in wrong_leaves.into_iter() {
            assert_ne!(wrong_leaf, leaf);
            assert_ne!(proof.root_from_leaf(wrong_leaf, None), root);
        }
    }

    #[test]
    fn test_indexed_vs_non_indexed_roots_differ() {
        let leaves: [&[u8]; 4] = [b"A", b"B", b"C", b"D"];

        let regular_root = merkle_root_from_leaves(&leaves, None).unwrap();
        let indexed_root = merkle_root_from_indexed_leaves(&leaves, None).unwrap();

        // Roots should differ between indexed and non-indexed.
        assert_ne!(regular_root, indexed_root);

        let common_index = 0;
        let common_leaf = b"A";
        let common_prefix = None;

        // Regular proof should not verify with indexed root.
        let regular_proof = MerkleProof::from_leaves(&leaves, common_index, common_prefix).unwrap();
        assert_ne!(
            regular_proof.root_from_leaf(common_leaf, common_prefix),
            indexed_root
        );

        // Indexed proof should not verify with regular root.
        let indexed_proof =
            MerkleProof::from_indexed_leaves(&leaves, common_index, common_prefix).unwrap();
        assert_ne!(
            indexed_proof.root_from_leaf(common_leaf, common_prefix),
            regular_root
        );
    }

    #[test]
    fn test_indexed_single_leaf_tree() {
        const LEAF_PREFIX: &[u8] = b"test_indexed_single_leaf_tree";

        let leaves: [&[u8]; 1] = [b"single"];
        let proof = MerkleProof::from_indexed_leaves(&leaves, 0, Some(LEAF_PREFIX)).unwrap();
        let root = merkle_root_from_indexed_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        assert_eq!(proof.root_from_leaf(b"single", Some(LEAF_PREFIX)), root);
        assert!(proof.is_empty());
        assert_eq!(proof.len(), 0);

        assert_ne!(proof.root_from_leaf(b"wrong", Some(LEAF_PREFIX)), root);
        assert_ne!(proof.root_from_leaf(b"single", None), root);
    }

    #[test]
    fn test_indexed_two_leaf_tree() {
        const LEAF_PREFIX: &[u8] = b"test_indexed_two_leaf_tree";

        let leaves: [&[u8]; 2] = [b"left", b"right"];

        // Test proof for left leaf at index 0.
        let proof_left = MerkleProof::from_indexed_leaves(&leaves, 0, Some(LEAF_PREFIX)).unwrap();
        let root = merkle_root_from_indexed_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        assert_eq!(proof_left.root_from_leaf(b"left", Some(LEAF_PREFIX)), root);
        assert_eq!(proof_left.len(), 1);
        assert!(!proof_left.is_empty());

        // Test proof for right leaf at index 1.
        let proof_right = MerkleProof::from_indexed_leaves(&leaves, 1, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(
            proof_right.root_from_leaf(b"right", Some(LEAF_PREFIX)),
            root
        );
        assert_eq!(proof_right.len(), 1);
        assert!(!proof_right.is_empty());

        // Cross-verify proofs don't work with wrong leaves.
        assert_ne!(proof_left.root_from_leaf(b"right", Some(LEAF_PREFIX)), root);
        assert_ne!(proof_right.root_from_leaf(b"left", Some(LEAF_PREFIX)), root);
    }

    #[test]
    fn test_indexed_deterministic_roots() {
        const LEAF_PREFIX: &[u8] = b"test_indexed_deterministic_roots";

        let leaves: [&[u8]; 4] = [b"apple", b"banana", b"cherry", b"date"];

        let root1 = merkle_root_from_indexed_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();
        let root2 = merkle_root_from_indexed_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(root1, root2);

        let reordered: [&[u8]; 4] = [b"banana", b"apple", b"cherry", b"date"];
        let root3 = merkle_root_from_indexed_leaves(&reordered, Some(LEAF_PREFIX)).unwrap();

        // With indexed leaves, reordering changes the root because each leaf
        // is bound to its index.
        assert_ne!(root1, root3);
    }

    #[test]
    fn test_indexed_empty_tree() {
        assert!(merkle_root_from_indexed_leaves(&[], None).is_none());
        assert!(MerkleProof::from_indexed_leaves(&[], 0, None).is_none());
    }

    #[test]
    fn test_indexed_invalid_indices() {
        let leaves: [&[u8]; 3] = [b"one", b"two", b"three"];
        assert!(MerkleProof::from_indexed_leaves(&leaves, 3, None).is_none());
    }

    #[test]
    fn test_indexed_odd_number_of_leaves() {
        const LEAF_PREFIX: &[u8] = b"test_indexed_odd_number_of_leaves";

        let leaves: [&[u8]; 5] = [b"A", b"B", b"C", b"D", b"E"];
        let root = merkle_root_from_indexed_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        // Test proof for a leaf in the middle.
        let leaf_index = 2; // "C"
        let leaf = leaves[leaf_index as usize];

        let proof =
            MerkleProof::from_indexed_leaves(&leaves, leaf_index, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(proof.root_from_leaf(leaf, Some(LEAF_PREFIX)), root);

        // Verify proof length. For 5 leaves, ceil(log2(5)) = 3.
        assert_eq!(proof.len(), 3);

        // Test proof for the last leaf.
        let last_leaf_index = 4; // "E"
        let last_leaf = leaves[last_leaf_index as usize];
        let proof_last =
            MerkleProof::from_indexed_leaves(&leaves, last_leaf_index, Some(LEAF_PREFIX)).unwrap();
        assert_eq!(
            proof_last.root_from_leaf(last_leaf, Some(LEAF_PREFIX)),
            root
        );
    }

    #[test]
    fn test_indexed_odd_number_of_leaves_cannot_spoof_root() {
        const LEAF_PREFIX: &[u8] = b"test_indexed_odd_number_of_leaves_cannot_spoof_root";

        let leaves: [&[u8]; 5] = [b"A", b"B", b"C", b"D", b"E"];
        let root = merkle_root_from_indexed_leaves(&leaves, Some(LEAF_PREFIX)).unwrap();

        // We should not be able to spoof the root by duplicating the last leaf.
        let leaves_duplicated_last: [&[u8]; 6] = [b"A", b"B", b"C", b"D", b"E", b"E"];
        let root_duplicated_last =
            merkle_root_from_indexed_leaves(&leaves_duplicated_last, Some(LEAF_PREFIX)).unwrap();
        assert_ne!(root, root_duplicated_last);

        let last_left_index = 4;
        let proof_last_left = MerkleProof::from_indexed_leaves(
            &leaves_duplicated_last,
            last_left_index,
            Some(LEAF_PREFIX),
        )
        .unwrap();
        assert_ne!(
            proof_last_left.root_from_leaf(
                leaves_duplicated_last[last_left_index as usize],
                Some(LEAF_PREFIX)
            ),
            root
        );

        let last_right_index = 5;
        let proof_last_right = MerkleProof::from_indexed_leaves(
            &leaves_duplicated_last,
            last_right_index,
            Some(LEAF_PREFIX),
        )
        .unwrap();
        assert_ne!(
            proof_last_right.root_from_leaf(
                leaves_duplicated_last[last_right_index as usize],
                Some(LEAF_PREFIX)
            ),
            root
        );

        // Even though the data is the same ("E"), with indexed leaves they
        // produce the same root because they're both validating against the
        // 6-leaf tree. The important point is that neither can produce the
        // original 5-leaf root.
        assert_eq!(
            proof_last_left.root_from_leaf(
                leaves_duplicated_last[last_left_index as usize],
                Some(LEAF_PREFIX)
            ),
            root_duplicated_last
        );
        assert_eq!(
            proof_last_right.root_from_leaf(
                leaves_duplicated_last[last_right_index as usize],
                Some(LEAF_PREFIX),
            ),
            root_duplicated_last
        );
    }

    #[test]
    fn test_indexed_generic_as_ref() {
        const LEAF_PREFIX: &[u8] = b"test_indexed_generic_as_ref";

        // Test various types that implement AsRef<[u8]>.
        let string_leaves = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];
        let vec_leaves = vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()];
        let str_leaves = vec!["apple", "banana", "cherry"];

        // Test from_indexed_byte_ref_leaves with different types.
        let proof_string =
            MerkleProof::from_indexed_byte_ref_leaves(&string_leaves, 1, Some(LEAF_PREFIX))
                .unwrap();
        let proof_vec =
            MerkleProof::from_indexed_byte_ref_leaves(&vec_leaves, 1, Some(LEAF_PREFIX)).unwrap();
        let proof_str =
            MerkleProof::from_indexed_byte_ref_leaves(&str_leaves, 1, Some(LEAF_PREFIX)).unwrap();

        // All proofs should be identical since the underlying bytes are the
        // same.
        assert_eq!(proof_string.len(), proof_vec.len());
        assert_eq!(proof_string.len(), proof_str.len());

        // Test root_from_byte_ref_leaf with different types.
        let root_string =
            proof_string.root_from_byte_ref_leaf(&string_leaves[1], Some(LEAF_PREFIX));
        let root_vec = proof_vec.root_from_byte_ref_leaf(&vec_leaves[1], Some(LEAF_PREFIX));
        let root_str = proof_str.root_from_byte_ref_leaf(&str_leaves[1], Some(LEAF_PREFIX));

        assert_eq!(root_string, root_vec);
        assert_eq!(root_string, root_str);

        // Test merkle_root_from_indexed_byte_ref_leaves.
        let merkle_root_string =
            merkle_root_from_indexed_byte_ref_leaves(&string_leaves, Some(LEAF_PREFIX)).unwrap();
        let merkle_root_vec =
            merkle_root_from_indexed_byte_ref_leaves(&vec_leaves, Some(LEAF_PREFIX)).unwrap();
        let merkle_root_str =
            merkle_root_from_indexed_byte_ref_leaves(&str_leaves, Some(LEAF_PREFIX)).unwrap();

        assert_eq!(merkle_root_string, merkle_root_vec);
        assert_eq!(merkle_root_string, merkle_root_str);

        // Verify the roots match.
        assert_eq!(root_string, merkle_root_string);
    }

    #[test]
    fn test_indexed_proof_structure() {
        // Test that indexed proofs maintain correct structure.
        let leaves: [&[u8]; 4] = [b"leaf0", b"leaf1", b"leaf2", b"leaf3"];
        let root = merkle_root_from_indexed_leaves(&leaves, None).unwrap();

        // For a 4-leaf tree, each proof should have exactly 2 siblings.
        for i in 0..4 {
            let proof = MerkleProof::from_indexed_leaves(&leaves, i, None).unwrap();
            assert_eq!(proof.len(), 2);
            assert_eq!(proof.root_from_leaf(leaves[i as usize], None), root);

            // Verify the proof has the correct leaf index stored.
            assert_eq!(proof.leaf_index.unwrap(), i);
        }
    }

    #[test]
    fn test_indexed_same_data_different_positions() {
        // Test that the same data at different positions produces different
        // hashes. This is important because it ensures that the proof is
        // position-bound, even though this is not a realistic use case.
        let leaves: [&[u8]; 4] = [b"same", b"same", b"different", b"same"];

        let proof0 = MerkleProof::from_indexed_leaves(&leaves, 0, None).unwrap();
        let proof1 = MerkleProof::from_indexed_leaves(&leaves, 1, None).unwrap();
        let proof3 = MerkleProof::from_indexed_leaves(&leaves, 3, None).unwrap();

        let root = merkle_root_from_indexed_leaves(&leaves, None).unwrap();

        // All proofs should verify correctly with their respective positions.
        assert_eq!(proof0.root_from_leaf(b"same", None), root);
        assert_eq!(proof1.root_from_leaf(b"same", None), root);
        assert_eq!(proof3.root_from_leaf(b"same", None), root);

        // But the proofs themselves should be different because they are for
        // different positions.
        assert_ne!(proof0.siblings[0].hash, proof1.siblings[0].hash);
        assert_ne!(proof0.siblings[0].hash, proof3.siblings[0].hash);
        assert_ne!(proof1.siblings[0].hash, proof3.siblings[0].hash);
    }

    #[test]
    fn test_indexed_with_custom_prefix() {
        const CUSTOM_PREFIX: &[u8] = b"my_custom_prefix";

        let leaves: [&[u8]; 3] = [b"alpha", b"beta", b"gamma"];

        // Create indexed proof with custom prefix.
        let proof = MerkleProof::from_indexed_leaves(&leaves, 1, Some(CUSTOM_PREFIX)).unwrap();
        let root = merkle_root_from_indexed_leaves(&leaves, Some(CUSTOM_PREFIX)).unwrap();

        // Should verify with the same custom prefix.
        assert_eq!(proof.root_from_leaf(b"beta", Some(CUSTOM_PREFIX)), root);

        // Should NOT verify with different or no prefix.
        assert_ne!(proof.root_from_leaf(b"beta", None), root);
        assert_ne!(proof.root_from_leaf(b"beta", Some(b"wrong_prefix")), root);

        // Different prefix should produce different root.
        let root_no_prefix = merkle_root_from_indexed_leaves(&leaves, None).unwrap();
        assert_ne!(root, root_no_prefix);
    }

    #[test]
    fn test_odd_tree_multiple_dummies_proof_content() {
        // This test verifies the exact content of a proof for a
        // non-power-of-two tree. For 7 leaves (A, B, C, D, E, F, G), the tree structure is
        // unbalanced. The lone leaf G at the first level is duplicated to
        // create its partner node.
        //
        //                     ROOT
        //                    /    \
        //                   /      \
        //            hash_ABCD      hash_EFG_dummy
        //             /    \           /        \
        //            /      \         /          \
        //       hash_AB   hash_CD   hash_EF   hash_G_dummy
        //        /  \      /  \      /  \       /      \
        //       /    \    /    \    /    \     /        \
        //    h(A)  h(B) h(C) h(D) h(E) h(F) h(G)  DUMMY(7,h(G))
        //
        // The proof path for h(B) is as follows:
        // 1. To get to `hash_AB`, we need `h(A)`, which is the Left sibling.
        // 2. To get to `hash_ABCD`, we need `hash_CD`, which is the Right sibling.
        // 3. To get to `root`, we need `hash_EFG_dummy`, which is the Right sibling.
        // The final proof should be `[ (h(A), Left), (hash_CD, Right), (hash_EFG_dummy, Right) ]`.

        let leaves: [&[u8]; 7] = [b"A", b"B", b"C", b"D", b"E", b"F", b"G"];

        // Manually compute the expected hashes
        let hash_a = double_hash(b"A", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_c = double_hash(b"C", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_d = double_hash(b"D", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_e = double_hash(b"E", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_f = double_hash(b"F", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);
        let hash_g = double_hash(b"G", DEFAULT_LEAF_PREFIX, DEFAULT_LEAF_PREFIX);

        // Compute intermediate hashes
        let hash_cd = hash_pair(&hash_c, &hash_d);
        let hash_ef = hash_pair(&hash_e, &hash_f);

        // At the second level, G is paired with a dummy to create the `hash_g_dummy` node.
        let dummy = dummy_right_leaf(6 + 1, hash_g);
        let hash_g_dummy = hash_pair(&hash_g, &dummy);

        // At the third level, hash_EF is paired with hash_G_dummy to create `hash_EFG_dummy`.
        let hash_efg_dummy = hash_pair(&hash_ef, &hash_g_dummy);

        // Generate proof for leaf "B" (index 1)
        let proof = MerkleProof::from_leaves(&leaves, 1, None).unwrap();

        // Verification
        let root = merkle_root_from_leaves(&leaves, None).unwrap();
        assert_eq!(proof.root_from_leaf(b"B", None), root);

        // Assert the exact proof structure as derived from the diagram.
        assert_eq!(proof.len(), 3);

        // 1. First sibling is h(A) on the left.
        assert_eq!(proof.siblings[0].hash, hash_a);
        assert_eq!(proof.siblings[0].side, LeafSide::Left);

        // 2. Second sibling is hash_CD on the right.
        assert_eq!(proof.siblings[1].hash, hash_cd);
        assert_eq!(proof.siblings[1].side, LeafSide::Right);

        // 3. Third sibling is hash_EFG_dummy on the right.
        assert_eq!(proof.siblings[2].hash, hash_efg_dummy);
        assert_eq!(proof.siblings[2].side, LeafSide::Right);
    }
}
