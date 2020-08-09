import numba as nb
import numpy as np


# Neighbor joining takes as input a distance matrix specifying the distance between each pair of taxa.
# The algorithm starts with a completely unresolved tree, whose topology corresponds to that of a star network,
# and iterates over the following steps until the tree is completely resolved and all branch lengths are known:
#     1. Based on the current distance matrix calculate the matrix Q (defined below).
#     2. Find the pair of distinct taxa i and j (i.e. with i ≠ j) for which Q(i,j) has its lowest value.
#        These taxa are joined to a newly created node, which is connected to the central node
#     3. Calculate the distance from each of the taxa in the pair to this new node (branch lengths).
#     4. Calculate the distance from each of the taxa outside of this pair to the new node.
#     5. Start the algorithm again, replacing the pair of joined neighbors with the new node and using the distances calculated in the previous step.


@nb.njit
# @numba_cc.export('neighbor_joining', '(f64[:])')
def neighbor_joining(distance_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Runs the neighbor joining algorithm on a distance matrix
    Returns guide tree as adjacency list + branch lengths

    Parameters
    ----------
    distance_matrix

    Returns
    -------
    tree (node_1, node_2) pairs
    branch_lengths corresponding to each tree pair
    (node indices above length of the distance_matrix correspond to intermediate nodes)
    """
    length = n = distance_matrix.shape[0]
    tree = np.zeros((length ** 2, 2), dtype=np.uint64)
    branch_lengths = np.zeros((length ** 2, 1), dtype=np.float64)
    index = 0
    true_indices = np.array(list(range(length)))
    num_intermediate_nodes = 0
    while n > 3:
        # indices of nodes to be joined (according to the current distance_matrix, not the initial one!)
        min_ij = _find_join_nodes(distance_matrix)
        # branch lengths of each node being joined to the new node created after they are joined
        delta_ij_u = _find_branch_length(distance_matrix, min_ij[0], min_ij[1])

        # make an intermediate node
        intermediate_node = num_intermediate_nodes + length
        num_intermediate_nodes += 1

        # add to tree
        tree[index] = np.array((true_indices[min_ij[0]], intermediate_node))
        branch_lengths[index] = delta_ij_u[0]
        index += 1
        tree[index] = np.array((true_indices[min_ij[1]], intermediate_node))
        branch_lengths[index] = delta_ij_u[1]
        index += 1

        # Distances of remaining indices to newly created node (step 4)
        indices = np.array([i for i in range(n) if i != min_ij[0] and i != min_ij[1]])
        new_distance_matrix = np.zeros((n - 1, n - 1))
        new_distance_matrix[1:, 1:] = distance_matrix[indices, :][:, indices]
        for i in range(len(indices)):
            new_distance_matrix[0, i + 1] = new_distance_matrix[i + 1, 0] = 0.5 * (
                distance_matrix[min_ij[0], indices[i]]
                + distance_matrix[min_ij[1], indices[i]]
                - distance_matrix[min_ij[0], min_ij[1]]
            )

        # Repeat (step 5)
        distance_matrix = new_distance_matrix
        n = distance_matrix.shape[0]
        true_indices = np.array(
            [intermediate_node] + [true_indices[i] for i in indices]
        )

    # Last 3 nodes
    delta_ij_u = _find_branch_length(distance_matrix, 1, 2)
    intermediate_node = num_intermediate_nodes + length
    num_intermediate_nodes += 1

    tree[index] = np.array((true_indices[1], intermediate_node))
    branch_lengths[index] = delta_ij_u[0]
    index += 1

    tree[index] = np.array((true_indices[2], intermediate_node))
    branch_lengths[index] = delta_ij_u[1]
    index += 1

    tree[index] = np.array((true_indices[0], intermediate_node))
    branch_lengths[index] = 0.5 * (
        distance_matrix[1, 0] + distance_matrix[2, 0] - distance_matrix[1, 2]
    )
    index += 1

    return tree[:index], branch_lengths[:index]


# Q matrix calculation + minimum i, j (step 1 & 2)
@nb.njit
# @numba_cc.export('_find_join_nodes', '(f64[:])')
def _find_join_nodes(distance_matrix):
    """
    Finds which nodes to join next

    Parameters
    ----------
    distance_matrix

    Returns
    -------
    indices (i, j) of nodes to join
    """
    n = distance_matrix.shape[0]
    q_matrix = np.zeros((n, n))
    q_matrix[:] = np.inf
    min_ij = np.array([0, 0], dtype=np.uint64)
    min_q = np.inf
    for i in range(n):
        for j in range(n):
            if i != j:
                q_matrix[i, j] = (
                    (n - 2) * distance_matrix[i, j]
                    - np.sum(distance_matrix[i, :])
                    - np.sum(distance_matrix[j, :])
                )
                # q_matrix[i, j] = distance_matrix[i, j]
                if q_matrix[i, j] < min_q:
                    min_ij[0], min_ij[1] = i, j
                    min_q = q_matrix[i, j]
    return min_ij


# Branch length calculation (step 3)
@nb.njit
# @numba_cc.export('_find_branch_length', '(f64[:], i64, i64)')
def _find_branch_length(distance_matrix, i, j):
    """
    Finds branch lengths of old nodes to newly created node

    Parameters
    ----------
    distance_matrix
    i
        first node to join
    j
        second node to join

    Returns
    -------
    (branch length of i to new node, branch length of j to new node)
    """
    n = distance_matrix.shape[0]
    delta_i_u = 0.5 * distance_matrix[i, j] + (0.5 / (n - 2)) * (
        np.sum(distance_matrix[i, :]) - np.sum(distance_matrix[j, :])
    )
    delta_j_u = distance_matrix[i, j] - delta_i_u
    return np.array([delta_i_u, delta_j_u])
