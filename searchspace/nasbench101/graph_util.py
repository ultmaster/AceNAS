import hashlib
import itertools
import logging

import numpy as np


_logger = logging.getLogger(__name__)


def gen_is_edge_fn(bits):
    """Generate a boolean function for the edge connectivity.
    Given a bitstring FEDCBA and a 4x4 matrix, the generated matrix is
        [[0, A, B, D],
         [0, 0, C, E],
         [0, 0, 0, F],
         [0, 0, 0, 0]]
    Note that this function is agnostic to the actual matrix dimension due to
    order in which elements are filled out (column-major, starting from least
    significant bit). For example, the same FEDCBA bitstring (0-padded) on a 5x5
    matrix is
        [[0, A, B, D, 0],
         [0, 0, C, E, 0],
         [0, 0, 0, F, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    Args:
        bits: integer which will be interpreted as a bit mask.
    Returns:
        vectorized function that returns True when an edge is present.
    """

    def is_edge(x, y):
        """Is there an edge from x to y (0-indexed)?"""
        if x >= y:
            return 0
        # Map x, y to index into bit string
        index = x + (y * (y - 1) // 2)
        return (bits >> index) % 2 == 1

    return np.vectorize(is_edge)


def is_full_dag(matrix):
    """Full DAG == all vertices on a path from vert 0 to (V-1).
    i.e. no disconnected or "hanging" vertices.
    It is sufficient to check for:
        1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
        2) no cols of 0 except for col 0 (only input vertex has no in-edges)
    Args:
        matrix: V x V upper-triangular adjacency matrix
    Returns:
        True if the there are no dangling vertices.
    """
    shape = np.shape(matrix)

    rows = matrix[:shape[0] - 1, :] == 0
    rows = np.all(rows, axis=1)  # Any row with all 0 will be True
    rows_bad = np.any(rows)

    cols = matrix[:, 1:] == 0
    cols = np.all(cols, axis=0)  # Any col with all 0 will be True
    cols_bad = np.any(cols)

    return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
    """Computes number of edges in adjacency matrix."""
    return np.sum(matrix)


def hash_module(matrix, labeling):
    """Computes a graph-invariance MD5 hash of the matrix and label pair.
    Args:
        matrix: np.ndarray square upper-triangular adjacency matrix.
        labeling: list of int labels of length equal to both dimensions of matrix.
    Returns:
        MD5 hash of the matrix and labeling.
    """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint


def permute_graph(graph, label, permutation):
    """Permutes the graph and labels based on permutation.
    Args:
        graph: np.ndarray adjacency matrix.
        label: list of labels of same length as graph dimensions.
        permutation: a permutation list of ints of same length as graph dimensions.
    Returns:
        np.ndarray where vertex permutation[v] is vertex v from the original graph
    """
    # vertex permutation[v] in new graph is vertex v in the old graph
    forward_perm = zip(permutation, list(range(len(permutation))))
    inverse_perm = [x[1] for x in sorted(forward_perm)]

    def edge_fn(x, y): return graph[inverse_perm[x], inverse_perm[y]] == 1

    new_matrix = np.fromfunction(np.vectorize(edge_fn),
                                 (len(label), len(label)),
                                 dtype=np.int8)
    new_label = [label[inverse_perm[i]] for i in range(len(label))]
    return new_matrix, new_label


def is_isomorphic(graph1, graph2):
    """Exhaustively checks if 2 graphs are isomorphic."""
    matrix1, label1 = np.array(graph1[0]), graph1[1]
    matrix2, label2 = np.array(graph2[0]), graph2[1]
    assert np.shape(matrix1) == np.shape(matrix2)
    assert len(label1) == len(label2)

    vertices = np.shape(matrix1)[0]
    # Note: input and output in our constrained graphs always map to themselves
    # but this script does not enforce that.
    for perm in itertools.permutations(range(0, vertices)):
        pmatrix1, plabel1 = permute_graph(matrix1, label1, perm)
        if np.array_equal(pmatrix1, matrix2) and plabel1 == label2:
            return True

    return False


def compute_vertex_channels(output_channels, matrix):
    """
    Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Args:
        output_channels: output channel count.
        matrix: adjacency matrix for the module (pruned by model_spec).
    Returns:
        list of channel counts, in order of the vertices.
    """

    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = -1  # input channels can be inferred
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going backwards.
    # (num_vertices - 2) index skipped because it only connects to output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    _logger.debug('vertex_channels: %s', str(vertex_channels))

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels
