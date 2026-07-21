import numpy as np
import pytest
from backbone.filter import apply_disparity_filter, matrix_to_graph
from backbone.utils import compute_degrees
import networkx as nx


def simple_matrix():
    """
    4-node weighted undirected graph:
      0-1 (w=10), 0-2 (w=1), 0-3 (w=1), 1-2 (w=1), 2-3 (w=10)
    Node 0: strong link to 1, weak to 2 and 3
    Node 2: strong link to 3, weak to 0 and 1
    """
    M = np.array([
        [0, 10,  1,  1],
        [10,  0,  1,  0],
        [1,   1,  0, 10],
        [1,   0, 10,  0],
    ], dtype=float)
    return M


def test_low_alpha_prunes_weak_edges():
    """At low alpha only the strongest edges survive."""
    M = simple_matrix()
    k = compute_degrees(M)
    degree1_mask = (k == 1)
    M_alpha = apply_disparity_filter(M, k, alpha=0.1, degree1_mask=degree1_mask)
    assert M_alpha[0, 2] == 0.0
    assert M_alpha[0, 3] == 0.0
    assert M_alpha[1, 2] == 0.0


def test_low_alpha_keeps_few_edges():
    """At very low alpha the backbone is a subset of the original edges."""
    M = simple_matrix()
    k = compute_degrees(M)
    degree1_mask = (k == 1)
    M_alpha = apply_disparity_filter(M, k, alpha=0.01, degree1_mask=degree1_mask)
    assert (M_alpha > 0).sum() <= (M > 0).sum()


def test_high_alpha_keeps_all_edges():
    """At alpha=1.0 no edge can ever be pruned."""
    M = simple_matrix()
    k = compute_degrees(M)
    degree1_mask = (k == 1)
    M_alpha = apply_disparity_filter(M, k, alpha=1.0, degree1_mask=degree1_mask)
    np.testing.assert_array_equal(M_alpha > 0, M > 0)


def test_degree1_edge_always_kept():
    """A degree-1 node's single edge must never be pruned at any alpha."""
    M = np.array([
        [0, 1, 0],
        [1, 0, 100],
        [0, 100, 0],
    ], dtype=float)
    k = compute_degrees(M)
    degree1_mask = (k == 1)
    # Even at the strictest alpha, degree-1 edge must survive
    M_alpha = apply_disparity_filter(M, k, alpha=0.0, degree1_mask=degree1_mask)
    assert M_alpha[0, 1] > 0
    assert M_alpha[1, 0] > 0


def test_symmetry_preserved():
    """Filtered matrix must remain symmetric."""
    M = simple_matrix()
    k = compute_degrees(M)
    degree1_mask = (k == 1)
    M_alpha = apply_disparity_filter(M, k, alpha=0.3, degree1_mask=degree1_mask)
    np.testing.assert_array_almost_equal(M_alpha, M_alpha.T)


def test_matrix_to_graph_isolates():
    """Degree-0 nodes must appear in the backbone graph as isolates."""
    M = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=float)
    nodes = [0, 1, 2]
    G = matrix_to_graph(M, nodes)
    assert 2 in G.nodes()
    assert G.degree(2) == 0


def test_matrix_to_graph_node_attributes():
    """Node attributes from the original graph must be copied."""
    M = np.array([[0, 1], [1, 0]], dtype=float)
    nodes = ["a", "b"]
    G_orig = nx.Graph()
    G_orig.add_node("a", color="red")
    G_orig.add_node("b", color="blue")
    G_orig.add_edge("a", "b", weight=1)
    G = matrix_to_graph(M, nodes, G_original=G_orig)
    assert G.nodes["a"]["color"] == "red"
    assert G.nodes["b"]["color"] == "blue"