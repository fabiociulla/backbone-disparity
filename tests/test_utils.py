import numpy as np
import networkx as nx
import pytest
from backbone.utils import parse_input, validate_undirected, compute_degrees


def symmetric_matrix():
    return np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)


def test_parse_numpy():
    M, nodes = parse_input(symmetric_matrix())
    assert M.shape == (3, 3)
    assert nodes == [0, 1, 2]


def test_parse_networkx():
    G = nx.path_graph(4)
    M, nodes = parse_input(G)
    assert M.shape == (4, 4)
    assert set(nodes) == {0, 1, 2, 3}


def test_validate_symmetric():
    validate_undirected(symmetric_matrix())  # should not raise


def test_validate_asymmetric():
    bad = np.array([[0, 1], [0, 0]], dtype=float)
    with pytest.raises(ValueError):
        validate_undirected(bad)


def test_digraph_rejected():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    with pytest.raises(TypeError):
        parse_input(G)


def test_degrees():
    M = symmetric_matrix()
    k = compute_degrees(M)
    np.testing.assert_array_equal(k, [2, 2, 2])


def test_degree_isolated():
    M = np.array([[0, 0], [0, 0]], dtype=float)
    k = compute_degrees(M)
    np.testing.assert_array_equal(k, [0, 0])