import pytest
import networkx as nx
from backbone.community import compute_louvain, compute_infomap


def two_cliques():
    """Two well-separated cliques — should yield high modularity."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])        # clique A
    G.add_edges_from([(3, 4), (3, 5), (4, 5)])        # clique B
    G.add_edge(2, 3, weight=0.01)                      # weak bridge
    return G


def test_louvain_returns_partition_and_score(two_cliques=two_cliques()):
    partition, score = compute_louvain(two_cliques)
    assert isinstance(partition, dict)
    assert set(partition.keys()) == set(two_cliques.nodes())
    assert isinstance(score, float)


def test_louvain_high_modularity(two_cliques=two_cliques()):
    _, score = compute_louvain(two_cliques)
    assert score > 0.3, f"Expected high modularity for two cliques, got {score:.4f}"


def test_louvain_empty_graph():
    G = nx.Graph()
    partition, score = compute_louvain(G)
    assert partition == {}
    assert score == 0.0


def test_infomap_returns_partition_and_score(two_cliques=two_cliques()):
    partition, score = compute_infomap(two_cliques)
    assert isinstance(partition, dict)
    assert set(partition.keys()) == set(two_cliques.nodes())
    assert isinstance(score, float)


def test_infomap_score_is_negative_codelength(two_cliques=two_cliques()):
    """Score must be ≤ 0 since it's –codelength and codelength ≥ 0."""
    _, score = compute_infomap(two_cliques)
    assert score <= 0.0


def test_infomap_empty_graph():
    G = nx.Graph()
    partition, score = compute_infomap(G)
    assert partition == {}
    assert score == 0.0