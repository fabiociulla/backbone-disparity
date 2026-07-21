import numpy as np
import networkx as nx
import pytest
from backbone import compute_backbone, BackboneResult


def weighted_graph():
    """
    Two dense communities connected by a single weak bridge.
    The disparity filter should cleanly separate them.
    """
    G = nx.Graph()
    # Community A: nodes 0-4
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(i, j, weight=10.0)
    # Community B: nodes 5-9
    for i in range(5, 10):
        for j in range(i + 1, 10):
            G.add_edge(i, j, weight=10.0)
    # Weak bridge
    G.add_edge(4, 5, weight=0.1)
    return G


def test_single_alpha_returns_result():
    G = weighted_graph()
    result = compute_backbone(G, alpha=0.2, plot=False)
    assert isinstance(result, BackboneResult)
    assert isinstance(result.backbone, nx.Graph)
    assert isinstance(result.communities, dict)
    assert 0.0 <= result.non_orphan_ratio <= 1.0    # ← was gc_ratio


def test_single_alpha_combined_score_consistent():
    G = weighted_graph()
    result = compute_backbone(G, alpha=0.2, plot=False)
    assert result.combined_score == pytest.approx(result.score * result.non_orphan_ratio)  # ← was gc_ratio
    

def test_range_alpha_returns_best():
    G = weighted_graph()
    alphas = [0.05, 0.1, 0.2, 0.4, 0.6]
    result = compute_backbone(G, alpha=alphas, plot=False)
    assert result.alpha in alphas
    # alphas_data must have one record per alpha
    assert len(result.alphas_data) == len(alphas)


def test_range_alpha_best_is_max_combined():
    G = weighted_graph()
    alphas = np.linspace(0.05, 0.8, 10).tolist()
    result = compute_backbone(G, alpha=alphas, plot=False)
    all_combined = [r.combined_score for r in result.alphas_data]
    assert result.combined_score == pytest.approx(max(all_combined))


def test_auto_search_returns_result():
    G = weighted_graph()
    result = compute_backbone(G, alpha=None, plot=False)
    assert isinstance(result, BackboneResult)
    assert 0.01 <= result.alpha <= 0.99


def test_matrix_input():
    M = nx.to_numpy_array(weighted_graph())
    result = compute_backbone(M, alpha=0.2, plot=False)
    assert isinstance(result, BackboneResult)


def test_asymmetric_matrix_rejected():
    M = np.array([[0, 1], [0, 0]], dtype=float)
    with pytest.raises(ValueError, match="symmetric"):
        compute_backbone(M, alpha=0.1, plot=False)


def test_digraph_rejected():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    with pytest.raises(TypeError):
        compute_backbone(G, alpha=0.1, plot=False)


def test_isolates_preserved():
    G = weighted_graph()
    G.add_node(99)   # isolated node
    result = compute_backbone(G, alpha=0.2, plot=False)
    assert 99 in result.backbone.nodes()


def test_degree1_edge_retained():
    """Pendant node's edge must survive even at very high alpha."""
    G = nx.Graph()
    G.add_edges_from([(0, 1, {"weight": 1}),
                      (1, 2, {"weight": 100}),
                      (1, 3, {"weight": 100})])
    result = compute_backbone(G, alpha=0.99, plot=False)
    assert result.backbone.has_edge(0, 1) or result.backbone.has_edge(1, 0)