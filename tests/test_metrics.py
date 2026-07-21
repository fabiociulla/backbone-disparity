import networkx as nx
import pytest
from backbone.metrics import non_orphan_ratio, combined_score, evaluate_backbone


def connected_graph():
    return nx.path_graph(5)


def disconnected_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])   # component of size 3
    G.add_edge(3, 4)                      # component of size 2
    return G


def test_non_orphan_ratio_fully_connected():
    # all nodes have degree > 0 in a path graph
    assert non_orphan_ratio(connected_graph()) == pytest.approx(1.0)


def test_non_orphan_ratio_disconnected():
    # all 5 nodes still have degree > 0 even if disconnected
    assert non_orphan_ratio(disconnected_graph()) == pytest.approx(1.0)


def test_non_orphan_ratio_with_isolates():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])   # 3 connected nodes
    G.add_nodes_from([3, 4])             # 2 isolates
    assert non_orphan_ratio(G) == pytest.approx(3 / 5)


def test_non_orphan_ratio_all_isolates():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    assert non_orphan_ratio(G) == pytest.approx(0.0)


def test_non_orphan_ratio_empty():
    assert non_orphan_ratio(nx.Graph()) == 0.0


def test_combined_score():
    assert combined_score(0.8, 0.75) == pytest.approx(0.6)


def test_combined_score_zero_ratio():
    assert combined_score(0.9, 0.0) == pytest.approx(0.0)


def test_evaluate_backbone_louvain():
    G = connected_graph()
    partition, score, nor, comb = evaluate_backbone(G, method="louvain")
    assert isinstance(partition, dict)
    assert 0.0 <= nor <= 1.0
    assert comb == pytest.approx(score * nor)


def test_evaluate_backbone_infomap():
    G = connected_graph()
    partition, score, nor, comb = evaluate_backbone(G, method="infomap")
    assert isinstance(partition, dict)
    assert comb == pytest.approx(score * nor)


def test_evaluate_backbone_invalid_method():
    with pytest.raises(ValueError, match="Unknown method"):
        evaluate_backbone(connected_graph(), method="kmeans")