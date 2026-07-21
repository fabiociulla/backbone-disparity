"""Scoring: non-orphan ratio and combined score."""

from __future__ import annotations

import networkx as nx

from .community import compute_louvain, compute_infomap


def non_orphan_ratio(G: nx.Graph) -> float:
    """
    Fraction of nodes with degree > 0 in the backbone graph G.
    Isolates (degree = 0) are counted as 'orphans' and penalise the score.
    Returns 0.0 for an empty graph.
    """
    if G.number_of_nodes() == 0:
        return 0.0
    n_non_orphan = sum(1 for n in G.nodes() if G.degree(n) > 0)
    return n_non_orphan / G.number_of_nodes()


def combined_score(community_score: float, ratio: float) -> float:
    """community_score × non-orphan ratio."""
    return community_score * ratio


def evaluate_backbone(
    G: nx.Graph,
    method: str = "louvain",
) -> tuple[dict, float, float, float]:
    """
    Run community detection on G and return all scoring information.

    Parameters
    ----------
    G      : backbone graph
    method : 'louvain' or 'infomap'

    Returns
    -------
    partition      : dict[node → community_id]
    score          : community metric (Q or –codelength)
    non_orphan_r   : fraction of nodes with degree > 0 in G
    combined       : score × non_orphan_ratio
    """
    method = method.lower()
    if method == "louvain":
        partition, score = compute_louvain(G)
    elif method == "infomap":
        partition, score = compute_infomap(G)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'louvain' or 'infomap'.")

    nor = non_orphan_ratio(G)
    comb = combined_score(score, nor)
    return partition, score, nor, comb