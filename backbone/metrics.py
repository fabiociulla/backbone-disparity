"""Scoring: multipliers and combined scores."""

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


def giant_component_ratio(G: nx.Graph) -> float:
    """
    Fraction of nodes belonging to the largest connected component.
    """
    if G.number_of_nodes() == 0:
        return 0.0
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / G.number_of_nodes()


def evaluate_backbone(
    G: nx.Graph,
    method: str = "louvain",
    rationales: list[str] | None = None,
) -> tuple[dict, float, dict[str, float], dict[str, float]]:
    """
    Run community detection on G and return all scoring information.

    Parameters
    ----------
    G          : backbone graph
    method     : 'louvain' or 'infomap'
    rationales : list of rationale names to calculate multipliers for

    Returns
    -------
    partition   : dict[node → community_id]
    score       : community metric (Q or –codelength)
    multipliers : dict[rationale → multiplier_value]
    combined    : dict[rationale → score × multiplier_value]
    """
    if rationales is None:
        rationales = ["non orphan ratio"]

    method = method.lower()
    if method == "louvain":
        partition, score = compute_louvain(G)
    elif method == "infomap":
        partition, score = compute_infomap(G)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'louvain' or 'infomap'.")

    multipliers = {}
    combined = {}
    
    # Easily extensible to new rationales in the future
    for rat in rationales:
        if rat == "non orphan ratio":
            val = non_orphan_ratio(G)
        elif rat == "giant component ratio":
            val = giant_component_ratio(G)
        else:
            raise ValueError(f"Unknown rationale metric '{rat}'.")
        
        multipliers[rat] = val
        combined[rat] = score * val

    return partition, score, multipliers, combined