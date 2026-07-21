"""Core disparity filter."""

from __future__ import annotations

from copy import copy

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize


def apply_disparity_filter(
    M: np.ndarray,
    k: np.ndarray,
    alpha: float,
    degree1_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply the disparity filter to adjacency matrix M.

    An edge (i,j) is KEPT if at least one endpoint finds it significant,
    i.e. its p-value (1 - p_ij)^(k_i - 1) is BELOW alpha.

    Equivalently, an edge is PRUNED only when BOTH endpoints find it
    INsignificant, i.e. the p-value >= alpha from both sides.

    Parameters
    ----------
    M            : float adjacency matrix (symmetric)
    k            : degree vector computed on the *original* graph
    alpha        : significance threshold (like a p-value cutoff)
                   low alpha  → few edges kept (strict)
                   high alpha → many edges kept (lenient)
    degree1_mask : boolean array — True for nodes whose degree in the
                   *original* graph is exactly 1 (their edge is always kept)

    Returns
    -------
    M_alpha : filtered adjacency matrix
    """
    # Row-normalise (L1) to get relative weights p_ij = w_ij / strength_i
    M_norm = normalize(M, axis=1, norm="l1")

    M_alpha = copy(M)

    # p-value for each edge from the perspective of each endpoint:
    # pval_ij = (1 - p_ij)^(k_i - 1)
    # An edge is insignificant for node i when pval_ij >= alpha
    # An edge is PRUNED when it is insignificant from BOTH sides (union rule)
    with np.errstate(divide="ignore", invalid="ignore"):
        pval = (1.0 - M_norm) ** (k[:, None] - 1)

    # idx_to_zero is True where the edge is insignificant from that side
    idx_to_zero = pval >= alpha                 # insignificant from row side
    idx_to_zero = idx_to_zero & idx_to_zero.T  # insignificant from BOTH sides → prune

    # Override: degree-1 nodes (from original graph) always keep their edge
    protected = degree1_mask[:, None] | degree1_mask[None, :]
    idx_to_zero = idx_to_zero & ~protected

    M_alpha[idx_to_zero] = 0.0
    return M_alpha


def matrix_to_graph(
    M_alpha: np.ndarray,
    nodes: list,
    G_original: nx.Graph | None = None,
) -> nx.Graph:
    """
    Build a nx.Graph from a filtered adjacency matrix.

    - Preserves node attributes from G_original when available.
    - Degree-0 nodes in M_alpha are added back as isolates (spec 5).
    """
    G = nx.from_numpy_array(M_alpha)

    # Relabel integer indices back to original node labels
    mapping = {i: node for i, node in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)

    # Copy node attributes from original graph
    if G_original is not None:
        for node in G.nodes():
            if node in G_original.nodes:
                G.nodes[node].update(G_original.nodes[node])

    # Ensure all original nodes are present (degree-0 isolates)
    for node in nodes:
        if node not in G:
            G.add_node(node)
            if G_original is not None and node in G_original.nodes:
                G.nodes[node].update(G_original.nodes[node])

    return G