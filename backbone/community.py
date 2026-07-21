"""Community detection wrappers: Louvain and Infomap."""

from __future__ import annotations

import numpy as np
import networkx as nx

# ── Louvain ──────────────────────────────────────────────────────────────────
try:
    import community as community_louvain  # python-louvain
    _LOUVAIN_AVAILABLE = True
except ImportError:
    _LOUVAIN_AVAILABLE = False

# ── Infomap ───────────────────────────────────────────────────────────────────
try:
    import infomap as _infomap_module
    _INFOMAP_AVAILABLE = True
except ImportError:
    _INFOMAP_AVAILABLE = False

_DEFAULT_SEED = 42


def compute_louvain(
    G: nx.Graph,
    n_runs: int = 10,
    seed: int = _DEFAULT_SEED,
) -> tuple[dict, float]:
    """
    Run the Louvain algorithm n_runs times and return the partition
    and modularity of the best run.

    Each run uses a deterministic but different seed derived from
    the base seed, so results are fully reproducible.

    Parameters
    ----------
    G      : input graph
    n_runs : number of independent runs (higher = more stable, slower)
    seed   : base random seed (default 42)

    Returns
    -------
    partition : dict[node → community_id]
    score     : float  best modularity Q across all runs
    """
    if not _LOUVAIN_AVAILABLE:
        raise ImportError(
            "python-louvain is not installed. Run: pip install python-louvain"
        )

    if G.number_of_nodes() == 0:
        return {}, 0.0

    if G.number_of_edges() == 0:
        partition = {node: i for i, node in enumerate(G.nodes())}
        return partition, 0.0

    best_partition = None
    best_score = -np.inf

    for i in range(n_runs):
        # Each run gets a deterministic unique seed derived from the base seed
        run_seed = seed + i
        partition = community_louvain.best_partition(
            G, weight="weight", random_state=run_seed
        )
        try:
            score = community_louvain.modularity(partition, G, weight="weight")
        except ValueError:
            score = 0.0

        if score > best_score:
            best_score = score
            best_partition = partition

    return best_partition, best_score


def compute_infomap(
    G: nx.Graph,
    seed: int = _DEFAULT_SEED,
) -> tuple[dict, float]:
    """
    Run the Infomap algorithm on G.

    Parameters
    ----------
    G    : input graph
    seed : random seed (default 42)

    Returns
    -------
    partition : dict[node → community_id]
    score     : float  –codelength  (higher = better, consistent with Louvain)
    """
    if not _INFOMAP_AVAILABLE:
        raise ImportError(
            "infomap is not installed. Run: pip install infomap"
        )

    if G.number_of_nodes() == 0:
        return {}, 0.0

    if G.number_of_edges() == 0:
        partition = {node: i for i, node in enumerate(G.nodes())}
        return partition, 0.0

    # Pass seed directly to Infomap for reproducibility
    im = _infomap_module.Infomap(silent=True, seed=seed)

    node_list = list(G.nodes())
    label_to_int = {n: i for i, n in enumerate(node_list)}

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        im.add_link(label_to_int[u], label_to_int[v], w)

    im.run()

    partition = {
        node_list[node.node_id]: node.module_id
        for node in im.nodes
    }

    codelength = im.codelength
    score = -codelength
    return partition, score