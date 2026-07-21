"""
backbone
========
Public API for computing the disparity-filter backbone network.

Usage
-----
Single alpha (user-supplied)
    >>> result = compute_backbone(G, alpha=0.05)

Range of alphas
    >>> result = compute_backbone(G, alpha=[0.01, 0.05, 0.1, 0.2, 0.5])

Auto-search (golden-section, no alpha given)
    >>> result = compute_backbone(G)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import networkx as nx

from .utils import parse_input, validate_undirected, compute_degrees
from .filter import apply_disparity_filter, matrix_to_graph
from .metrics import evaluate_backbone
from .optimize import golden_section_search
from .plotting import plot_metrics, plot_network

# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AlphaRecord:
    """Scalar metrics for a single alpha value (range mode)."""
    alpha: float
    community_score: float
    non_orphan_ratio: float           # ← renamed from gc_ratio
    combined_score: float


@dataclass
class BackboneResult:
    """Unified return type for all invocation modes."""
    alpha: float                          # best / only / searched alpha
    backbone: nx.Graph                    # backbone graph at best alpha
    communities: dict                     # node → community_id
    score: float                          # community metric at best alpha
    non_orphan_ratio: float           # ← renamed from gc_ratio
    combined_score: float                 # score × gc_ratio
    method: str                           # 'louvain' or 'infomap'
    alphas_data: list[AlphaRecord] = field(default_factory=list)  # range mode only


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_backbone(
    data: np.ndarray | nx.Graph,
    alpha: float | list[float] | np.ndarray | None = None,
    method: str = "louvain",
    tol: float = 1e-3,
    plot: bool = True,
    save_plot: str | None = None,
    log_scale: bool = True,          # ← new parameter
) -> BackboneResult:

    """
    Compute the disparity-filter backbone and community structure.

    Parameters
    ----------
    data   : np.ndarray, scipy sparse matrix, or nx.Graph (undirected)
    alpha  : float         → single run at that value
             list/ndarray  → sweep over all values, return best
             None          → golden-section search for optimal alpha
    method : 'louvain' or 'infomap'
    tol    : tolerance for golden-section search (ignored in other modes)
    plot   : produce diagnostic plot (range and search modes only)
    save_plot : file path to save the plot image (optional)

    Returns
    -------
    BackboneResult dataclass
    """
    # ── 1. Parse & validate ───────────────────────────────────────────────────
    M, nodes = parse_input(data)
    validate_undirected(M)

    G_original = (
        data if isinstance(data, nx.Graph)
        else nx.from_numpy_array(M)
    )

    # ── 2. Pre-compute degrees from the ORIGINAL graph (spec 2 & 3) ───────────
    k = compute_degrees(M)                          # shape (N,)
    degree1_mask = (k == 1)                         # nodes with degree 1 in original

    # ── 3. Build the per-alpha objective (backbone → evaluate) ────────────────
    def _objective(a: float) -> tuple[nx.Graph, dict, float, float, float]:
        """Returns (G_backbone, partition, score, non_orphan_ratio, combined)."""
        M_alpha = apply_disparity_filter(M, k, a, degree1_mask)
        G_b = matrix_to_graph(M_alpha, nodes, G_original)
        partition, score, nor, comb = evaluate_backbone(G_b, method=method)
        return G_b, partition, score, nor, comb

    def _score_only(a: float) -> float:
        """Scalar combined score for optimisation."""
        _, _, _, _, comb = _objective(a)
        return comb

    # ── 4. Dispatch on alpha type ─────────────────────────────────────────────

    # ── 4a. Single float ──────────────────────────────────────────────────────
    if isinstance(alpha, (int, float)):
        alpha = float(alpha)
        G_b, partition, score, nor, comb = _objective(alpha)   # ← was gc
        return BackboneResult(
            alpha=alpha,
            backbone=G_b,
            communities=partition,
            score=score,
            non_orphan_ratio=nor,
            combined_score=comb,
            method=method,
        )
    # ── 4b. Range of alphas ───────────────────────────────────────────────────
    if isinstance(alpha, (list, np.ndarray)):
        alphas = list(alpha)
        records: list[AlphaRecord] = []

        best_combined  = -np.inf
        best_alpha_val = alphas[0]
        best_G         = None
        best_partition = {}
        best_score     = 0.0
        best_nor       = 0.0           # ← renamed

        comm_scores, nor_values, comb_scores = [], [], []

        for a in alphas:
            G_b, partition, score, nor, comb = _objective(a)
            records.append(AlphaRecord(
                alpha=a,
                community_score=score,
                non_orphan_ratio=nor,      # ← renamed
                combined_score=comb,
            ))
            comm_scores.append(score)
            nor_values.append(nor)         # ← renamed
            comb_scores.append(comb)

            if comb > best_combined:
                best_combined  = comb
                best_alpha_val = a
                best_G         = G_b
                best_partition = partition
                best_score     = score
                best_nor       = nor       # ← renamed

        if plot:
            fig = plot_metrics(
                alphas=alphas,
                community_scores=comm_scores,
                non_orphan_ratios=nor_values,      # plotting.py still accepts this arg
                combined_scores=comb_scores,
                method=method,
                best_alpha=best_alpha_val,
                save_path=save_plot,
                log_scale=log_scale,
            )
            fig.show()

        return BackboneResult(
            alpha=best_alpha_val,
            backbone=best_G,
            communities=best_partition,
            score=best_score,
            non_orphan_ratio=best_nor,     # ← renamed
            combined_score=best_combined,
            method=method,
            alphas_data=records,
        )

    # ── 4c. None → golden-section search ─────────────────────────────────────
    if alpha is None:
        alpha_opt, comb_opt = golden_section_search(
            objective=_score_only,
            a=0.01,
            b=0.99,
            tol=tol,
        )
        G_b, partition, score, nor, comb = _objective(alpha_opt)

        if plot:
            alphas_plot = np.linspace(0.01, 0.99, 60).tolist()
            comm_s, nor_s, comb_s = [], [], []
            for a in alphas_plot:
                _, _, s, n, c = _objective(a)
                comm_s.append(s)
                nor_s.append(n)
                comb_s.append(c)

            fig = plot_metrics(
                alphas=alphas_plot,
                community_scores=comm_s,
                non_orphan_ratios=nor_s,           # plotting.py still accepts this arg
                combined_scores=comb_s,
                method=method,
                best_alpha=alpha_opt,
                save_path=save_plot,
                log_scale=log_scale,
            )
            fig.show()

        return BackboneResult(
            alpha=alpha_opt,
            backbone=G_b,
            communities=partition,
            score=score,
            non_orphan_ratio=nor,          # ← renamed
            combined_score=comb,
            method=method,
        )

    raise TypeError(
        f"`alpha` must be a float, list/ndarray, or None. Got {type(alpha)}."
    )