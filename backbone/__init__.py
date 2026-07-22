"""
backbone
========
Public API for computing the disparity-filter backbone network.

Usage
-----
Single alpha (user-supplied)
    >>> results = compute_backbone(G, alpha=0.05)

Range of alphas
    >>> results = compute_backbone(
    ...     G, alpha=[0.01, 0.05, 0.1], 
    ...     rationales=["non orphan ratio", "giant component ratio"]
    ... )

Auto-search (golden-section, no alpha given)
    >>> results = compute_backbone(G, rationales=["non orphan ratio", "giant component ratio"])
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
    multiplier_value: float           
    combined_score: float
    rationale: str


@dataclass
class BackboneResult:
    """Unified return type for a specific rationale."""
    rationale: str                        
    alpha: float                          
    backbone: nx.Graph                    
    communities: dict                     
    score: float                          
    multiplier_value: float               
    combined_score: float                 
    method: str                           
    alphas_data: list[AlphaRecord] = field(default_factory=list)  


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_backbone(
    data: np.ndarray | nx.Graph,
    alpha: float | list[float] | np.ndarray | None = None,
    method: str = "louvain",
    tol: float = 1e-3,
    plot: bool = True,
    save_plot: str | None = None,
    log_scale: bool = True,
    rationales: list[str] | None = None,
) -> dict[str, BackboneResult]:
    """
    Compute the disparity-filter backbone and community structure.

    Returns
    -------
    Dictionary mapping rationale strings to their corresponding BackboneResult.
    """
    if rationales is None:
        rationales = ["non orphan ratio"]

    # ── 1. Parse & validate ───────────────────────────────────────────────────
    M, nodes = parse_input(data)
    validate_undirected(M)

    G_original = (
        data if isinstance(data, nx.Graph)
        else nx.from_numpy_array(M)
    )

    # ── 2. Pre-compute degrees from the ORIGINAL graph ────────────────────────
    k = compute_degrees(M)                          
    degree1_mask = (k == 1)                         

    # ── 3. Build the per-alpha objective ──────────────────────────────────────
    def _objective(a: float) -> tuple[nx.Graph, dict, float, dict[str, float], dict[str, float]]:
        M_alpha = apply_disparity_filter(M, k, a, degree1_mask)
        G_b = matrix_to_graph(M_alpha, nodes, G_original)
        partition, score, multipliers_dict, combined_dict = evaluate_backbone(
            G_b, method=method, rationales=rationales
        )
        return G_b, partition, score, multipliers_dict, combined_dict

    results_dict = {}

    # ── 4a. Single float ──────────────────────────────────────────────────────
    if isinstance(alpha, (int, float)):
        alpha = float(alpha)
        G_b, partition, score, multipliers_dict, combined_dict = _objective(alpha)
        
        for rat in rationales:
            results_dict[rat] = BackboneResult(
                rationale=rat,
                alpha=alpha,
                backbone=G_b,
                communities=partition,
                score=score,
                multiplier_value=multipliers_dict[rat],
                combined_score=combined_dict[rat],
                method=method,
            )
        return results_dict

    # ── 4b. Range of alphas ───────────────────────────────────────────────────
    if isinstance(alpha, (list, np.ndarray)):
        alphas = list(alpha)
        
        # Track states independently per rationale
        best_states = {
            rat: {
                "best_combined": -np.inf, "best_alpha_val": alphas[0],
                "best_G": None, "best_partition": {}, "best_score": 0.0,
                "best_mult": 0.0, "records": [], "comm_scores": [],
                "mult_values": [], "comb_scores": []
            }
            for rat in rationales
        }

        for a in alphas:
            G_b, partition, score, mults_dict, combs_dict = _objective(a)
            
            for rat in rationales:
                state = best_states[rat]
                comb = combs_dict[rat]
                mult = mults_dict[rat]
                
                state["records"].append(AlphaRecord(
                    alpha=a, community_score=score, multiplier_value=mult,
                    combined_score=comb, rationale=rat
                ))
                state["comm_scores"].append(score)
                state["mult_values"].append(mult)
                state["comb_scores"].append(comb)

                if comb > state["best_combined"]:
                    state["best_combined"] = comb
                    state["best_alpha_val"] = a
                    state["best_G"] = G_b
                    state["best_partition"] = partition
                    state["best_score"] = score
                    state["best_mult"] = mult

        for rat in rationales:
            state = best_states[rat]
            results_dict[rat] = BackboneResult(
                rationale=rat, alpha=state["best_alpha_val"],
                backbone=state["best_G"], communities=state["best_partition"],
                score=state["best_score"], multiplier_value=state["best_mult"],
                combined_score=state["best_combined"], method=method,
                alphas_data=state["records"],
            )

        if plot:
            # Reorganize payload for the plotting function
            first_rat = rationales[0]
            fig = plot_metrics(
                alphas=alphas,
                community_scores=best_states[first_rat]["comm_scores"],
                multipliers_dict={r: s["mult_values"] for r, s in best_states.items()},
                combined_scores_dict={r: s["comb_scores"] for r, s in best_states.items()},
                best_alphas_dict={r: s["best_alpha_val"] for r, s in best_states.items()},
                method=method,
                save_path=save_plot,
                log_scale=log_scale,
            )
            fig.show()

        return results_dict

    # ── 4c. None → golden-section search ─────────────────────────────────────
    if alpha is None:
        
        # Golden-section search strictly evaluates objectives uniquely per rationale
        for rat in rationales:
            
            def _score_only_for_rationale(a: float) -> float:
                _, _, _, _, combs = _objective(a)
                return combs[rat]

            alpha_opt, _ = golden_section_search(
                objective=_score_only_for_rationale,
                a=0.01, b=0.99, tol=tol,
            )
            
            G_b, partition, score, mults, combs = _objective(alpha_opt)
            
            results_dict[rat] = BackboneResult(
                rationale=rat, alpha=alpha_opt, backbone=G_b,
                communities=partition, score=score,
                multiplier_value=mults[rat], combined_score=combs[rat],
                method=method,
            )

        if plot:
            alphas_plot = np.linspace(0.01, 0.99, 60).tolist()
            comm_s = []
            mults_s = {r: [] for r in rationales}
            combs_s = {r: [] for r in rationales}
            
            for a in alphas_plot:
                _, _, s, mults_dict, combs_dict = _objective(a)
                comm_s.append(s)
                for rat in rationales:
                    mults_s[rat].append(mults_dict[rat])
                    combs_s[rat].append(combs_dict[rat])

            fig = plot_metrics(
                alphas=alphas_plot,
                community_scores=comm_s,
                multipliers_dict=mults_s,
                combined_scores_dict=combs_s,
                best_alphas_dict={r: res.alpha for r, res in results_dict.items()},
                method=method,
                save_path=save_plot,
                log_scale=log_scale,
            )
            fig.show()

        return results_dict

    raise TypeError(
        f"`alpha` must be a float, list/ndarray, or None. Got {type(alpha)}."
    )