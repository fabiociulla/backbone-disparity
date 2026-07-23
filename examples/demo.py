"""
Quick end-to-end demonstration of the backbone-disparity library.

Covers:
  1. Single alpha
  2. Range of alphas
  3. Automatic alpha search (golden-section)
  4. Matrix input
  5. Infomap instead of Louvain
"""

import sys
import os

# Add the parent directory (repository root) to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from backbone import compute_backbone


# ── Helper: build a toy weighted graph ───────────────────────────────────────

def build_demo_graph(seed: int = 42) -> nx.Graph:
    """
    Two dense weighted communities (nodes 0–9 and 10–19),
    connected by two weak inter-community bridges.
    Node 20 is a pendant (degree-1) node attached to node 0.
    Node 21 is a complete isolate (degree 0).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    # Community A (nodes 0–9): strong internal edges
    for i in range(10):
        for j in range(i + 1, 10):
            G.add_edge(i, j, weight=float(rng.integers(5, 15)))

    # Community B (nodes 10–19): strong internal edges
    for i in range(10, 20):
        for j in range(i + 1, 20):
            G.add_edge(i, j, weight=float(rng.integers(5, 15)))

    # Weak inter-community bridges
    G.add_edge(9,  10, weight=0.5)
    G.add_edge(8,  11, weight=0.3)

    # Pendant node (degree 1) — its edge must always be retained
    G.add_edge(20, 0, weight=1.0)

    # Isolate node (degree 0) — must always be retained as isolate
    G.add_node(21)

    return G


# ── 1. Single alpha ───────────────────────────────────────────────────────────

def demo_single_alpha(G: nx.Graph) -> None:
    print("\n" + "=" * 60)
    print("DEMO 1 — Single alpha (α = 0.15, Louvain)")
    print("=" * 60)

    results = compute_backbone(G, alpha=0.15, method="louvain", plot=False)
    result = results["non orphan ratio"]

    print(f"  Original graph : {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"  Backbone       : {result.backbone.number_of_nodes()} nodes, "
          f"{result.backbone.number_of_edges()} edges")
    print(f"  Alpha          : {result.alpha:.4f}")
    print(f"  Modularity Q   : {result.score:.4f}")
    print(f"  Multiplier     : {result.multiplier_value:.4f}")
    print(f"  Combined score : {result.combined_score:.4f}")
    print(f"  # communities  : {len(set(result.communities.values()))}")

    # Check special cases
    assert 20 in result.backbone.nodes(), "Pendant node 20 must be in backbone"
    assert 21 in result.backbone.nodes(), "Isolate node 21 must be in backbone"
    assert result.backbone.has_edge(20, 0) or result.backbone.has_edge(0, 20), \
        "Pendant edge (20-0) must always be retained"
    print("  ✓ Pendant node edge retained")
    print("  ✓ Isolate node retained")


# ── 2. Range of alphas ────────────────────────────────────────────────────────

def demo_range_alpha(G: nx.Graph) -> None:
    print("\n" + "=" * 60)
    print("DEMO 2 — Range of alphas (Louvain)")
    print("=" * 60)

    alphas = np.linspace(0.01, 0.95, 40).tolist()

    results = compute_backbone(
        G,
        alpha=alphas,
        method="louvain",
        plot=True,
        save_plot="backbone_metrics_range.png",
    )
    result = results["non orphan ratio"]

    print(f"  Swept {len(alphas)} alpha values from "
          f"{alphas[0]:.2f} to {alphas[-1]:.2f}")
    print(f"  Best alpha     : {result.alpha:.4f}")
    print(f"  Modularity Q   : {result.score:.4f}")
    print(f"  Multiplier     : {result.multiplier_value:.4f}")
    print(f"  Combined score : {result.combined_score:.4f}")
    print(f"  # communities  : {len(set(result.communities.values()))}")
    print(f"  Backbone       : {result.backbone.number_of_nodes()} nodes, "
          f"{result.backbone.number_of_edges()} edges")
    print(f"  Diagnostic plot saved to: backbone_metrics_range.png")

    # Print a compact sweep table
    print("\n  α        Q       Mult    Combined")
    print("  " + "-" * 38)
    for rec in result.alphas_data:
        marker = " ← best" if abs(rec.alpha - result.alpha) < 1e-9 else ""
        print(f"  {rec.alpha:.3f}   {rec.community_score:+.4f}  "
              f"{rec.multiplier_value:.4f}  {rec.combined_score:.4f}{marker}")


# ── 3. Automatic alpha search ─────────────────────────────────────────────────

def demo_auto_search(G: nx.Graph) -> None:
    print("\n" + "=" * 60)
    print("DEMO 3 — Automatic alpha search via golden-section (Louvain)")
    print("=" * 60)

    results = compute_backbone(
        G,
        alpha=None,       # triggers golden-section search
        method="louvain",
        tol=1e-3,
        plot=True,
        save_plot="backbone_metrics_search.png",
    )
    result = results["non orphan ratio"]

    print(f"  Optimal alpha  : {result.alpha:.6f}")
    print(f"  Modularity Q   : {result.score:.4f}")
    print(f"  Multiplier     : {result.multiplier_value:.4f}")
    print(f"  Combined score : {result.combined_score:.4f}")
    print(f"  # communities  : {len(set(result.communities.values()))}")
    print(f"  Backbone       : {result.backbone.number_of_nodes()} nodes, "
          f"{result.backbone.number_of_edges()} edges")
    print(f"  Diagnostic plot saved to: backbone_metrics_search.png")


# ── 4. Matrix input ───────────────────────────────────────────────────────────

def demo_matrix_input(G: nx.Graph) -> None:
    print("\n" + "=" * 60)
    print("DEMO 4 — NumPy matrix input (α = 0.2, Louvain)")
    print("=" * 60)

    M = nx.to_numpy_array(G, weight="weight")
    print(f"  Matrix shape   : {M.shape}")
    print(f"  Symmetric      : {np.allclose(M, M.T)}")

    results = compute_backbone(M, alpha=0.2, method="louvain", plot=False)
    result = results["non orphan ratio"]

    print(f"  Backbone       : {result.backbone.number_of_nodes()} nodes, "
          f"{result.backbone.number_of_edges()} edges")
    print(f"  Modularity Q   : {result.score:.4f}")
    print(f"  Combined score : {result.combined_score:.4f}")


# ── 5. Infomap ────────────────────────────────────────────────────────────────

def demo_infomap(G: nx.Graph) -> None:
    print("\n" + "=" * 60)
    print("DEMO 5 — Infomap community detection (α = 0.15)")
    print("=" * 60)

    results = compute_backbone(G, alpha=0.15, method="infomap", plot=False)
    result = results["non orphan ratio"]

    print(f"  Backbone       : {result.backbone.number_of_nodes()} nodes, "
          f"{result.backbone.number_of_edges()} edges")
    print(f"  −Codelength    : {result.score:.4f}")
    print(f"  Multiplier     : {result.multiplier_value:.4f}")
    print(f"  Combined score : {result.combined_score:.4f}")
    print(f"  # communities  : {len(set(result.communities.values()))}")


# ── 6. Visualise the backbone ─────────────────────────────────────────────────

def demo_visualise(G: nx.Graph) -> None:
    print("\n" + "=" * 60)
    print("DEMO 6 — Visualisation of original vs backbone")
    print("=" * 60)

    results = compute_backbone(G, alpha=0.15, method="louvain", plot=False)
    result = results["non orphan ratio"]
    B = result.backbone

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Shared layout based on original graph
    pos = nx.spring_layout(G, seed=42)
    node_colors = [result.communities.get(n, 0) for n in G.nodes()]

    # Original graph
    axes[0].set_title(
        f"Original graph\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        fontsize=12,
    )
    nx.draw_networkx(
        G, pos=pos, ax=axes[0],
        node_color=node_colors, cmap="tab10",
        node_size=200, font_size=7,
        width=[G[u][v].get("weight", 1) / 15 for u, v in G.edges()],
        edge_color="gray", alpha=0.7,
    )

    # Backbone
    axes[1].set_title(
        f"Backbone (α=0.15)\n{B.number_of_nodes()} nodes, {B.number_of_edges()} edges",
        fontsize=12,
    )
    backbone_node_colors = [result.communities.get(n, 0) for n in B.nodes()]
    nx.draw_networkx(
        B, pos={n: pos[n] for n in B.nodes()}, ax=axes[1],
        node_color=backbone_node_colors, cmap="tab10",
        node_size=200, font_size=7,
        width=[B[u][v].get("weight", 1) / 15 for u, v in B.edges()],
        edge_color="gray", alpha=0.9,
    )

    fig.suptitle("Disparity filter backbone — community colours", fontsize=14)
    fig.tight_layout()
    fig.savefig("backbone_visualisation.png", dpi=150, bbox_inches="tight")
    print("  Visualisation saved to: backbone_visualisation.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    G = build_demo_graph(seed=42)

    demo_single_alpha(G)
    demo_range_alpha(G)
    demo_auto_search(G)
    demo_matrix_input(G)
    demo_infomap(G)
    demo_visualise(G)

    print("\n" + "=" * 60)
    print("All demos completed successfully.")
    print("=" * 60)