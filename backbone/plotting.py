"""Single-panel diagnostic plot for backbone metrics vs alpha."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx

def plot_metrics(
    alphas: list[float],
    community_scores: list[float],
    gc_ratios: list[float],
    combined_scores: list[float],
    method: str = "louvain",
    best_alpha: float | None = None,
    save_path: str | None = None,
    log_scale: bool = True,          # ← new parameter
) -> plt.Figure:

    """
    Single-panel figure with twin y-axes:
      left  → community metric (Q or −codelength)
      right → giant-component ratio & combined score

    A ★ marker is placed at the alpha with the highest combined score.

    Parameters
    ----------
    alphas, community_scores, gc_ratios, combined_scores : parallel lists
    method     : 'louvain' or 'infomap'  (used for axis label)
    best_alpha : if provided, adds a vertical dashed line
    save_path  : if provided, saves the figure to that path
    """
    alphas = np.array(alphas)
    community_scores = np.array(community_scores)
    gc_ratios = np.array(gc_ratios)
    combined_scores = np.array(combined_scores)

    metric_label = (
        "Modularity $Q$" if method.lower() == "louvain" else "$-$Codelength"
    )

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    # ── Left axis: community score ────────────────────────────────────────────
    color_comm = "#2166ac"
    l1, = ax1.plot(
        alphas, community_scores,
        color=color_comm, lw=2, label=metric_label,
    )
    ax1.set_xlabel(r"$\alpha$", fontsize=13)
    if log_scale:                          # ← add this
        ax1.set_xscale("log")              # ← and this
    ax1.set_ylabel(metric_label, color=color_comm, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_comm)

    # ── Right axis: GC ratio & combined score ─────────────────────────────────
    color_gc   = "#4dac26"
    color_comb = "#d01c8b"

    l2, = ax2.plot(
        alphas, gc_ratios,
        color=color_gc, lw=2, linestyle="--", label="GC ratio",
    )
    l3, = ax2.plot(
        alphas, combined_scores,
        color=color_comb, lw=2.5, linestyle="-.", label="Combined score",
    )
    # ── Right axis: non-orphan ratio & combined score ─────────────────────────
    ax2.set_ylabel("Non-orphan ratio / Combined score", fontsize=12)  # ← updated label
    ax2.set_ylim(0, max(gc_ratios.max(), combined_scores.max()) * 1.15)

    # ── Star at argmax combined score ─────────────────────────────────────────
    best_idx = int(np.argmax(combined_scores))
    ax2.plot(
        alphas[best_idx], combined_scores[best_idx],
        marker="*", markersize=16, color=color_comb,
        zorder=5, linestyle="None",
    )

    # Optional vertical line at externally supplied best_alpha
    if best_alpha is not None:
        ax1.axvline(best_alpha, color="gray", lw=1, linestyle=":",
                    label=f"best α={best_alpha:.3f}")

    # ── Unified legend (single call across both axes) ─────────────────────────
    star_handle = mlines.Line2D(
        [], [], marker="*", color=color_comb, markersize=12,
        linestyle="None", label=f"Best $\\alpha$={alphas[best_idx]:.3f}",
    )
    handles = [l1, l2, l3, star_handle]
    labels  = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right", framealpha=0.9, fontsize=10)

    ax1.set_title("Backbone disparity filter — metrics vs $\\alpha$", fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

def plot_network(
    G: nx.Graph,
    communities: dict | None = None,
    labels: list | None = None,
    title: str = "Network",
    node_size: int = 300,
    font_size: int = 8,
    figsize: tuple = (10, 10),
    save_path: str | None = None,
    seed: int = 42,
    layout: str = "spring",          # ← new parameter: "spring", "neato", "dot", "fdp", "sfdp", "circo"
) -> plt.Figure:
    """
    Plot a NetworkX graph.

    Parameters
    ----------
    G           : nx.Graph to plot
    communities : dict[node → community_id], optional
                  If provided, nodes are coloured by community.
    labels      : list, optional
                  If provided, node i receives the label labels[i].
                  The list must be ordered consistently with the node
                  integer ids (0, 1, 2, ...), i.e. labels[node_id] = label.
    title       : figure title
    node_size   : size of each node circle
    font_size   : font size for node labels
    figsize     : figure size in inches
    save_path   : if provided, saves the figure to that path
    seed        : random seed for the spring layout (ignored for pygraphviz layouts)
    layout      : layout algorithm to use:
                    "spring"  — NetworkX spring layout (default)
                    "neato"   — pygraphviz neato (spring-model, best for general graphs)
                    "dot"     — pygraphviz dot (hierarchical)
                    "fdp"     — pygraphviz fdp (spring-model, large graphs)
                    "sfdp"    — pygraphviz sfdp (multiscale, very large graphs)
                    "circo"   — pygraphviz circo (circular)

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # ── Layout ────────────────────────────────────────────────────────────────
    _PYGRAPHVIZ_LAYOUTS = {"neato", "dot", "fdp", "sfdp", "circo", "twopi"}

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout in _PYGRAPHVIZ_LAYOUTS:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog=layout)
        except ImportError:
            raise ImportError(
                "pygraphviz is not installed. "
                "Run: pip install pygraphviz\n"
                "On Ubuntu/Debian you may also need: "
                "sudo apt install graphviz libgraphviz-dev"
            )
        except Exception as e:
            raise RuntimeError(
                f"pygraphviz layout '{layout}' failed: {e}\n"
                "Make sure graphviz system binaries are installed."
            )
    else:
        raise ValueError(
            f"Unknown layout '{layout}'. "
            f"Choose 'spring', or a pygraphviz layout: "
            f"{sorted(_PYGRAPHVIZ_LAYOUTS)}."
        )

    # ── Node colours by community ─────────────────────────────────────────────
    if communities is not None:
        community_ids = [communities.get(node, 0) for node in G.nodes()]
        unique_communities = sorted(set(community_ids))
        n_communities = len(unique_communities)

        cmap = plt.cm.get_cmap("tab20", n_communities)
        comm_to_idx = {c: i for i, c in enumerate(unique_communities)}
        node_colors = [cmap(comm_to_idx[c]) for c in community_ids]
    else:
        node_colors = "#4393c3"

    # ── Node labels ───────────────────────────────────────────────────────────
    if labels is not None:
        label_map = {node: labels[node] for node in G.nodes() if node < len(labels)}
    else:
        label_map = {node: str(node) for node in G.nodes()}

    # ── Draw ──────────────────────────────────────────────────────────────────
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax,
        alpha=1., edge_color="black",
        width=[G[u][v].get("weight", 1.0) for u, v in G.edges()],
    )
    nx.draw_networkx_nodes(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
    )
    nx.draw_networkx_labels(
        G, pos=pos, ax=ax,
        labels=label_map,
        font_size=font_size,
        font_color="black",
    )

    # ── Legend (one entry per community) ─────────────────────────────────────
    if communities is not None:
        handles = [
            plt.Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor=cmap(comm_to_idx[c]),
                markersize=10,
                label=f"Community {c}",
            )
            for c in unique_communities
        ]
        ax.legend(
            handles=handles,
            title="Communities",
            loc="best",
            framealpha=0.9,
            fontsize=9,
        )

    ax.set_title(f"{title}  [{layout}]", fontsize=14)
    ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig