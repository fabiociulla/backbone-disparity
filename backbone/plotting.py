"""Single-panel diagnostic plot for backbone metrics vs alpha."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx

def plot_metrics(
    alphas: list[float],
    community_scores: list[float],
    multipliers_dict: dict[str, list[float]],
    combined_scores_dict: dict[str, list[float]],
    best_alphas_dict: dict[str, float],
    method: str = "louvain",
    save_path: str | None = None,
    log_scale: bool = True,
) -> plt.Figure:

    """
    Single-panel figure with twin y-axes:
      left  → community metric (Q or −codelength)
      right → dynamic multipliers & combined scores per rationale
    """
    alphas = np.array(alphas)
    community_scores = np.array(community_scores)

    metric_label = (
        "Modularity $Q$" if method.lower() == "louvain" else "$-$Codelength"
    )

    fig, ax1 = plt.subplots(figsize=(16, 10), facecolor="white")
    ax2 = ax1.twinx()

    # ── Left axis: community score ────────────────────────────────────────────
    color_comm = "#2166ac"
    l1, = ax1.plot(
        alphas, community_scores,
        color=color_comm, lw=2, label=metric_label,
    )
    ax1.set_xlabel(r"$\alpha$", fontsize=13)
    if log_scale:
        ax1.set_xscale("log")
    ax1.set_ylabel(metric_label, color=color_comm, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_comm)

    # ── Right axis: Dynamic Multipliers & Combined Scores ─────────────────────
    # Pre-defined contrasting palettes (Multiplier Color, Combined Color)
    palettes = [
        ("#4dac26", "#d01c8b"),  # Green & Magenta
        ("#e66101", "#5e3c99"),  # Orange & Purple
        ("#0571b0", "#ca0020"),  # Light Blue & Red
        ("#008837", "#7b3294"),  # Dark Green & Dark Purple
    ]

    ax2.set_ylabel("Multipliers & Combined Scores", fontsize=12)
    
    handles = [l1]
    global_max = 0.0

    for idx, (rat, mult_vals) in enumerate(multipliers_dict.items()):
        c_mult, c_comb = palettes[idx % len(palettes)]
        
        mults = np.array(mult_vals)
        combs = np.array(combined_scores_dict[rat])
        best_alpha = best_alphas_dict[rat]
        
        # Track the absolute maximum to scale the Y axis accurately
        global_max = max(global_max, mults.max(), combs.max())
        
        l_m, = ax2.plot(
            alphas, mults,
            color=c_mult, lw=2, linestyle="--", 
            label=f"{rat.title()} Multiplier",
        )
        l_c, = ax2.plot(
            alphas, combs,
            color=c_comb, lw=2.5, linestyle="-.", 
            label=f"Combined ({rat.title()})",
        )
        
        # Star at argmax combined score
        best_idx = int(np.argmax(combs))
        ax2.plot(
            alphas[best_idx], combs[best_idx],
            marker="o", markersize=16, color=c_comb,
            zorder=5, linestyle="None",
        )
        
        # Vertical line for best alpha
        ax1.axvline(best_alpha, color=c_comb, lw=1, linestyle=":", alpha=0.7)
        
        star_handle = mlines.Line2D(
            [], [], marker="o", color=c_comb, markersize=12,
            linestyle="None", label=f"Best α ({rat}) = {best_alpha:.3f}",
        )
        handles.extend([l_m, l_c, star_handle])

    ax2.set_ylim(0, global_max * 1.15)

    # ── Unified legend ────────────────────────────────────────────────────────
    labels  = [h.get_label() for h in handles]
    # Place legend slightly outside if there are many entries to avoid covering data
    ax1.legend(handles, labels, loc="center left", bbox_to_anchor=(1.10, 0.5), framealpha=0.9, fontsize=10)

    ax1.set_title("Backbone disparity filter — metrics vs $\\alpha$", fontsize=13)
    
    # Adjust layout to fit external legend
    fig.tight_layout(rect=[0, 0, 0.85, 1])

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