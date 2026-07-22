# backbone-disparity

A Python library for extracting the **backbone** of a weighted network using the
[disparity filter](https://www.pnas.org/doi/10.1073/pnas.0808904106) (Serrano et al., 2009),
combined with community detection to identify the most meaningful network structure.

---

## Table of Contents

- [Background](#background)
  - [The Problem with Thresholding](#the-problem-with-thresholding)
  - [The Disparity Filter](#the-disparity-filter)
  - [Community Detection](#community-detection)
  - [Choosing Alpha Automatically](#choosing-alpha-automatically)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Input Formats](#input-formats)
  - [Single Alpha](#single-alpha)
  - [Range of Alphas](#range-of-alphas)
  - [Automatic Alpha Search](#automatic-alpha-search)
  - [Choosing a Community Detection Method](#choosing-a-community-detection-method)
  - [Working with the Result](#working-with-the-result)
- [API Reference](#api-reference)
- [Output: BackboneResult](#output-backboneresult)
- [Diagnostic Plot](#diagnostic-plot)
- [Special Cases](#special-cases)
- [References](#references)

---

## Background

### The Problem with Thresholding

Real-world networks — trade flows, social interactions, co-authorship, brain
connectivity — are almost always **weighted**. A natural way to simplify them is
to remove edges whose weight falls below some threshold. However, this naive
approach systematically destroys the structure of **low-strength nodes**: a node
that participates in only weak interactions loses all its connections, even if
some of those connections are the *only* significant relationships it has.

### The Disparity Filter

The disparity filter (Serrano, Boguñá & Vespignani, 2009) solves this by asking
a local question for every node *i* and each of its edges *(i, j)*:

> *Is the weight of edge (i, j) statistically compatible with a null model in
> which node i distributes its total strength uniformly among its k neighbours?*

Under this null model the probability that a single edge carries a fraction
*p* of the total strength is:

$$\alpha_{ij} = (1 - p_{ij})^{k_i - 1}$$

Edge *(i, j)* is **significant** for node *i* when $\alpha_{ij} < \alpha$, where
$\alpha$ is the user-chosen significance level. Because the test is *directional*
(node *i* may find the edge significant even if node *j* does not), the filter
retains an edge when **at least one** endpoint finds it significant — this is the
*union* criterion implemented here.

In matrix form, using the row-normalised weight matrix $\hat{M}$:

```
M_norm      = L1-row-normalise(M)
idx_to_zero = (1 − M_norm)^(k−1) ≥ α        # insignificant from each side
idx_to_zero = idx_to_zero AND idx_to_zero.T  # must be insignificant from BOTH sides
M_alpha     = M, then set M_alpha[idx_to_zero] = 0
```

The result is a **sparse backbone** that preserves the multiscale structure of
the network — both heavy hubs and lightweight but locally important nodes keep
their relevant connections.

### Community Detection

Once the backbone is extracted, community structure is measured using one of two
algorithms:

| Algorithm | Score | Interpretation |
|-----------|-------|----------------|
| **Louvain** | Modularity *Q* ∈ (−1, 1) | Higher → clearer community structure |
| **Infomap** | −Codelength (bits) | Higher (less negative) → more compressible, clearer structure |

To account for backbone networks that are highly fragmented by strict filtering, the raw community score is multiplied by a structural **rationale metric**:

$$\text{combined score}(\alpha) = \text{community score}(\alpha) \times \text{multiplier}(\alpha)$$

You can optimize the backbone using one or more rationales simultaneously via the `rationales` parameter. The currently supported rationales are:
1. **`"non orphan ratio"`** (default): The fraction of nodes with degree > 0 in the backbone. Penalises alpha values that over-prune the network into isolated nodes.
2. **`"giant component ratio"`**: The fraction of original nodes belonging to the largest connected component of the backbone. Heavily penalises fragmentation into disconnected islands.

### Choosing Alpha Automatically

| Mode | How to invoke | Strategy |
|------|--------------|----------|
| **Single value** | `alpha=0.05` | Run once, return result |
| **Range sweep** | `alpha=[0.01, 0.05, …]` | Evaluate all, return best |
| **Auto search** | `alpha=None` *(default)* | Golden-section search |

The **golden-section search** assumes the combined score is unimodal in α (a
reasonable approximation: very small α leaves the graph unpruned and community
structure is noisy; very large α over-prunes into fragments). It converges to
the optimum in approximately
$$\lceil\log(\text{tol}/(b-a))\,/\,\log\phi\rceil \approx 45$$ evaluations for
the default tolerance of `1e-3`, far fewer than an exhaustive grid search.

---

## Repository Structure

```
backbone-disparity/
├── backbone/
│   ├── __init__.py      ← public API: compute_backbone(), BackboneResult
│   ├── utils.py         ← input parsing, validation, degree computation
│   ├── filter.py        ← disparity filter + graph reconstruction
│   ├── community.py     ← Louvain and Infomap wrappers
│   ├── metrics.py       ← giant-component ratio, combined score
│   ├── optimize.py      ← golden-section search
│   └── plotting.py      ← single-panel diagnostic figure
├── tests/
│   ├── test_utils.py
│   ├── test_filter.py
│   ├── test_community.py
│   ├── test_metrics.py
│   └── test_backbone.py
├── examples/
│   └── demo.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/fabiociulla/backbone-disparity.git
cd backbone-disparity

python3.8 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python examples/demo.py
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Matrix operations |
| `scipy` | ≥ 1.10 | Sparse matrices |
| `networkx` | ≥ 3.0 | Graph construction and analysis |
| `scikit-learn` | ≥ 1.3 | L1 row normalisation |
| `python-louvain` | ≥ 0.16 | Louvain community detection |
| `infomap` | ≥ 2.7 | Infomap community detection |
| `matplotlib` | ≥ 3.7 | Diagnostic plots |
| `pygraphviz` | ≥ 1.11 | pygraphviz layouts (neato, dot, fdp, sfdp, circo) |


> **Note on `infomap`**: the `infomap` package requires a C++ compiler at
> install time on some platforms. If you encounter build issues, see the
> [official installation guide](https://mapequation.github.io/infomap/).

> **Note on `pygraphviz`**: requires the graphviz system binaries in addition to the Python package.
>
> Ubuntu / Debian:
> ```bash
> sudo apt install graphviz libgraphviz-dev
> pip install pygraphviz
> ```
> macOS:
> ```bash
> brew install graphviz
> pip install pygraphviz
> ```

### Running the tests

```bash
pytest tests/ -v --tb=short
```

---

## Quick Start


```python
import networkx as nx
from backbone import compute_backbone

# Build any weighted undirected graph
G = nx.karate_club_graph()
for u, v in G.edges():
    G[u][v]["weight"] = 1.0          # add weights if missing

# Let the library find the best alphas automatically
results = compute_backbone(G)
best = results["non orphan ratio"]   # Extract the default rationale

print(f"Best alpha      : {best.alpha:.4f}")
print(f"Backbone edges  : {best.backbone.number_of_edges()}")
print(f"Modularity Q    : {best.score:.4f}")
print(f"Multiplier      : {best.multiplier_value:.4f}")
print(f"Combined score  : {best.combined_score:.4f}")
print(f"Communities     : {set(best.communities.values())}")
```

---

## Usage

### Input Formats

The library accepts three equivalent input formats:

```python
import numpy as np
import scipy.sparse as sp
import networkx as nx
from backbone import compute_backbone

# 1. NumPy adjacency matrix (must be square and symmetric)
M = np.array([[0, 3, 1],
              [3, 0, 2],
              [1, 2, 0]], dtype=float)
result = compute_backbone(M, alpha=0.1)

# 2. SciPy sparse matrix
M_sparse = sp.csr_matrix(M)
result = compute_backbone(M_sparse, alpha=0.1)

# 3. NetworkX undirected graph (with optional 'weight' edge attribute)
G = nx.from_numpy_array(M)
result = compute_backbone(G, alpha=0.1)
```

> **Only undirected graphs are accepted.** Passing an asymmetric matrix or a
> `nx.DiGraph` raises an error immediately.

---

### Single Alpha

```python
results = compute_backbone(
    G, alpha=0.05, method="louvain", plot=False,
    rationales=["non orphan ratio", "giant component ratio"]
)

res_gc = results["giant component ratio"]
print(res_gc.backbone)       # nx.Graph – the backbone network
print(res_gc.communities)    # dict: node → community id
print(res_gc.score)          # modularity Q
print(res_gc.multiplier_value) # fraction of nodes in giant component
print(res_gc.combined_score) # Q × multiplier_value
```

---

### Range of Alphas

Sweep over a list of alpha values. Only scalar metrics are stored per step
(no intermediate graph objects). The backbone and communities corresponding to
the **maximum combined score** are returned, together with a diagnostic plot:

```python
import numpy as np
from backbone import compute_backbone

alphas = np.linspace(0.01, 0.95, 50).tolist()

results = compute_backbone(
    G, alpha=alphas, method="louvain", plot=True, save_plot="backbone_metrics.png",
    rationales=["non orphan ratio", "giant component ratio"]
)

# Inspect best alphas for each rationale
for rationale, best_result in results.items():
    print(f"Best alpha for '{rationale}': {best_result.alpha:.4f}")

# Inspect the full sweep for one rationale
for record in results["non orphan ratio"].alphas_data:
    print(f"  α={record.alpha:.3f}  Q={record.community_score:.3f}  "
          f"Mult={record.multiplier_value:.3f}  combined={record.combined_score:.3f}")
```

---

### Automatic Alpha Search

```python
from backbone import compute_backbone

results = compute_backbone(
    G,
    alpha=None,       # default — triggers golden-section search
    method="infomap",
    tol=1e-3,
    plot=True,
    rationales=["non orphan ratio", "giant component ratio"]
)

print(f"Optimum (Non Orphan) : {results['non orphan ratio'].alpha:.4f}")
print(f"Optimum (Giant Comp) : {results['giant component ratio'].alpha:.4f}")
```

The search typically requires **~45 backbone evaluations** to converge,
regardless of graph size.

---

### Choosing a Community Detection Method

```python
# Louvain — fast, maximises modularity Q
result = compute_backbone(G, alpha=0.1, method="louvain")

# Infomap — flow-based, score is −codelength (higher = better)
result = compute_backbone(G, alpha=0.1, method="infomap")
```

---

### Working with the Result

```python
results = compute_backbone(G, alpha=0.1)
best = results["non orphan ratio"]

# Access the backbone as a standard NetworkX graph
B = best.backbone
print(nx.info(B))

# Visualise with communities as colours
import matplotlib.pyplot as plt
node_colors = [best.communities[n] for n in B.nodes()]
nx.draw(B, node_color=node_colors, with_labels=True, cmap="tab10")
plt.show()

# Export to edge list
nx.write_edgelist(B, "backbone.edgelist", data=["weight"])

# Export to GML for Gephi / Cytoscape
nx.write_gml(B, "backbone.gml")
```

---

## API Reference

### `compute_backbone`

```python
backbone.compute_backbone(
    data,
    alpha=None,
    method="louvain",
    tol=1e-3,
    plot=True,
    save_plot=None,
    log_scale=True,
    rationales=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `np.ndarray`, `scipy.sparse`, `nx.Graph` | — | Input network (undirected) |
| `alpha` | `float`, `list`, `None` | `None` | Significance level(s) |
| `method` | `str` | `"louvain"` | `"louvain"` or `"infomap"` |
| `tol` | `float` | `1e-3` | Convergence tolerance for golden-section search |
| `plot` | `bool` | `True` | Show diagnostic plot (range / search modes) |
| `save_plot` | `str \| None` | `None` | File path to save the diagnostic figure |
| `log_scale` | `bool` | `True` | Plot alpha axis in logarithmic scale |


**Returns**: `BackboneResult`

---

## Output: `BackboneResult`

```python
@dataclass
class BackboneResult:
    rationale        : str               # e.g., 'non orphan ratio'
    alpha            : float             # best / supplied / searched alpha
    backbone         : nx.Graph          # backbone network at best alpha
    communities      : dict              # node → community id
    score            : float             # community metric (Q or −codelength)
    multiplier_value : float             # value of the rationale metric at this alpha
    combined_score   : float             # score × multiplier_value
    method           : str               # 'louvain' or 'infomap'
    alphas_data      : list[AlphaRecord] # populated in range mode only
```

Each entry in `alphas_data` is an `AlphaRecord`:

```python
@dataclass
class AlphaRecord:
    alpha            : float
    community_score  : float
    multiplier_value : float
    combined_score   : float
    rationale        : str
```

---

## Diagnostic Plot

When `plot=True` in range or auto-search mode, a single-panel figure is
produced with **twin y-axes**:

```
left  axis  →  community metric (Q  or  −codelength)
right axis  →  dynamic multipliers & combined scores per rationale
x-axis      →  α values
★           →  argmax of combined score(s)
· · ·       →  vertical dotted lines at the returned best alpha(s)
```

Save the figure by passing `save_plot="my_figure.png"` to `compute_backbone`.

---

## Special Cases

| Situation | Behaviour |
|-----------|-----------|
| Node with **degree 0** in the original graph | Always retained as an isolate in every backbone |
| Node with **degree 1** in the original graph | Its single edge is always retained, regardless of α |
| **Disconnected** backbone | Giant-component ratio < 1 penalises the combined score |
| **Over-pruned** backbone (many orphan nodes) | Non-orphan ratio < 1 penalises the combined score |
| Alpha axis | Plotted in logarithmic scale by default (`log_scale=True`) |
| `nx.DiGraph` input | Raises `TypeError` immediately |
| Asymmetric matrix input | Raises `ValueError` immediately |
| Graph with no edges | Returns the original node set with no edges |

---

## References

- Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). **Extracting the
  multiscale backbone of complex weighted networks.** *Proceedings of the
  National Academy of Sciences*, 106(16), 6483–6488.
  https://doi.org/10.1073/pnas.0808904106

- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008).
  **Fast unfolding of communities in large networks.** *Journal of Statistical
  Mechanics: Theory and Experiment*, 2008(10), P10008.
  https://doi.org/10.1088/1742-5468/2008/10/P10008

- Rosvall, M., & Bergstrom, C. T. (2008). **Maps of random walks on complex
  networks reveal community structure.** *Proceedings of the National Academy
  of Sciences*, 105(4), 1118–1123.
  https://doi.org/10.1073/pnas.0706851105

- NetworkX documentation: https://networkx.org/documentation/stable/

- Infomap documentation: https://mapequation.github.io/infomap/

- python-louvain documentation: https://python-louvain.readthedocs.io/
