"""Input parsing, validation and degree computation."""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import issparse


def parse_input(data: np.ndarray | nx.Graph) -> tuple[np.ndarray, list]:
    """
    Accept either an adjacency matrix (np.ndarray or scipy sparse)
    or a nx.Graph.

    Returns
    -------
    M : np.ndarray  – float adjacency matrix
    nodes : list    – ordered node labels (indices for matrices)
    """
    if isinstance(data, nx.Graph):
        if isinstance(data, nx.DiGraph):
            raise TypeError("Only undirected graphs are supported. Got a DiGraph.")
        nodes = list(data.nodes())
        M = nx.to_numpy_array(data, nodelist=nodes, weight="weight", dtype=float)
    elif isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError("Adjacency matrix must be a square 2-D array.")
        M = data.astype(float)
        nodes = list(range(M.shape[0]))
    elif issparse(data):
        M = data.toarray().astype(float)
        nodes = list(range(M.shape[0]))
    else:
        raise TypeError(
            f"Input must be a np.ndarray, scipy sparse matrix, or nx.Graph. Got {type(data)}."
        )
    return M, nodes


def validate_undirected(M: np.ndarray) -> None:
    """Raise ValueError if M is not symmetric (i.e. directed)."""
    if not np.allclose(M, M.T):
        raise ValueError(
            "Adjacency matrix is not symmetric. Only undirected graphs are supported."
        )


def compute_degrees(M: np.ndarray) -> np.ndarray:
    """
    Compute the degree of each node from the adjacency matrix.
    Works regardless of whether input was originally a matrix or a graph.

    Degree = number of non-zero neighbours (binary, ignores weights).
    """
    return (M > 0).sum(axis=1).astype(float)