"""
Golden-section search to find the alpha that maximises
combined_score(alpha) without a user-supplied range.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


_GOLDEN_RATIO = (np.sqrt(5) - 1) / 2   # ≈ 0.618


def golden_section_search(
    objective: Callable[[float], float],
    a: float = 0.001,
    b: float = 0.9,
    tol: float = 1e-3,
    max_iter: int = 500,
) -> tuple[float, float]:
    """
    Maximise `objective` over [a, b] assuming a single peak (unimodal).

    Uses the golden-section search — converges in
    O(log(tol / (b-a))) evaluations (~50 for default settings).

    Returns
    -------
    alpha_opt : float  – optimal alpha
    score_opt : float  – objective value at alpha_opt
    """
    c = b - _GOLDEN_RATIO * (b - a)
    d = a + _GOLDEN_RATIO * (b - a)

    fc = objective(c)
    fd = objective(d)

    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if fc < fd:
            a = c
            c, fc = d, fd
            d = a + _GOLDEN_RATIO * (b - a)
            fd = objective(d)
        else:
            b = d
            d, fd = c, fc
            c = b - _GOLDEN_RATIO * (b - a)
            fc = objective(c)

    alpha_opt = (a + b) / 2
    score_opt = objective(alpha_opt)
    return alpha_opt, score_opt