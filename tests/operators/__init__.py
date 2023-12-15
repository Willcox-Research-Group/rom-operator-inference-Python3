# operators/__init__.py
"""Helper routines for setting up operators tests."""

import numpy as np


def _get_operator_entries(r=10, m=3, expanded=False):
    """Construct fake model operators."""
    c = np.random.random(r)
    A = np.eye(r)
    H = np.zeros((r, r**2 if expanded else r * (r + 1) // 2))
    G = np.zeros((r, r**3 if expanded else r * (r + 1) * (r + 2) // 6))
    B = np.random.random((r, m)) if m else None
    N = np.random.random((r, r * m)) if m else None
    return c, A, H, G, B, N
