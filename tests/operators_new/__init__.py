# operators/__init__.py
"""Helper routines for setting up operators tests."""

import numpy as np


def _get_operators(n=60, m=20, expanded=False):
    """Construct fake model operators."""
    c = np.random.random(n)
    A = np.eye(n)
    H = np.zeros((n, n**2 if expanded else n*(n+1)//2))
    G = np.zeros((n, n**3 if expanded else n*(n+1)*(n+2)//6))
    B = np.random.random((n, m)) if m else None
    return c, A, H, G, B
