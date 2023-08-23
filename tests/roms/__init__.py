# roms/__init__.py
"""Global variables and helper routines for setting up roms tests."""

import itertools
import numpy as np

import opinf


# Global variables for testsing ===============================================
MODELFORM_KEYS = opinf.roms._base._BaseROM._MODELFORM_KEYS

MODEL_FORMS = [''.join(s) for k in range(1, len(MODELFORM_KEYS)+1)
               for s in itertools.combinations(MODELFORM_KEYS, k)]


# Helper functions for testing ================================================
def _get_data(n=60, k=25, m=20):
    """Get fake snapshot, time derivative, and input data."""
    X = np.random.random((n, k))
    Xdot = np.random.random((n, k))
    U = np.ones((m, k))

    return X, Xdot, U


def _get_operators(n=60, m=20, expanded=False):
    """Construct fake model operators."""
    c = np.random.random(n)
    A = np.eye(n)
    H = np.zeros((n, n**2 if expanded else n*(n+1)//2))
    G = np.zeros((n, n**3 if expanded else n*(n+1)*(n+2)//6))
    B = np.random.random((n, m)) if m else None
    return c, A, H, G, B


def _trainedmodel(ModelClass, modelform, basis, m=20):
    """Construct a base class with model operators already constructed."""
    n, r = basis.shape
    c, A, H, G, B = _get_operators(r, m)
    operators = {}
    if "c" in modelform:
        operators['c_'] = c
    if "A" in modelform:
        operators['A_'] = A
    if "H" in modelform:
        operators['H_'] = H
    if "G" in modelform:
        operators['G_'] = G
    if "B" in modelform:
        operators['B_'] = B

    return ModelClass(modelform)._set_operators(basis, **operators)
