# roms/__init__.py
"""Global variables and helper routines for setting up roms tests."""

import itertools
import numpy as np

import opinf


# Global variables for testsing ===============================================
MODELFORM_KEYS = "cAHGBN"

MODEL_FORMS = [
    "".join(s)
    for k in range(1, len(MODELFORM_KEYS) + 1)
    for s in itertools.combinations(MODELFORM_KEYS, k)
]


# Helper functions for testing ================================================
def _get_data(n=60, k=25, m=20):
    """Get dummy snapshot, time derivative, and input data."""
    Q = np.random.random((n, k))
    Qdot = np.random.random((n, k))
    U = np.ones((m, k))

    return Q, Qdot, U


def _get_operators(operatorkeys, r=8, m=1):
    """Construct fake model operators."""
    ops = []
    for key in operatorkeys:
        if key == "c":
            ops.append(opinf.operators.ConstantOperator(np.random.random(r)))
        elif key == "A":
            ops.append(opinf.operators.LinearOperator(np.eye(r)))
        elif key == "H":
            entries = np.zeros((r, r * (r + 1) // 2))
            ops.append(opinf.operators.QuadraticOperator(entries))
        elif key == "G":
            entries = np.zeros((r, r * (r + 1) * (r + 2) // 6))
            ops.append(opinf.operators.CubicOperator(entries))
        elif key == "B":
            ops.append(opinf.operators.InputOperator(np.random.random((r, m))))
        elif key == "N":
            entries = np.random.random((r, r * m))
            ops.append(opinf.operators.StateInputOperator(entries))
        else:
            raise KeyError
    return ops


def _trainedmodel(ModelClass, operatorkeys, r, m=20):
    """Construct a base class with model operators already constructed."""
    return ModelClass(_get_operators(operatorkeys, r, m))
