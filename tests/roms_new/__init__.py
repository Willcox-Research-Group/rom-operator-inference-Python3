# roms/__init__.py
"""Global variables and helper routines for setting up roms tests."""

import itertools
import numpy as np

import opinf


opinf_operators = opinf.operators_new    # TEMP

# Global variables for testsing ===============================================
MODELFORM_KEYS = "cAHGBN"

MODEL_FORMS = [''.join(s) for k in range(1, len(MODELFORM_KEYS)+1)
               for s in itertools.combinations(MODELFORM_KEYS, k)]


# Helper functions for testing ================================================
def _get_data(n=60, k=25, m=20):
    """Get fake snapshot, time derivative, and input data."""
    X = np.random.random((n, k))
    Xdot = np.random.random((n, k))
    U = np.ones((m, k))

    return X, Xdot, U


def _get_operators(operatorkeys, r=8, m=1):
    """Construct fake model operators."""
    possibles = {
        'c': opinf_operators.ConstantOperator(np.random.random(r)),
        'A': opinf_operators.LinearOperator(np.eye(r)),
        'H': opinf_operators.QuadraticOperator(np.zeros((r, r*(r+1)//2))),
        'G': opinf_operators.CubicOperator(np.zeros((r, r*(r+1)*(r+2)//6))),
        'B': opinf_operators.InputOperator(np.random.random((r, m))),
        'N': opinf_operators.StateInputOperator(np.random.random((r, r*m))),
    }
    return [possibles[k] for k in operatorkeys]


def _trainedmodel(ModelClass, modelform, basis, m=20):
    """Construct a base class with model operators already constructed."""
    n, r = basis.shape
    rom = ModelClass(_get_operators(modelform, r, m))
    rom.basis = basis
    return rom
