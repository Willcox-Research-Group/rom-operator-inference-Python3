# lstsq/__init__.py
"""Least-squares solvers for the Operator Inference problem."""

from ._tikhonov import *


# TODO: rename lstsq_size() -> size()
def lstsq_size(modelform, r, m=0, affines=None):
    """Calculate the number of columns in the operator matrix O in the Operator
    Inference least-squares problem. This is also the number of columns in the
    data matrix D.

    Parameters
    ---------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    r : int
        The dimension of the reduced order model.

    m : int
        The dimension of the inputs of the model.
        Must be zero unless 'B' is in `modelform`.

    affines : dict(str -> list(callables))
        Functions that define the structures of the affine operators.
        Keys must match the modelform:
        * 'c': Constant term c(µ).
        * 'A': Linear state matrix A(µ).
        * 'H': Quadratic state matrix H(µ).
        * 'G': Cubic state matrix G(µ).
        * 'B': linear Input matrix B(µ).
        For example, if the constant term has the affine structure
        c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

    Returns
    -------
    ncols : int
        The number of columns in the Operator Inference least-squares problem.
    """
    if 'B' in modelform and m == 0:
        raise ValueError("argument m > 0 required since 'B' in modelform")
    if 'B' not in modelform and m != 0:
        raise ValueError(f"argument m={m} invalid since 'B' in modelform")

    if affines is None:
        affines = {}

    qs = [(len(affines[op]) if (op in affines and op in modelform)
           else 1 if op in modelform else 0) for op in "cAHGB"]
    rs = [1, r, r*(r+1)//2, r*(r+1)*(r+2)//6, m]

    return sum(qq*rr for qq, rr in zip(qs, rs))
