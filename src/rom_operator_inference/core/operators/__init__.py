# core/operators/__init__.py
"""Operator classes for the individual components of polynomial models.

That is, for models with the form
    dq / dt = c + Aq(t) + H[q(t) ⊗ q(t)] + G[q(t) ⊗ q(t) ⊗ q(t)],
these classes represent the operators c (constant), A (linear), H (quadratic),
and G (cubic).

Classes
-------
*
"""

from ._nonparametric import *
# from ._affine import *
# from ._interpolate import *
