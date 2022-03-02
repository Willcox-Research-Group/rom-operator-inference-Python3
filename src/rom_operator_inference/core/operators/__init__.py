# core/operators/__init__.py
"""Operator classes for the individual components of polynomial models.

That is, for models with the form
    dq / dt = c + Aq(t) + H[q(t) ⊗ q(t)] + G[q(t) ⊗ q(t) ⊗ q(t)] + Bu(t),
these classes represent the operators c (constant), A (linear), H (quadratic),
G (cubic), and B (input).

Nonparametric Operator Classes
------------------------------
* ConstantOperator: constant operators (c)
* LinearOperator: linear operators for state and input (A, B)
* QuadraticOperator: quadratic operators (H)
* CubicOperator: quadratic operators (G)

Parametric Operators: 1D Cubic Spline Intepolation
--------------------------------------------------
TODO

Parametric Operators: Affine Expansion
--------------------------------------
TODO
"""

from ._nonparametric import *
from ._affine import *
from ._interpolate import *
