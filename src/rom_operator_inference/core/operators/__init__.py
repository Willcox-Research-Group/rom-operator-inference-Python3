# core/operators/__init__.py
"""Operator classes for the individual components of polynomial models.

That is, for models with the form
    dq / dt = c + Aq(t) + H[q(t) ⊗ q(t)] + G[q(t) ⊗ q(t) ⊗ q(t)] + Bu(t),
these classes represent the operators c (constant), A (linear), H (quadratic),
G (cubic), and B (input).

Nonparametric Operator Classes
==============================
These classes represent operators that do not depend on external parameters.
* ConstantOperator: constant operators, c.
* LinearOperator: linear operators for state / input, A & B.
* QuadraticOperator: quadratic operators, H.
* CubicOperator: quadratic operators, G.
Used by [Discrete|Continuous]OpInfROM.

The remaining classes are for operators that depend on a external parameters,
i.e., A = A(µ). There are several parameterization strategies.

Interpolated Operator Classes
=============================
These classes handle the parametric dependence of an operator A = A(µ)
with elementwise interpolation between known operator matrices, i.e.,
    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ),
where µ1, µ2, ... are parameter values and A1, A2, ... are the corresponding
operator matrices, e.g., A1 = A(µ1).
* InterpolatedConstantOperator: constant operators, c(µ).
* InterpolatedLinearOperator: linear operators for state / input, A(µ) & B(µ).
* InterpolatedQuadraticOperator: quadratic operators, H(µ).
* InterpolatedCubicOperator: cubic operators, G(µ).
Used by Interpolated[Discrete|Continuous]OpInfROM.

Affine Operator Classes
=======================
These operators assume the parametric dependence of an operator A = A(µ) has a
known "affine" structure,
    A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i},
where θ_{i} are scalar-valued functions and A_{i} are matrices.
* AffineConstantOperator: constant operators, c(µ).
* AffineLinearOperator: linear operators for state / input, A(µ) & B(µ).
* AffineQuadraticOperator: quadratic operators, H(µ).
* AffineCubicOperator: cubic operators, G(µ).
Used by Affine[Discrete|Continuous]OpInfROM.
"""

from ._base import *
from ._nonparametric import *
from ._affine import *
from ._interpolate import *
