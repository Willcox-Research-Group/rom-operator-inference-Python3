# operators/__init__.py
r"""Operator classes for the individual components of polynomial models.

.. currentmodule:: opinf.operators

For instance, for models with the form

.. math::
    \frac{\textup{d}}{\textup{d}t}\mathbf{q}(t)
    = \mathbf{c}
    + \mathbf{Aq}(t)
    + \mathbf{H}[\mathbf{q}(t) \otimes \mathbf{q}(t)]
    + \mathbf{G}[\mathbf{q}(t) \otimes \mathbf{q}(t) \otimes \mathbf{q}(t)]
    + \mathbf{Bu}(t),

these classes represent the operators :math:`\mathbf{c}` (constant),
:math:`\mathbf{A}` (linear), :math:`\mathbf{H}` (quadratic),
:math:`\mathbf{G}` (cubic), and :math:`\mathbf{B}` (input).


Nonparametric Operator Classes
==============================

These classes represent operators that do not depend on external parameters.
They are used by :class:`ContinuousOpInfROM <opinf.ContinuousOpInfROM>` and
:class:`DiscreteOpInfROM <opinf.DiscreteOpInfROM>`

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ConstantOperator
    LinearOperator
    QuadraticOperator
    CubicOperator


Parametric Operator Classes
===========================

The remaining classes are for operators that depend on a external parameters,
for example, :math:`\mathbf{A} = \mathbf{A}(\mu)` where
:math:`\mu \in \mathbb{R}^{p}` is free parameter vector (or scalar if
:math:`p = 1`). There are several parameterization strategies.


Interpolated Operator Classes
-----------------------------

These classes handle the parametric dependence of an operator
:math:`\mathbf{A} = \mathbf{A}(\mu)`
with element-wise interpolation between known operator matrices, i.e.,

.. math::
    \mathbf{A}(\mu)_{ij}
    = \operatorname{Interpolator}(
        [\mu_{1}, \mu_{2}, \ldots],
        [\mathbf{A}^{(1)}_{i,j}, \mathbf{A}^{(2)}_{i,j}, \ldots])(\mu),

where :math:`\mu_{1}, \mu_{2}, \ldots` are parameter values and
:math:`\mathbf{A}^{(1)}, \mathbf{A}^{(2)}, ...` are the corresponding operator
matrices, i.e., :math:`\mathbf{A}^{(1)} = \mathbf{A}(\mu_{i})`.
These operator classes are used by
:class:`InterpolatedDiscreteOpInfROM <opinf.InterpolatedDiscreteOpInfROM>` and
:class:`InterpolatedContinuousOpInfROM <opinf.InterpolatedContinuousOpInfROM>`.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    InterpolatedConstantOperator
    InterpolatedLinearOperator
    InterpolatedQuadraticOperator
    InterpolatedCubicOperator


Affine Operator Classes
-----------------------

These operators assume the parametric dependence of an operator A = A(µ) has a
known "affine" structure,

.. math::
    \mathbf{A}(\mu)
    = \sum_{i=1} \theta_{i}(\mu)\mathbf{A}^{(i)},

where :math:`\theta_{i}` are scalar-valued functions and
:math:`\mathbf{A}^{(i)}` are matrices.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    AffineConstantOperator
    AffineLinearOperator
    AffineQuadraticOperator
    AffineCubicOperator
"""

from ._base import *
from ._nonparametric import *
from ._affine import *
from ._interpolate import *
