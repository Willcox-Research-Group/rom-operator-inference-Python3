# operators/__init__.py
r"""Operator classes for the individual terms of reduced-order models.

.. currentmodule:: opinf.operators_new

For instance, for models with the form

.. math::
    \frac{\textup{d}}{\textup{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}[
    \widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)]
    + \widehat{\mathbf{G}}[
    \widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)
    \otimes\widehat{\mathbf{q}}(t)]
    + \widehat{\mathbf{B}}\mathbf{u}(t),

these classes represent the operators :math:`\widehat{\mathbf{c}}` (constant),
:math:`\widehat{\mathbf{A}}` (linear), :math:`\widehat{\mathbf{H}}`
(quadratic), :math:`\widehat{\mathbf{G}}` (cubic), and
:math:`\widehat{\mathbf{B}}` (input).


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
    InputOperator
    StateInputOperator
"""

from ._nonparametric import *
