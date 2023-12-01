# operators/__init__.py
r"""Operator classes for the individual terms of reduced-order models.

.. currentmodule:: opinf.operators_new

For models with the form

.. math::
    \ddt\qhat(t)
    = \chat + \Ahat\qhat(t) + \Hhat[\qhat(t)\otimes\qhat(t)]
    + \Ghat[\qhat(t)\otimes\qhat(t)\otimes\qhat(t)]
    + \Bhat\u(t),

these classes represent the operators :math:`\chat` (constant),
:math:`\Ahat` (linear), :math:`\Hhat`
(quadratic), :math:`\widehat{\mathbf{G}}` (cubic), and
:math:`\Bhat` (input).


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
