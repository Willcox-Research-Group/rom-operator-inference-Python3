# operators/__init__.py
r"""Operator classes for the individual terms of reduced-order models.

.. currentmodule:: opinf.operators_new

Operator inference ROMs can be written as

.. math::
   \ddt\qhat(t)
   = \sum_{\ell=1}^{n_\textrm{terms}}
   \widehat{\mathcal{F}}_{\ell}(\qhat(t),\u(t))

where each :math:`\widehat{\mathcal{F}}_{\ell}` is an "operator"
that is polynomial in the state :math:`\qhat` and in the input :math:`\u`.
The classes in this module represent different types of operators.
For example, a linear time-invariant (LTI) system

.. math::
   \ddt\qhat(t)
   = \Ahat\qhat(t) + \Bhat\u(t)

can be written as

.. math::
   \ddt\qhat(t)
   = \widehat{\mathcal{A}}(\qhat(t),\u(t))
   + \widehat{\mathcal{B}}(\qhat(t),\u(t))

where
:math:`\widehat{\mathcal{A}}(\qhat,\u) = \Ahat\q` (:class:`ConstantOperator`)
and :math:`\widehat{\mathcal{B}}(\qhat,\u) = \Bhat\u` (:class:`InputOperator`).


Nonparametric Operator Classes
==============================

These classes represent operators that do not depend on external parameters.
They are used by :class:`ContinuousROM` and
:class:`DiscreteROM`

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
