# lstsq/__init__.py
r"""Solvers for Operator Inference least-squares problems.

.. currentmodule:: opinf.lstsq

Introduction
============

The following :ref:`least-squares regression problem <subsec-opinf-regression>`
is at the heart of Operator Inference:

.. math::
    \min_{
        \chat,
        \Ahat,
        \Hhat,
        \Bhat}
    \sum_{j=0}^{k-1}\left\|
        \chat
        + \Ahat\qhat_{j}
        + \Hhat[
            \qhat_{j}\otimes\qhat_{j}]
        + \Bhat\u_{j}
        - \dot{\qhat}_{j}
    \right\|_{2}^{2}
    \\
    = \min_{\Ohat}\left\|
        \D\Ohat\trp - \mathbf{Y}\trp
    \right\|_{F}^{2},

where

* :math:`\qhat_{j}` is a low-dimensional representation of the
  state at time :math:`t_{j}`,
* :math:`\dot{\qhat}_{j}` is the time derivative of the
  low-dimensional state at time :math:`t_{j}`,
* :math:`\u_{j} = \u(t_{j})` is the input at time
  :math:`t_{j}`,
* :math:`\D` is the *data matrix* containing low-dimensional state
  data,
* :math:`\Ohat` is the *operator matrix* of unknown operators to
  be inferred, and
* :math:`\mathbf{Y}` is the matrix of low-dimensional time derivative data.

This module defines least-squares solver classes for solving the above
regression problem. These classes should be instantiated before being passed to
the ``fit()`` method of a ROM class using the ``solver`` keyword argument.
In addition, the following function calculates the column dimension of
:math:`\Ohat` and :math:`\D`.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    lstsq_size


Default Solver
==============

If ``solver`` is not specified, the default is to use
the following class.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    PlainSolver


Tikhonov Regularization
=======================

It is often helpful to add a *regularization term*
:math:`\mathcal{R}(\Ohat)` to the least-squares objective
function in order to penalize the entries of the learned operators. This
promotes stability and accuracy in the learned reduced-order model by
preventing overfitting. The problem stated above then becomes

.. math::
    \min_{\Ohat}\left\|
        \D\Ohat\trp - \mathbf{Y}\trp
    \right\|_{F}^{2} + \mathcal{R}(\Ohat).

The following classes solve the above problem with different choices of
:math:`\mathcal{R}(\Ohat)`.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    L2Solver
    L2SolverDecoupled
    TikhonovSolver
    TikhonovSolverDecoupled


Total Least-Squares Solver
============================

If you want to use the total least-squares solver use
the following class.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    TotalLeastSquaresSolver
"""

from ._base import *
from ._tikhonov import *
from ._total import *
