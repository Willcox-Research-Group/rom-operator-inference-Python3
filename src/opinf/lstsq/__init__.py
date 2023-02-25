# lstsq/__init__.py
r"""Least-squares solvers for the Operator Inference problem.

.. currentmodule:: opinf.lstsq

Introduction
============

The following :ref:`least-squares regression problem <subsec-opinf-regression>`
is at the heart of Operator Inference:

.. math::
    \min_{
        \widehat{\mathbf{c}},
        \widehat{\mathbf{A}},
        \widehat{\mathbf{H}},
        \widehat{\mathbf{B}}}
    \sum_{j=0}^{k-1}\left\|
        \widehat{\mathbf{c}}
        + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
        + \widehat{\mathbf{H}}[
            \widehat{\mathbf{q}}_{j}\otimes\widehat{\mathbf{q}}_{j}]
        + \widehat{\mathbf{B}}\mathbf{u}_{j}
        - \dot{\widehat{\mathbf{q}}}_{j}
    \right\|_{2}^{2}
    \\
    = \min_{\widehat{\mathbf{O}}}\left\|
        \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{Y}^{\mathsf{T}}
    \right\|_{F}^{2},

where

* :math:`\widehat{\mathbf{q}}_{j}` is a low-dimensional representation of the
  state at time :math:`t_{j}`,
* :math:`\dot{\widehat{\mathbf{q}}}_{j}` is the time derivative of the
  low-dimensional state at time :math:`t_{j}`,
* :math:`\mathbf{u}_{j} = \mathbf{u}(t_{j})` is the input at time
  :math:`t_{j}`,
* :math:`\mathbf{D}` is the *data matrix* containing low-dimensional state
  data,
* :math:`\widehat{\mathbf{O}}` is the *operator matrix* of unknown operators to
  be inferred, and
* :math:`\mathbf{Y}` is the matrix of low-dimensional time derivative data.

This module defines least-squares solver classes for solving the above
regression problem. These classes should be instantiated before being passed to
the ``fit()`` method of a ROM class using the ``solver`` keyword argument.
In addition, the following function calculates the column dimension of
:math:`\widehat{\mathbf{O}}` and :math:`\mathbf{D}`.

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
:math:`\mathcal{R}(\widehat{\mathbf{O}})` to the least-squares objective
function in order to penalize the entries of the learned operators. This
promotes stability and accuracy in the learned reduced-order model by
preventing overfitting. The problem stated above then becomes

.. math::
    \min_{\widehat{\mathbf{O}}}\left\|
        \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{Y}^{\mathsf{T}}
    \right\|_{F}^{2} + \mathcal{R}(\widehat{\mathbf{O}}).

The following classes solve the above problem with different choices of
:math:`\mathcal{R}(\widehat{\mathbf{O}})`.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    L2Solver
    L2SolverDecoupled
    TikhonovSolver
    TikhonovSolverDecoupled
"""

from ._base import *
from ._tikhonov import *
