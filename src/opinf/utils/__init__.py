# utils/__init__.py
r"""Utility functions.

.. currentmodule:: opinf.utils

This module contains miscellaneous support functions for the rest of the
package.


Time Derivative Estimation
==========================

For time-continuous reduced-order models, Operator Inference requires the time
derivative of the state snapshots. If they are not available from a full-order
solver, the time derivatives can often be estimated from the snapshots.
The following functions implement finite difference estimators for the time
derivative of snapshots.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ddt
    ddt_nonuniform
    ddt_uniform


Kronecker Products
==================

The matrix-vector product
:math:`\Hhat[\qhat\otimes\qhat]`,
where :math:`\qhat\in\RR^{r}` and
:math:`\Hhat\in\RR^{r\times r^{2}}`,
can be represented more compactly as
:math:`\check{\H}[\qhat
\ \widehat{\otimes}\ \qhat]`
where $\check{\H}\in\RR^{r\times r(r+1)/2}$ and
:math:`\widehat{\otimes}` is a compressed version of the Kronecker product.
Specifically, if
:math:`\qhat = [\hat{q}_{1},\ldots,\hat{q}_{r}]\trp`,
then the full Kronecker product of :math:`\qhat` with itself is

.. math::
    \qhat\otimes\qhat
    = \left[\begin{array}{c}
        \hat{q}_{1}\qhat
        \\ \vdots \\
        \hat{q}_{r}\qhat
    \end{array}\right]
    =
    \left[\begin{array}{c}
        \hat{q}_{1}^{2} \\
        \hat{q}_{1}\hat{q}_{2} \\
        \vdots \\
        \hat{q}_{1}\hat{q}_{r} \\
        \hat{q}_{1}\hat{q}_{2} \\
        \hat{q}_{2}^{2} \\
        \vdots \\
        \hat{q}_{2}\hat{q}_{r} \\
        \vdots
        \hat{q}_{r}^{2}
    \end{array}\right] \in\RR^{r^{2}}.

The term :math:`\hat{q}_{1}\hat{q}_{2}` appears twice in the full Kronecker
product :math:`\qhat\otimes\qhat`.
The compressed Kronecker product is defined here as

.. math::
    \qhat\ \widehat{\otimes}\ \qhat
    = \left[\begin{array}{c}
        \hat{q}_{1}^2
        \\
        \hat{q}_{2}\qhat_{1:2}
        \\ \vdots \\
        \hat{q}_{r}\qhat_{1:r}
    \end{array}\right]
    = \left[\begin{array}{c}
        \hat{q}_{1}^2 \\
        \hat{q}_{1}\hat{q}_{2} \\ \hat{q}_{2}^{2} \\
        \\ \vdots \\ \hline
        \hat{q}_{1}\hat{q}_{r} \\ \hat{q}_{2}\hat{q}_{r}
        \\ \vdots \\ \hat{q}_{r}^{2}
    \end{array}\right]
    \in \RR^{r(r+1)/2},

where

.. math::
    \qhat_{1:i}
    &= \left[\begin{array}{c}
        \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
    \end{array}\right]\in\RR^{i}.

The following functions facilitate compressed Kronecker products of this type.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    compress_cubic
    compress_quadratic
    expand_cubic
    expand_quadratic
    kron2c
    kron2c_indices
    kron3c
    kron3c_indices


Load/Save HDF5 Utilities
========================

Many ``opinf`` classes have ``save()`` methods that export the object to an
HDF5 file and a ``load()`` class method for importing an object from an HDF5
file. The following functions facilitate that data transfer.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    hdf5_loadhandle
    hdf5_savehandle

"""

from ._hdf5 import *
from ._kronecker import *
from ._finite_difference import *
