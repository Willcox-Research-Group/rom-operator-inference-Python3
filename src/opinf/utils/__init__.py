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
:math:`\widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]`,
where :math:`\widehat{\mathbf{q}}\in\mathbb{R}^{r}` and
:math:`\widehat{\mathbf{H}}\in\mathbb{R}^{r\times r^{2}}`,
can be represented more compactly as
:math:`\check{\mathbf{H}}[\widehat{\mathbf{q}}
\ \widehat{\otimes}\ \widehat{\mathbf{q}}]`
where $\check{\mathbf{H}}\in\mathbb{R}^{r\times r(r+1)/2}$ and
:math:`\widehat{\otimes}` is a compressed version of the Kronecker product.
Specifically, if
:math:`\widehat{\mathbf{q}} = [\hat{q}_{1},\ldots,\hat{q}_{r}]^{\mathsf{T}}`,
then the full Kronecker product of :math:`\widehat{\mathbf{q}}` with itself is

.. math::
    \widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}
    = \left[\begin{array}{c}
        \hat{q}_{1}\widehat{\mathbf{q}}
        \\ \vdots \\
        \hat{q}_{r}\widehat{\mathbf{q}}
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
    \end{array}\right] \in\mathbb{R}^{r^{2}}.

The term :math:`\hat{q}_{1}\hat{q}_{2}` appears twice in the full Kronecker
product :math:`\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}`.
The compressed Kronecker product is defined here as

.. math::
    \widehat{\mathbf{q}}\ \widehat{\otimes}\ \widehat{\mathbf{q}}
    = \left[\begin{array}{c}
        \hat{q}_{1}^2
        \\
        \hat{q}_{2}\widehat{\mathbf{q}}_{1:2}
        \\ \vdots \\
        \hat{q}_{r}\widehat{\mathbf{q}}_{1:r}
    \end{array}\right]
    = \left[\begin{array}{c}
        \hat{q}_{1}^2 \\
        \hat{q}_{1}\hat{q}_{2} \\ \hat{q}_{2}^{2} \\
        \\ \vdots \\ \hline
        \hat{q}_{1}\hat{q}_{r} \\ \hat{q}_{2}\hat{q}_{r}
        \\ \vdots \\ \hat{q}_{r}^{2}
    \end{array}\right]
    \in \mathbb{R}^{r(r+1)/2},

where

.. math::
    \widehat{\mathbf{q}}_{1:i}
    &= \left[\begin{array}{c}
        \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
    \end{array}\right]\in\mathbb{R}^{i}.

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
