# utils/__init__.py
r"""Utility functions for Operator Inference.

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

The quadratic term
:math:`\widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]`,
where :math:`\widehat{\mathbf{H}}\in\mathbb{R}^{r\times r^{2}}`, can be
represented more compactly as
:math:`\check{\mathbf{H}}[\widehat{\mathbf{q}}
\ \widehat{\otimes}\ \widehat{\mathbf{q}}]`
where $\check{\mathbf{H}}\in\mathbb{R}^{r\times r(r+1)/2}$ and
:math:`\widehat{\otimes}` is a compressed version of the Kronecker product.
The following functions facilitate this compressed evaluation.

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
