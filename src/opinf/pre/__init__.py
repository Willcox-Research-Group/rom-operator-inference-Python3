# pre/__init__.py
r"""Tools for preprocessing and compressing snapshot data.
See :ref:`the preprocessing guide <sec-preprocessing-guide>` for discussion and examples.

.. currentmodule:: opinf.pre

Data Scaling
============

Raw dynamical systems data often need to be lightly preprocessed before use in Operator Inference.
The following tools enable centering/shifting and scaling/nondimensionalization of snapshot data.

**Classes**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    SnapshotTransformer
    SnapshotTransformerMulti

**Functions**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    scale
    shift


Data Compression
================

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of reducing the dimension of the state :math:`\mathbf{q}(t)\in\mathbb{R}^{n}` from :math:`n` to :math:`r \ll n`.
This is accomplished by introducing a low-dimensional approximation :math:`\mathbf{q}(t) \approx \boldsymbol{\Gamma}(\widehat{\mathbf{q}}(t))`, where :math:`\widehat{\mathbf{q}}(t)\in\mathbb{R}^{r}`.
The following tools construct this approximation.

.. currentmodule:: opinf.pre

**Classes**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    LinearBasis
    LinearBasisMulti
    PODBasis
    PODBasisMulti

**Functions**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    cumulative_energy
    pod_basis
    projection_error
    residual_energy
    svdval_decay


Derivative Estimation
=====================

For time-continuous reduced-order models, Operator Inference requires the time derivative of the state snapshots.
If they are not available from a full-order solver, the time derivatives can often be estimated from the snapshots.
The following functions implement finite difference estimators for the time derivative of snapshots.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ddt
    ddt_nonuniform
    ddt_uniform
"""

from .basis import *
from .transform import *
from ._reprojection import *
from ._finite_difference import *
