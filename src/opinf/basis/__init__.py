# basis/__init__.py
r"""Tools for compressing snapshot data.

The purpose of learning a reduced-order model is to achieve a computational
speedup, which is a result of reducing the dimension of the state
:math:`\mathbf{q}(t)\in\mathbb{R}^{n}` from :math:`n` to :math:`r \ll n`.
This is accomplished by introducing a low-dimensional approximation
:math:`\mathbf{q}(t) \approx \boldsymbol{\Gamma}(\widehat{\mathbf{q}}(t))`,
where :math:`\widehat{\mathbf{q}}(t)\in\mathbb{R}^{r}`.
The following tools construct this approximation.

.. currentmodule:: opinf.basis

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
"""

from ._linear import *
from ._pod import *
