# basis/__init__.py
r"""Tools for compressing snapshot data.

.. currentmodule:: opinf.basis

The purpose of learning a reduced-order model is to achieve a computational
speedup, which is a result of reducing the dimension of the state
:math:`\q(t)\in\RR^{n}` from :math:`n` to :math:`r \ll n`.
This is accomplished by introducing a low-dimensional approximation
:math:`\q(t) \approx \boldsymbol{\Gamma}(\qhat(t))`,
where :math:`\qhat(t)\in\RR^{r}`.
The following tools construct this approximation.

**Classes**

.. autosummary::
    :toctree: _autosummaries

    LinearBasis
    LinearBasisMulti
    PODBasis
    PODBasisMulti

**Functions**

.. autosummary::
    :toctree: _autosummaries

    cumulative_energy
    pod_basis
    projection_error
    residual_energy
    svdval_decay
"""

from ._linear import *
from ._pod import *
