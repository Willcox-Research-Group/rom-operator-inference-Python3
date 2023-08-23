# pre/__init__.py
r"""Tools for preprocessing snapshot data, prior to compression.
See :ref:`the preprocessing guide <sec-preprocessing-guide>` for discussion
and examples.

.. currentmodule:: opinf.pre

Data Scaling
============

Raw dynamical systems data often need to be lightly preprocessed before use
in Operator Inference. The following tools enable centering/shifting and
scaling/nondimensionalization of snapshot data.

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
"""

from ._shiftscale import *


# Deprecations ================================================================
import warnings as _warnings
from ..utils import (
    ddt_uniform as _ddt_uniform,
    ddt_nonuniform as _ddt_nonuniform,
    ddt as _ddt,
)


def ddt_uniform(*args, **kwargs):
    _warnings.warn(DeprecationWarning,
                   "ddt_uniform() has been moved to the utils submodule"
                   " and will be removed from pre in a future release")
    return _ddt_uniform(*args, **kwargs)


def ddt_nonuniform(*args, **kwargs):
    _warnings.warn(DeprecationWarning,
                   "ddt_nonuniform() has been moved to the utils submodule"
                   " and will be removed from pre in a future release")
    return _ddt_nonuniform(*args, **kwargs)


def ddt(*args, **kwargs):
    _warnings.warn(DeprecationWarning,
                   "ddt() has been moved to the utils submodule"
                   " and will be removed from pre in a future release")
    return _ddt(*args, **kwargs)
