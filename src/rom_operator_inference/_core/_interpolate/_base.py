# _core/_interpolate/_base.py
"""Parametric ROM classes that use interpolation.

Classes
-------
* _InterpolatedMixin(_ParametricMixin)
"""

__all__ = []

import numpy as np
import scipy.interpolate as interp

from .._base import _ParametricMixin


class _InterpolatedMixin(_ParametricMixin):
    """Mixin class for interpolatory parametric reduced model classes."""
    @property
    def cs_(self):
        """The constant terms for each submodel."""
        return [m.c_ for m in self.models_] if self.has_constant else None

    @property
    def As_(self):
        """The linear state matrices for each submodel."""
        return [m.A_ for m in self.models_] if self.has_linear else None

    @property
    def Hs_(self):
        """The full quadratic state matrices for each submodel."""
        return [m.H_ for m in self.models_] if self.has_quadratic else None

    @property
    def Gs_(self):
        """The full cubic state matrices for each submodel."""
        return [m.G_ for m in self.models_] if self.has_cubic else None

    @property
    def Bs_(self):
        """The linear input matrices for each submodel."""
        return [m.B_ for m in self.models_] if self.has_inputs else None

    @property
    def fs_(self):
        """The reduced-order operators for each submodel."""
        return [m.f_ for m in self.models_]

    def __len__(self):
        """The number of trained models."""
        return len(self.models_) if hasattr(self, "models_") else 0


class _Interp2DMulti:
    def __init__(self, µs, z):
        """Construct linear 2D interpolators for each operator entry.

        Parameters
        ----------
        µs : (s,2) ndarray or list of s ndarrays
            Parameter samples (interpolation points).

        z : (s,...) ndarray or list of s ndarrays
            Inferred operators corresponding to the parameter samples
            (interpolation values)
        """
        # Validate and unpack training parameters.
        µs = np.array(µs)
        if µs.ndim != 2 or µs.shape[1] != 2:
            raise ValueError("parameter samples must be two-dimensional")
        s = µs.shape[0]
        µ1s, µ2s = µs[:,0], µs[:,1]

        # Construct a single interpolator for each operator entry.
        if len(z) != s:
            raise ValueError("unequal number of samples and values")
        self.shape_ = z[0].shape
        data = np.reshape(z, (s,-1))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("once")
            self.interpolators_ = [interp.interp2d(µ1s, µ2s, data[:,j])
                                   for j in range(data.shape[-1])]

    def __call__(self, µ):
        """Evaluate the interpolators and reshape the output appropriately.

        Parameters
        ----------
        µ : ndarray
            A single test parameter.

        Returns
        -------
        out: ndarray
            Interpolation evaluation of this operator at the test parameter.
        """
        # Validate and unpack test parameter.
        µ = np.array(µ)
        if µ.shape != (2,):
            raise ValueError("expected a single two-entry parameter")

        # Evaluate the interpolators at the test parameter.
        return np.reshape([float(f(µ[0], µ[1]))
                           for f in self.interpolators_], self.shape_)
