# _interpolate/inferred.py
"""Parametric ROM classes that use interpolation.

Classes
-------
* _InterpolatedMixin(_ParametricMixin)
"""

__all__ = []

import numpy as np

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
    def Hcs_(self):
        """The compact quadratic state matrices for each submodel."""
        return [m.Hc_ for m in self.models_] if self.has_quadratic else None

    @property
    def Gs_(self):
        """The full cubic state matrices for each submodel."""
        return [m.G_ for m in self.models_] if self.has_cubic else None

    @property
    def Gcs_(self):
        """The compact cubic state matrices for each submodel."""
        return [m.Gc_ for m in self.models_] if self.has_cubic else None

    @property
    def Bs_(self):
        """The linear input matrices for each submodel."""
        return [m.B_ for m in self.models_] if self.has_inputs else None

    @property
    def fs_(self):
        """The reduced-order operators for each submodel."""
        return [m.f_ for m in self.models_]

    @property
    def dataconds_(self):
        """The condition numbers of the raw data matrices for each submodel."""
        return np.array([m.datacond_ for m in self.models_])

    @property
    def dataregconds_(self):
        """The condition numbers of the regularized data matrices for each
        submodel.
        """
        return np.array([m.dataregcond_ for m in self.models_])

    @property
    def residuals_(self):
        """The regularized least-squares residuals for each submodel."""
        return np.array([m.residual_ for m in self.models_])

    @property
    def misfits_(self):
        """The (nonregularized) least-squares data misfits for each
        submodel.
        """
        return np.array([m.misfit_ for m in self.models_])

    def __len__(self):
        """The number of trained models."""
        return len(self.models_) if hasattr(self, "models_") else 0
