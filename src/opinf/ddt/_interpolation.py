# ddt/_interpolation.py
"""Time derivative estimators based on interpolation."""

__all__ = [
    "InterpolationDerivativeEstimator",
]


import types
import warnings
import numpy as np
import scipy.interpolate as interp

from .. import errors
from ._base import DerivativeEstimatorTemplate


class InterpolationDerivativeEstimator(DerivativeEstimatorTemplate):
    r"""Time derivative estimator based on interpolation.

    For a set of (compressed) snapshots
    :math:`\Qhat = [~\qhat_0~~\cdots~~\qhat_{k-1}~] \in \RR^{r \times k}`,
    this class forms one-dimensional differentiable interpolants for each of
    the :math:`r` rows of :math:`\Qhat`.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain of the snapshot data.
    InterpolatorClass : str or scipy.interpolate class
        Class for performing the interpolation. Must obey the following syntax:

            >>> intrp = InterpolatorClass(time_domain, states, **options)
            >>> new_states = intrp(new_time_domain, 0)
            >>> state_ddts = intrp(new_time_domain, 1)

        The following strings are also accepted.

        * ``"cubic"`` (default): use :class:`scipy.interpolate.CubicSpline`.
        * ``"akima"``: use :class:`scipy.interpolate.Akima1DInterpolator`.
          This is a local interpolation method and is more resitant to
          outliers than :class:`scipy.interpolate.CubicSpline`. However, it is
          not recommended if the time points are not uniformly spaced.
    new_time_domain : (k',) ndarray or None
        If given, evaluate the interpolator at these points to generate new
        state snapshots and corresponding time derivatives. If input snapshots
        are also given, interpolate them as well and evaluate the interpolant
        at these new points. The `new_time_domain` should lie within the range
        of the original `time_domain`; a warning is raised if extrapolation is
        requested.
    options : dict
        Keyword arguments for the constructor of the :attr:`InterpolatorClass`.
    """

    _interpolators = types.MappingProxyType(
        {
            "cubic": interp.CubicSpline,
            "akima": interp.Akima1DInterpolator,
        }
    )

    # Constructor -------------------------------------------------------------
    def __init__(
        self,
        time_domain: np.ndarray,
        InterpolatorClass="cubic",
        new_time_domain: np.ndarray = None,
        **options,
    ):
        """Set the time domain, InterpolatorClass, and options."""
        DerivativeEstimatorTemplate.__init__(self, time_domain)

        # Set the InterpolatorClass.
        if not isinstance(InterpolatorClass, type):
            if InterpolatorClass not in self._interpolators:
                options = ", ".join([repr(k) for k in self._interpolators])
                raise ValueError(
                    f"invalid InterpolatorClass '{InterpolatorClass}', "
                    f"options: {options}"
                )
            InterpolatorClass = self._interpolators[InterpolatorClass]
            options["axis"] = 1
        self.__IC = InterpolatorClass

        # Set the new_time_domain.
        if (t2 := new_time_domain) is not None:
            if not isinstance(t2, np.ndarray) or t2.ndim != 1:
                raise TypeError(
                    "new_time_domain must be a one-dimensional array or None"
                )
            t1 = time_domain
            if t2.min() < t1.min() or t2.max() > t1.max():
                warnings.warn(
                    "new_time_domain extrapolates beyond time_domain",
                    errors.OpInfWarning,
                )
        self.__t2 = new_time_domain

        # Set the interpolator constructor options.
        self.__options = types.MappingProxyType(options)

    # Properties --------------------------------------------------------------
    def __str__(self):
        """String representation: class name, time domain."""
        head = DerivativeEstimatorTemplate.__str__(self)
        options = ", ".join(
            [
                f"{key}={repr(value)}"
                for key, value in self.options.items()
                if key != "axis"
            ]
        )
        tail = [
            f"InterpolatorClass: {self.InterpolatorClass.__name__}({options})"
        ]
        if (t2 := self.new_time_domain) is not None:
            tail.append(
                f"new_time_domain: {t2.size} entries "
                f"in [{t2.min()}, {t2.max()}]"
            )
        return f"{head}\n  " + "\n  ".join(tail)

    @property
    def InterpolatorClass(self):
        """One-dimensional differentiable interpolator class."""
        return self.__IC

    @property
    def new_time_domain(self):
        """Time domain at which to evaluate the interpolator and its
        first derivative.
        """
        return self.__t2

    @property
    def options(self):
        """Keyword arguments for the constructor of the
        :attr:`InterpolatorClass`.
        """
        return self.__options

    # Main routine ------------------------------------------------------------
    def estimate(self, states, inputs=None):
        """Estimate the first time derivatives of the states.

        Parameters
        ----------
        states : (r, k) ndarray
            State snapshots, either full or (preferably) reduced.
        inputs : (m, k) ndarray or None
            Inputs corresponding to the state snapshots, if applicable.

        Returns
        -------
        _states : (r, k') ndarray
            State snapshots or state estimates at the :attr:`new_time_domain`.
        ddts : (r, k') ndarray
            First time derivatives corresponding to ``_states``.
        _inputs : (m, k') ndarray or None
            Inputs corresponding to ``_states``, if applicable.
            **Only returned** if ``inputs`` is provided.
        """
        states, inputs = self._check_dimensions(states, inputs)

        statespline = self.InterpolatorClass(
            self.time_domain,
            states,
            **self.options,
        )

        t = self.time_domain
        if (t2 := self.new_time_domain) is not None:
            states = statespline(t2)
            if inputs is not None:
                inputspline = self.InterpolatorClass(
                    t,
                    inputs,
                    **self.options,
                )
                inputs = inputspline(t2)
            t = self.new_time_domain

        ddts = statespline(t, 1)

        if inputs is None:
            return states, ddts
        return states, ddts, inputs
