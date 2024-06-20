# ddt/_base.py
"""Template for time derivative estimators."""

__all__ = [
    "DerivativeEstimatorTemplate",
]


import abc
import collections
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from .. import errors, utils


class DerivativeEstimatorTemplate(abc.ABC):
    r"""Template for time derivative estimators.

    Operator Inference for time-continuous (semi-discrete) models requires
    state snapshots and their time derivatives in order to learn operator
    entries via regression. This class is a template for estimating the first
    time derivative of state snapshots. Specifically, from a collection of
    snapshots :math:`\qhat_0,\ldots,\qhat_{k-1}\in\RR^r` representing the state
    at time instances :math:`t_0,\ldots,t_{k-1}`, the goal is to estimate

    .. math::
        \dot{\qhat}_j \approx \ddt\qhat(t)\bigg|_{t = t_j} \in \RR^{r}

    for :math:`j = 0, \ldots, k - 1`.

    Depending on the estimation strategy, the derivatives may only be computed
    for a subset of the states. For example, a first-order backward difference
    may omit an estimate for :math:`\dot{\qhat}_0`.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain of the snapshot data.
    """

    __tests = (
        (
            r"$f(t) = t$",
            lambda t: t,
            np.ones_like,
        ),
        (
            r"$f(t) = t^4 - \frac{1}{3}t^3$",
            lambda t: t**4 - (t**3 / 3),
            lambda t: 4 * t**3 - t**2,
        ),
        (
            r"$f(t) = \sin(t)$",
            np.sin,
            np.cos,
        ),
        (
            r"$f(t) = e^t$",
            np.exp,
            np.exp,
        ),
        (
            r"$f(t) = \frac{1}{1 + t}$",
            lambda t: 1 / (t + 1),
            lambda t: -1 / (t + 1) ** 2,
        ),
        (
            r"$f(t) = t - t^3 + \cos(t) - e^{t/2}$",
            lambda t: t - t**3 + np.cos(t) - np.exp(t / 2),
            lambda t: 1 - 3 * t**2 - np.sin(t) - np.exp(t / 2) / 2,
        ),
    )

    # Constructor -------------------------------------------------------------
    def __init__(self, time_domain):
        """Set the time domain."""
        if not isinstance(time_domain, np.ndarray) or time_domain.ndim != 1:
            raise ValueError("time_domain must be a one-dimensional array")
        self.__t = time_domain

    # Properties --------------------------------------------------------------
    @property
    def time_domain(self):
        """Time domain of the snapshot data, a (k,) ndarray."""
        return self.__t

    def __str__(self):
        """String representation: class name, time domain."""
        out = [self.__class__.__name__]
        t = self.time_domain
        out.append(f"time_domain: {t.size} entries in [{t.min()}, {t.max()}]")
        return "\n  ".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Main routine ------------------------------------------------------------
    def _check_dimensions(self, states, inputs, check_against_time=True):
        """Check dimensions and alignment of the state and inputs."""
        if states.ndim != 2:
            raise errors.DimensionalityError("states must be two-dimensional")
        if check_against_time and states.shape[-1] != self.time_domain.size:
            raise errors.DimensionalityError(
                "states not aligned with time_domain"
            )
        if inputs is not None:
            if inputs.ndim == 1:
                inputs = inputs.reshape((1, -1))
            if inputs.shape[1] != states.shape[1]:
                raise errors.DimensionalityError(
                    "states and inputs not aligned"
                )
        return states, inputs

    @abc.abstractmethod
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
            Subset of the state snapshots.
        ddts : (r, k') ndarray
            First time derivatives corresponding to ``_states``.
        _inputs : (m, k') ndarray or None
            Inputs corresponding to ``_states``, if applicable.
            **Only returned** if ``inputs`` is provided.
        """
        raise NotImplementedError  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify_shapes(self, r: int = 5, m: int = 3):
        """Verify that :meth:`estimate()` is consistent in the sense that the
        all outputs have the same number of columns. This method does **not**
        check the accuracy of the derivative estimation.

        Parameters
        ----------
        r : int
            State dimension to use in the check.
        m : int
            Number of inputs to use in the check.
            The ``inputs`` argument used to verify :meth:`estimate()` is
            a two-dimensional array of shape ``(m, k)`` even if ``m = 1``,
            where ``k`` is the size of the time domain.
        """
        # Get random data. What matters here is the size, not the entries.
        k = self.time_domain.size
        Q = np.random.random((r, k))
        U = np.random.random((m, k))

        # Call estimate() with inputs=None.
        outputs = self.estimate(Q)
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise errors.VerificationError(
                "len(estimate(states, inputs=None)) != 2"
            )

        _Q, _dQdt = outputs
        if _Q.shape[0] != r:
            raise errors.VerificationError(
                "estimate(states)[0].shape[0] != states.shape[0]"
            )
        if _dQdt.shape[0] != r:
            raise errors.VerificationError(
                "estimate(states)[1].shape[0] != states.shape[0]"
            )
        if _Q.shape[1] != _dQdt.shape[1]:
            print(_Q.shape[1], _dQdt.shape[1])
            raise errors.VerificationError(
                "Q.shape[1] != dQdt.shape[1] "
                "where Q, dQdt = estimate(states, inputs=None)"
            )

        # Call estimate() with non-None inputs.
        outputs = self.estimate(Q, U)
        if not isinstance(outputs, tuple) or len(outputs) != 3:
            raise errors.VerificationError(
                "len(estimate(states, inputs)) != 3"
            )

        _Q, _dQdt, _U = outputs
        if _U is None:
            raise errors.VerificationError(
                "estimates(states, inputs)[2] should not be None"
            )
        if _U.shape[0] != m:
            raise errors.VerificationError(
                "estimate(states, inputs)[2].shape[0] != inputs.shape[0]"
            )
        if _U.shape[1] != _Q.shape[1]:
            raise errors.VerificationError(
                "Q.shape[1] != U.shape[1] "
                "where Q, _, U = estimate(states, inputs)"
            )

        print("estimate() output shapes are consistent")

    def verify(self, plot: bool = False, return_errors=False):
        """Verify that :meth:`estimate()` is consistent in the sense that the
        all outputs have the same number of columns and test the accuracy of
        the results on a few test problems.

        Parameters
        ----------
        plot : bool
            If ``True``, plot the relative errors of the derivative estimation
            errors as a function of the time step.
            If ``False`` (default), print a report of the relative errors.
        return_errors : bool
            If ``True``, return the errors for each test as a dictionary.
            If ``False`` (default), return nothing.

        Returns
        -------
        errors : dict
            Estimation errors for each test case.
            Time steps are listed as ``errors[dts]``.
            **Only returned** if ``return_errors=True``.
        """
        self.verify_shapes()

        time_domain = self.time_domain  # Record original time domain.
        dts = np.logspace(-12, -1, 12)[::-1]
        estimation_errors = collections.defaultdict(list)
        estimation_errors["dts"] = dts

        t_base = np.arange(1001)
        for dt in dts:
            # Construct test cases.
            t = 1 + (dt * t_base)
            Q = np.array([test[1](t) for test in self.__tests])
            dQdt = np.array([test[2](t) for test in self.__tests])
            self.__t = t

            # Call the derivative estimator.
            Q_est, dQdt_est = self.estimate(Q, None)

            # Use new states to infer the indices of the returns.
            start = np.argmin(
                la.norm(Q - Q_est[:, 0].reshape((-1, 1))), axis=0
            )
            s = slice(start, start + Q_est.shape[1])
            truth = dQdt[:, s]

            # Calculate the relative error of the time derivative estimate.
            denom = la.norm(truth, axis=1)
            errs = la.norm(dQdt_est - truth, axis=1) / denom
            for test, err in zip(self.__tests, errs):
                estimation_errors[test[0]].append(err)

        if plot:
            _, ax = plt.subplots(1, 1)
            for test in self.__tests:
                name = test[0]
                ax.loglog(
                    dts,
                    estimation_errors[name],
                    ".-",
                    linewidth=0.5,
                    markersize=5,
                    label=name,
                )
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=max(ymin, 1e-14), top=min(ymax, 10))
            ax.set_xlabel("time step")
            ax.set_ylabel("relative error")
            ax.legend(ncols=2, frameon=False, fontsize="small")
            ax.set_title("Time derivative estimation errors")
        else:
            print(
                (title := "\nTime derivative estimation relative errors"),
                "=" * len(title),
                sep="\n",
            )
            for test in self.__tests:
                name = test[0]
                rawname = name.strip("$")
                print(f"\n{rawname}", len(rawname) * "-", sep="\n")
                for dt, err in zip(dts, estimation_errors[name]):
                    print(f"dt = {dt:.1e}:\terror = {err:.4e}")

        self.__t = time_domain  # Restore original time domain.
        if return_errors:
            return estimation_errors
