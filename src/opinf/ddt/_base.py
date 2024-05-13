# ddt/_base.py
"""Template for time derivative estimators."""

__all__ = []


import abc
import collections
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from .. import errors


class _BaseDerivativeEstimator(abc.ABC):
    r"""Base class for time derivative estimators.

    Operator Inference for time-continuous (semi-discrete) models requires
    state snapshots and their time derivatives in order to learn operator
    entries via regression. This class is a base for all classes that
    estimate time derivatives of state snapshots.

    This class may be extended in the future for estimates of the _second_
    time derivative. For now, use :class:`DerivativeEstimatorTemplate` as a
    superclass of new estimators for the _first_ time derivative.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain of the snapshot data.
    """

    # Constructor -------------------------------------------------------------
    def __init__(self, time_domain):
        """Set the time domain."""
        self.time_domain = time_domain

    # Properties --------------------------------------------------------------
    @property
    def time_domain(self):
        """Time domain of the snapshot data, a (k,) ndarray."""
        return self.__t

    @time_domain.setter
    def time_domain(self, t):
        """Set the time domain."""
        self.__t = t

    # Main routine ------------------------------------------------------------
    @abc.abstractmethod
    def estimate(self, states, inputs=None):
        r"""Estimate the first time derivatives of the states.

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
    def verify(self, r: int = 5, m: int = 3):
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


class DerivativeEstimatorTemplate(_BaseDerivativeEstimator):
    r"""Template for time derivative estimators.

    Operator Inference for time-continuous (semi-discrete) models requires
    state snapshots and their time derivatives in order to learn operator
    entries via regression. This class is a template for estimating first time
    derivatives of state snapshots estimators. Specifically, from a collection
    of snapshots :math:`\q_0,\ldots,\q_{k-1}\in\RR^n` representing the state
    at time instances :math:`t_0,\ldots,t_{k-1}`, the goal is to estimate

    .. math::
        \dot{\q}_j = \ddt\q(t)\bigg|_{t = t_j}

    for :math:`j = 0, \ldots, k - 1`.

    Depending on the estimation scheme, the derivatives may only be computed
    for a subset of the states. For example, a first-order backward difference
    may omit an estimate for :math:`\dot{\q}_0`.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain of the snapshot data.
    """

    __tests = (
        (
            r"$f(t) = 1",
            np.ones_like,
            np.zeros_like,
        ),
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
            r"$f(t) = frac{1}{1 + t}$",
            lambda t: 1 / (t + 1),
            lambda t: -1 / (t + 1) ** 2,
        ),
        (
            r"$f(t) = t - t^3 + \cos(t) - e^{t/2}$",
            lambda t: t - t**3 + np.cos(t) - np.exp(t / 2),
            lambda t: 1 - 3 * t**2 - np.sin(t) - np.exp(t / 2) / 2,
        ),
    )

    def verify(self, plot: bool = False):
        """Verify that :meth:`estimate()` is consistent in the sense that the
        all outputs have the same number of columns and test the accuracy of
        the results on a few test problems.

        Parameters
        ----------
        plot : bool
            If ``True``, plot the relative errors of the derivative estimation
            errors as a function of the time step.
            If ``False`` (default), print a report of the relative errors.

        Returns
        -------
        errors : dict
            Estimation errors for each test case.
            Time steps are listed as ``errors[dts]``.
        """
        _BaseDerivativeEstimator.verify(self)

        time_domain = self.time_domain  # Record original time domain.
        dts = np.logspace(-12, -1, 12)[::-1]
        estimation_errors = collections.defaultdict(list)
        estimation_errors["dts"] = dts

        t_base = np.arange(1001)
        for dt in dts:
            # Construct test cases.
            t = 1 + (dt * t_base)
            Q = np.row_stack([test[1](t) for test in self.__tests])
            dQdt = np.row_stack([test[2](t) for test in self.__tests])
            self.time_domain = t

            # Call the derivative estimator.
            Q_est, dQdt_est = self.estimate(Q, None)

            # Use new states to infer the indices of the returns.
            start = np.argmin(
                la.norm(Q - Q_est[:, 0].reshape((-1, 1))), axis=0
            )
            s = slice(start, start + Q_est.shape[1])
            truth = dQdt[:, s]

            # Calculate the relative error of the time derivative estimate.
            kind = "relative"
            denom = la.norm(truth, axis=1)
            if np.any(denom < np.finfo(float).eps):
                denom = denom.max()
                kind = "absolute"
            errs = la.norm(dQdt_est - truth, axis=1) / denom
            for test, err in zip(self.__tests, errs):
                estimation_errors[test[0]].append(err)

        if plot:
            fig, ax = plt.subplots(1, 1)
            for test in self.__tests:
                name = test[0]
                ax.loglog(
                    dts,
                    estimation_errors[name],
                    ".-",
                    linewidth=0.5,
                    markersize=10,
                    label=name,
                )
                ax.legend(loc="lower right", frameon=False)
                ax.set_xlabel("time step")
                ax.set_ylabel(f"{kind} error")
                fig.suptitle("Time derivative estimation errors")
                plt.show()
        else:
            print(
                (title := f"Time derivative estimation ({kind}) errors"),
                "=" * len(title),
                sep="\n",
            )
            for test in self.__tests:
                name = test[0]
                rawname = name.strip("$")
                print(f"\n{rawname}", len(rawname) * "-", sep="\n")
                for dt, err in zip(dts, estimation_errors[name]):
                    print(f"dt = {dt:.1e}:\terror = {err:.4e}")

        self.time_domain = time_domain  # Restore original time domain.
        return estimation_errors
