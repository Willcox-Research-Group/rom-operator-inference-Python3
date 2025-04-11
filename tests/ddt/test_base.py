# ddt/test_base.py
"""Tests for ddt._base."""

import abc
import pytest
import numpy as np
import matplotlib.pyplot as plt

import opinf


class _TestDerivativeEstimatorTemplate:
    """Base class for classes that test time derivative estimators."""

    Estimator = NotImplemented  # Class to test.

    @abc.abstractmethod
    def get_estimators(self):
        """Yield estimators to test."""
        raise NotImplementedError

    def test_init(self, k=100):
        """Test __init__(), time_domain, __str__(), and __repr__()."""
        t = np.linspace(0, 1, k)
        t2D = np.zeros((2, 3, 3))

        with pytest.raises(ValueError) as ex:
            self.Estimator(t2D)
        assert ex.value.args[0] == (
            "time_domain must be a one-dimensional array"
        )

        estimator = self.Estimator(t)
        assert estimator.time_domain is t

        repr(estimator)

    def test_estimate(self, check_against_time: bool = True):
        """Use verify() to test estimate()."""
        for estimator in self.get_estimators():

            t_original = estimator.time_domain
            k = t_original.size

            # states must be two-dimensional.
            Q = np.random.random(k)
            with pytest.raises(opinf.errors.DimensionalityError) as ex:
                estimator.estimate(Q)
            assert ex.value.args[0] == "states must be two-dimensional"

            Q = np.random.random((2, k))
            if check_against_time:
                # states and time_domain must be aligned.
                with pytest.raises(opinf.errors.DimensionalityError) as ex:
                    estimator.estimate(Q[:, :-1])
                assert ex.value.args[0] == "states and time_domain not aligned"

            # states and inputs must be aligned.
            U = np.random.random((2, k))
            with pytest.raises(opinf.errors.DimensionalityError) as ex:
                estimator.estimate(Q, U[:, :-1])
            assert ex.value.args[0] == "states and inputs not aligned"

            # One-dimensional inputs.
            estimator.estimate(Q, U[0])

            # Test with verify().
            errors = estimator.verify(plot=False, return_errors=True)
            for label, results in errors.items():
                if label == "dts":
                    continue
                assert (
                    np.min(results) < 5e-7
                ), f"test '{label}' failed for estimator\n{estimator}"
            assert estimator.time_domain is t_original

            interactive = plt.isinteractive()
            plt.ion()
            errors = estimator.verify(plot=True)
            assert errors is None
            fig = plt.gcf()
            assert len(fig.axes) == 1
            plt.close(fig)

            if not interactive:
                plt.ioff()

            assert estimator.time_domain is t_original

    def test_mask(self):
        """Test mask()."""
        for estimator in self.get_estimators():
            k = estimator.time_domain.size
            Q1 = np.random.random((2, k))
            Q2 = np.random.random((2, k))

            Q1new, dQ = estimator.estimate(Q1)
            Q1mask = estimator.mask(Q1)
            assert np.all(Q1mask == Q1new)

            Q2mask = estimator.mask(Q2)
            assert Q2mask.shape == Q1new.shape


if __name__ == "__main__":
    pytest.main([__file__])
