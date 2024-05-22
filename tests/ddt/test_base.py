# ddt/test_base.py
"""Tests for ddt._base."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

import opinf


_module = opinf.ddt._base


class TestDerivativeEstimatorTemplate:
    """Test _base.DerivativeEstimatorTemplate."""

    Base = _module.DerivativeEstimatorTemplate

    class Dummy(Base):
        """Instantiable version of DerivativeEstimatorTemplate."""

        def estimate(self, states, inputs=None):
            if inputs is not None:
                return states, states, inputs
            return states, states

    def test_init(self, k=100):
        """Test __init__() and time_domain."""
        t = np.linspace(0, 1, k)
        dummy = self.Dummy(t)
        assert dummy.time_domain is t

    def test_verify_shapes(self, k=100):
        """Test verify_shapes()."""
        t = np.linspace(0, 1, k)

        # Bad number of outputs with inputs=None.

        class Dummy1(self.Dummy):
            def estimate(self, states, inputs=None):
                return states, states, inputs

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy1(t).verify_shapes()
        assert ex.value.args[0] == "len(estimate(states, inputs=None)) != 2"

        # Misaligned output shapes with inputs=None.

        class Dummy2(self.Dummy):
            def estimate(self, states, inputs=None):
                return states[1:], states

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2(t).verify_shapes()
        assert ex.value.args[0] == (
            "estimate(states)[0].shape[0] != states.shape[0]"
        )

        class Dummy3(self.Dummy):
            def estimate(self, states, inputs=None):
                return states, states[1:]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3(t).verify_shapes()
        assert ex.value.args[0] == (
            "estimate(states)[1].shape[0] != states.shape[0]"
        )

        class Dummy4(self.Dummy):
            def estimate(self, states, inputs=None):
                return states[:, :-1], states

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy4(t).verify_shapes()
        assert ex.value.args[0] == (
            "Q.shape[1] != dQdt.shape[1] "
            "where Q, dQdt = estimate(states, inputs=None)"
        )

        # Bad number of outputs with inputs != None

        class Dummy5(self.Dummy):
            def estimate(self, states, inputs=None):
                return states, states

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5(t).verify_shapes()
        assert ex.value.args[0] == "len(estimate(states, inputs)) != 3"

        # Misaligned output shapes with inputs != None.

        class Dummy6(self.Dummy):
            def estimate(self, states, inputs=None):
                if inputs is None:
                    return states, states
                return states, states, None

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6(t).verify_shapes()
        assert ex.value.args[0] == (
            "estimates(states, inputs)[2] should not be None"
        )

        class Dummy7(self.Dummy):
            def estimate(self, states, inputs=None):
                if inputs is None:
                    return states, states
                return states, states, inputs[1:]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy7(t).verify_shapes()
        assert ex.value.args[0] == (
            "estimate(states, inputs)[2].shape[0] != inputs.shape[0]"
        )

        class Dummy8(self.Dummy):
            def estimate(self, states, inputs=None):
                if inputs is None:
                    return states, states
                return states, states, inputs[:, :-1]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy8(t).verify_shapes()
        assert ex.value.args[0] == (
            "Q.shape[1] != U.shape[1] where Q, _, U = estimate(states, inputs)"
        )

    def test_verify(self, k=100):
        """Lilghtly test verify()."""
        t = np.sort(np.random.random(k))
        dummy = self.Dummy(t)

        # No plotting.
        errors = dummy.verify(plot=False, return_errors=True)
        num_tests = len(errors["dts"])
        for dataset in errors.values():
            assert len(dataset) == num_tests

        # Plotting.
        interactive = plt.isinteractive()
        plt.ion()
        errors = dummy.verify(plot=True)
        assert errors is None
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close(fig)

        if not interactive:
            plt.ioff()

        # Check that the original time domain was restored.
        assert dummy.time_domain is t
