# lift/test_base.py
"""Tests for lift._base.py."""

import pytest
import numpy as np

import opinf


class TestLifterTemplate:
    """Test opinf.lift.LifterTemplate."""

    class Dummy(opinf.lift.LifterTemplate):
        """Instantiable version of LifterTemplate."""

        @staticmethod
        def lift(states):
            return states

        @staticmethod
        def unlift(lifted_states):
            return lifted_states

    def test_init(self, n=10, k=20):
        """Test that classes are instantiable after implementing only lift()
        and unlift().
        """
        self.Dummy()
        Q = np.random.random((n, k))
        assert self.Dummy.lift(Q) is Q
        assert self.Dummy.unlift(Q) is Q

    def test_verify(self, n=12, k=18):
        """Coverage tests for verify()."""
        Q = np.random.random((n, k))
        self.Dummy().verify(Q)
        t = np.linspace(0, 1, k)

        class Dummy2(self.Dummy):
            @staticmethod
            def lift(states):
                return states[:, : states.shape[1] // 2]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2().verify(Q)
        assert ex.value.args[0] == (
            f"{k//2} = lift(states).shape[1] != states.shape[1] = {k}"
        )

        class Dummy3(self.Dummy):
            @staticmethod
            def unlift(lifted_states):
                return lifted_states[:, : lifted_states.shape[1] // 2]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3().verify(Q)
        assert ex.value.args[0] == (
            f"{(n, k//2)} = unlift(lift(states)).shape "
            f"!= states.shape = {(n, k)}"
        )

        class Dummy4(self.Dummy):
            @staticmethod
            def unlift(lifted_states):
                return lifted_states - 1

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy4().verify(Q)
        assert ex.value.args[0] == "unlift(lift(states)) != states"

        class Dummy5(self.Dummy):
            @staticmethod
            def lift_ddts(states, ddts):
                return ddts[:, : states.shape[1] // 2]

        with pytest.raises(ValueError) as ex:
            Dummy5().verify(Q)
        assert ex.value.args[0] == (
            "time domain 't' required for finite difference check"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5().verify(Q, t)
        assert ex.value.args[0] == (
            f"{(n, k//2)} = lift_ddts(states, ddts).shape "
            f"!= lift(states).shape = {(n, k)}"
        )

        class Dummy6(self.Dummy):
            @staticmethod
            def lift_ddts(states, ddts):
                return ddts - 10000

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6().verify(Q, t, tol=0)
        assert ex.value.args[0].startswith(
            "lift_ddts() failed finite difference check"
        )
