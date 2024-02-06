# basis/test_base.py
"""Tests for basis._base."""

import pytest
import numpy as np

import opinf


class TestBaseBasis:
    """Test basis._base.BasisTemplate."""

    class Dummy(opinf.basis.BasisTemplate):
        """Instantiable version of BasisTemplate."""

        def fit(self):
            pass

        def compress(self, states):
            return states + 2

        def decompress(self, states, locs=None):
            return states - 1

    def test_project(self, q=5):
        """Test BasisTemplate.project() and projection_error()."""
        basis = self.Dummy()
        assert basis.project(q) == (q + 1)
        assert basis.projection_error(q, relative=False) == 1
        assert basis.projection_error(q, relative=True) == 1 / q

    def test_verify(self, n=20, k=21):
        """Test BasisTemplate.verify()."""
        Q = np.random.random((n, k))

        basis = self.Dummy()
        with pytest.raises(ValueError) as ex:
            basis.verify(Q[0])
        assert ex.value.args[0] == (
            "two-dimensional states required for verification"
        )

        class Dummy2a(self.Dummy):
            def compress(self, states):
                return states[:, :-1]

        class Dummy2b(self.Dummy):
            def compress(self, states):
                if states.ndim == 1:
                    return 0
                return states

        basis = Dummy2a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify(Q)
        assert ex.value.args[0] == (
            "compress(states).shape[1] != states.shape[1]"
        )

        basis = Dummy2b()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify(Q)
        assert ex.value.args[0] == "compress(single_state_vector).ndim != 1"

        class Dummy3a(self.Dummy):
            def decompress(self, states, locs=None):
                return states[:-1, 1:]

        class Dummy3b(self.Dummy):
            def decompress(self, states, locs=None):
                if states.ndim == 1:
                    return 100
                return states

        basis = Dummy3a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify(Q)
        assert ex.value.args[0] == (
            "decompress(compress(states)).shape != states.shape"
        )

        basis = Dummy3b()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify(Q)
        assert ex.value.args[0] == (
            "decompress(compress(single_state_vector)).ndim != 1"
        )

        class Dummy4a(self.Dummy):
            pass

        class Dummy4b(self.Dummy):
            def decompress(self, states, locs=None):
                if locs is not None:
                    return states[locs] + 1
                return states

        basis = Dummy4a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify(Q)
        assert ex.value.args[0] == (
            "decompress(states_compressed, locs).shape "
            "!= decompress(states_compressed)[locs].shape"
        )

        basis = Dummy4b()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify(Q)
        assert ex.value.args[0] == (
            "decompress(states_compressed, locs) "
            "!= decompress(states_compressed)[locs]"
        )

        class Dummy5a(self.Dummy):
            def compress(self, states):
                return states

            def decompress(self, states, locs=None):
                return (states if locs is None else states[locs]) + 1

        class Dummy5b(Dummy5a):
            def compress(self, states):
                return states - 1

        basis = Dummy5a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify(Q)
        assert ex.value.args[0] == (
            "project(project(states)) != project(states)"
        )

        basis = Dummy5b()
        basis.verify(Q)


class TestUnivarBasisMixin:
    Mixin = opinf.basis._base._UnivarBasisMixin

    def test_state_dimension(self):
        """Test _UnivarBasisMixin.reduced_state_dimension."""
        mixin = self.Mixin("thename")
        assert mixin.full_state_dimension is None
        assert mixin.reduced_state_dimension is None
        assert mixin.name == "thename"

        mixin.full_state_dimension = 10.0
        n = mixin.full_state_dimension
        assert isinstance(n, int)
        assert mixin.full_state_dimension == n
        assert mixin.shape is None

        mixin.reduced_state_dimension = 4.0
        r = mixin.reduced_state_dimension
        assert isinstance(r, int)
        assert mixin.reduced_state_dimension == r
        assert mixin.shape == (n, r)

        mixin.full_state_dimension = None
        assert mixin.full_state_dimension is None

        mixin.reduced_state_dimension = None
        assert mixin.reduced_state_dimension is None
