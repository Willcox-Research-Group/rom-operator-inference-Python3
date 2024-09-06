# basis/test_base.py
"""Tests for basis._base."""

import pytest

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

    def test_state_dimensions(self):
        """Test BasisTemplate.__init__(), name, and dimension properties."""
        basis = self.Dummy("thename")
        assert basis.full_state_dimension is None
        assert basis.reduced_state_dimension is None
        assert basis.name == "thename"
        basis.name = "newname"
        assert basis.name == "newname"

        basis.full_state_dimension = 10.0
        n = basis.full_state_dimension
        assert isinstance(n, int)
        assert basis.full_state_dimension == n

        basis.reduced_state_dimension = 4.0
        r = basis.reduced_state_dimension
        assert isinstance(r, int)
        assert basis.reduced_state_dimension == r
        assert basis.shape == (n, r)

        basis.full_state_dimension = None
        assert basis.full_state_dimension is None

        basis.reduced_state_dimension = None
        assert basis.reduced_state_dimension is None

    def test_str(self):
        """Lightly test __str__() and __repr__()."""

        basis = self.Dummy()
        str(basis)

        basis.full_state_dimension = 10
        str(basis)

        basis.name = "varname"
        basis.reduced_state_dimension = 5
        assert repr(basis).count(str(basis)) == 1

    def test_project(self, q=5):
        """Test BasisTemplate.project() and projection_error()."""
        basis = self.Dummy()
        assert basis.project(q) == (q + 1)
        assert basis.projection_error(q, relative=False) == 1
        assert basis.projection_error(q, relative=True) == 1 / q

    def test_verify(self, n=20, k=21):
        """Test BasisTemplate.verify()."""

        dummy = self.Dummy()

        with pytest.raises(AttributeError) as ex:
            dummy.verify()
        assert ex.value.args[0] == "basis not trained, call fit()"

        class Dummy1(self.Dummy):
            def __init__(self, name=None):
                super().__init__(name)
                self.full_state_dimension = 20
                self.reduced_state_dimension = 5

        class Dummy2a(Dummy1):
            def compress(self, states):
                return states[:, :-1]

        class Dummy2b(Dummy1):
            def compress(self, states):
                if states.ndim == 1:
                    return 0
                return states

        basis = Dummy2a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify()
        assert ex.value.args[0] == (
            "compress(states).shape[1] != states.shape[1]"
        )

        basis = Dummy2b()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify()
        assert ex.value.args[0] == "compress(single_state_vector).ndim != 1"

        class Dummy3a(Dummy1):
            def decompress(self, states, locs=None):
                return states[:-1, 1:]

        class Dummy3b(Dummy1):
            def decompress(self, states, locs=None):
                if states.ndim == 1:
                    return 100
                return states

        basis = Dummy3a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify()
        assert ex.value.args[0] == (
            "decompress(compress(states)).shape != states.shape"
        )

        basis = Dummy3b()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify()
        assert ex.value.args[0] == (
            "decompress(compress(single_state_vector)).ndim != 1"
        )

        class Dummy4a(Dummy1):
            pass

        class Dummy4b(Dummy1):
            def decompress(self, states, locs=None):
                if locs is not None:
                    return states[locs] + 1
                return states

        basis = Dummy4a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify()
        assert ex.value.args[0] == (
            "decompress(states_compressed, locs).shape "
            "!= decompress(states_compressed)[locs].shape"
        )

        basis = Dummy4b()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify()
        assert ex.value.args[0] == (
            "decompress(states_compressed, locs) "
            "!= decompress(states_compressed)[locs]"
        )

        class Dummy5a(Dummy1):
            def compress(self, states):
                return states

            def decompress(self, states, locs=None):
                return (states if locs is None else states[locs]) + 1

        class Dummy5b(Dummy5a):
            def compress(self, states):
                return states - 1

        basis = Dummy5a()
        with pytest.raises(opinf.errors.VerificationError) as ex:
            basis.verify()
        assert ex.value.args[0] == (
            "project(project(states)) != project(states)"
        )

        basis = Dummy5b()
        basis.verify()
