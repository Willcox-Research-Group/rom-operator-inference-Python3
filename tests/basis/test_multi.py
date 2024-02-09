# basis/test_multi.py
"""Tests for basis._multi.py."""

import os
import pytest
import numpy as np

import opinf


class TestBasisMulti:
    """Tests for basis._base.BasisMulti."""

    Basis = opinf.basis.BasisMulti

    class Dummy(opinf.basis.BasisTemplate):
        def __init__(self):
            self.data = np.random.random(np.random.randint(1, 10, size=2))

        def fit(self, states):
            return self

        def compress(self, states):
            return states[: states.shape[0] // 2]

        def decompress(self, states_compressed, locs=None):
            states = np.concatenate((states_compressed, states_compressed))
            return states[locs] if locs is not None else states

    class Dummy2(Dummy):
        def __eq__(self, other):
            if self.data.shape != other.data.shape:
                return False
            return np.all(self.data == other.data)

        def save(self, savefile, overwrite=False):
            with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
                hf.create_dataset("data", data=self.data)

        @classmethod
        def load(cls, loadfile):
            dummy = cls()
            with opinf.utils.hdf5_loadhandle(loadfile) as hf:
                dummy.data = hf["data"][:]
            return dummy

    class Dummy3(Dummy2, opinf.basis._base._UnivarBasisMixin):
        def __init__(self, name=None):
            opinf.basis._base._UnivarBasisMixin.__init__(self, name)
            TestBasisMulti.Dummy2.__init__(self)

    def test_init(self):
        """Test BasisMulti.__init__(), bases, dimensions."""
        bases = [self.Dummy(), self.Dummy2(), self.Dummy3("third")]
        basis = self.Basis(bases)
        assert basis.num_variables == len(bases)
        assert hasattr(basis, "variable_names")
        for name in basis.variable_names:
            assert isinstance(name, str)
        assert basis.variable_names[-1] == "third"
        assert basis.full_state_dimension is None
        assert basis.reduced_state_dimension is None
        assert basis.reduced_state_dimensions is None

        for i, bs in enumerate(bases):
            assert basis.bases[i] is bs

        with pytest.raises(ValueError) as ex:
            basis.bases = bases[:-1]
        assert ex.value.args[0] == "len(bases) != num_variables"

        bases[0].full_state_dimension = 12
        bases[1].full_state_dimension = 15
        bases[2].full_state_dimension = 18

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Basis(bases)
        assert ex.value.args[0] == (
            "bases have inconsistent full_state_dimension"
        )

        bases[1].full_state_dimension = 12
        bases[2].full_state_dimension = 12
        basis = self.Basis(bases)
        assert basis.full_state_dimension == 36

        basis.full_state_dimension = 21
        assert basis.bases[-1].full_state_dimension == 7

        basis[0].reduced_state_dimension = 2
        basis[1].reduced_state_dimension = 3
        basis[2].reduced_state_dimension = 4

        basis = self.Basis(bases)
        assert basis.reduced_state_dimensions == (2, 3, 4)
        assert basis.reduced_state_dimension == 9
        assert basis.shape == (21, 9)

        basis.reduced_state_dimensions = [4, 2, 5]
        assert basis.bases[-1].reduced_state_dimension == 5
        assert basis.reduced_state_dimensions == (4, 2, 5)
        assert basis.reduced_state_dimension == 11

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis.reduced_state_dimensions = (20, 20, 20)
        assert wn[0].message.args[0] == (
            "reduced_state_dimension r = 60 > 21 = full_state_dimension n"
        )

    # Magic methods -----------------------------------------------------------
    def test_getitem(self):
        """Test BasisMulti.__getitem__()."""
        bases = [self.Dummy(), self.Dummy2(), self.Dummy()]
        basis = self.Basis(bases)
        for i, bs in enumerate(bases):
            assert basis[i] is bs

        basis.variable_names = "ABC"
        for i, name in enumerate(basis.variable_names):
            assert basis[name] is bases[i]

    def test_eq(self):
        """Test BasisMulti.__eq__()."""
        bases = [self.Dummy(), self.Dummy2(), self.Dummy3()]

        basis1 = self.Basis(bases)
        assert basis1 != 10

        basis2 = self.Basis(bases[:-1])
        assert not basis1 == basis2

        basis2 = self.Basis(bases[:-1] + [self.Dummy3()])
        basis2.bases[-1].data = basis1.bases[-1].data + 1
        assert basis1 != basis2

        basis2.bases[-1].data = basis1.bases[-1].data
        assert basis1 == basis2

    def test_str(self):
        """Test BasisMulti.__str__()."""
        bases = [self.Dummy(), self.Dummy2()]
        basis = self.Basis(bases)

        stringrep = str(basis)
        assert stringrep.startswith("2-variable BasisMulti\n")
        for bs in bases:
            assert str(bs) in stringrep

        # Quick repr() test.
        rep = repr(basis)
        assert stringrep in rep
        assert str(hex(id(basis))) in rep

    # Convenience methods -----------------------------------------------------
    def test_get_var(self, nx=10, k=5):
        """Test BasisMulti.get_var()."""
        bases = [self.Dummy(), self.Dummy2(), self.Dummy3()]
        basis = self.Basis(bases)
        basis.reduced_state_dimensions = (8, 6, 4)
        basis.full_state_dimension = (n := nx * len(bases))

        # Full-order state selection.
        q = np.random.random(n)
        qvar = basis.get_var(1, q)
        assert qvar.shape == (nx,)
        assert np.all(qvar == q[nx : 2 * nx])

        Q = np.random.random((n, k))
        Qvar = basis.get_var(2, Q)
        assert Qvar.shape == (nx, k)
        assert np.all(Qvar == Q[2 * nx : 3 * nx])

        # Reduced-order state selection.
        q = np.random.random(18)
        qvar = basis.get_var(0, q)
        assert qvar.shape == (8,)
        assert np.all(qvar == q[:8])
        qvar = basis.get_var(1, q)
        assert qvar.shape == (6,)
        assert np.all(qvar == q[8:14])
        qvar = basis.get_var(-1, q)
        assert qvar.shape == (4,)
        assert np.all(qvar == q[14:])

        Q = np.random.random((18, k))
        Qvar = basis.get_var(0, Q)
        assert Qvar.shape == (8, k)
        assert np.all(Qvar == Q[:8])
        Qvar = basis.get_var(1, Q)
        assert Qvar.shape == (6, k)
        assert np.all(Qvar == Q[8:14])
        Qvar = basis.get_var(-1, Q)
        assert Qvar.shape == (4, k)
        assert np.all(Qvar == Q[14:])

    def test_split(self, nx=10, k=5):
        """Test BasisMulti.split()."""
        bases = [self.Dummy(), self.Dummy2(), self.Dummy3()]
        basis = self.Basis(bases)
        basis.reduced_state_dimensions = (rs := (8, 6, 4))
        basis.full_state_dimension = (n := nx * len(bases))

        # Full-order splitting.
        q = np.random.random(n)
        qs = basis.split(q)
        assert len(qs) == 3
        for i, qvar in enumerate(qs):
            assert qvar.shape == (nx,)
            assert np.all(qvar == q[i * nx : (i + 1) * nx])

        # Reduced-order splitting
        q = np.random.random(sum(rs))
        qs = basis.split(q)
        assert len(qs) == len(rs)
        for i, qvar in enumerate(qs):
            assert qvar.shape == (rs[i],)
        assert np.all(qs[0] == q[: rs[0]])
        assert np.all(qs[-1] == q[-rs[-1] :])

    # Main routines -----------------------------------------------------------
    def test_mains(self, nx=50, k=400):
        """Use BasisMulti.verify() to run tests."""
        bases = [self.Dummy(), self.Dummy2()]
        n = len(bases) * nx
        Q = np.random.random((n, k))

        basis = self.Basis(bases)
        for method in "compress", "decompress":
            with pytest.raises(AttributeError) as ex:
                getattr(basis, method)(Q)
            assert ex.value.args[0] == ("basis not trained (call fit())")

        assert basis.fit(Q) is basis
        assert basis.full_state_dimension == n
        basis.verify(Q)

        for i in range(len(bases)):
            bases[i] = self.Dummy2()
        basis.fit(Q)
        t = np.linspace(0, 0.1, k)
        basis.verify(Q, t)

        with pytest.raises(Exception):
            basis.fit(100)
        assert basis.full_state_dimension == n

    # Persistence -------------------------------------------------------------
    def test_save(self):
        """Lightly test BasisMulti.save()."""
        target = "_savebasismultitest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        bases = [self.Dummy2(), self.Dummy2(), self.Dummy3()]
        basis = self.Basis(bases)
        basis.save(target)
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self):
        """Test BasisMulti.load()."""
        target = "_loadbasismultitest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Check that save() -> load() gives the same basis.
        bases = [self.Dummy2(), self.Dummy2(), self.Dummy3()]
        BCs = [obj.__class__ for obj in bases]
        basis_original = self.Basis(bases)

        basis_original.save(target)
        basis = self.Basis.load(target, BasisClasses=BCs)
        assert len(basis.bases) == len(bases)
        for i, bs in enumerate(bases):
            assert basis[i].__class__ is bs.__class__
            assert basis[i].data.shape == bs.data.shape
            assert np.all(basis[i].data == bs.data)
        assert basis.full_state_dimension is None

        basis_original.full_state_dimension = 4 * len(bases)
        basis_original.save(target, overwrite=True)
        basis = self.Basis.load(target, BasisClasses=BCs)
        assert (
            basis.full_state_dimension == basis_original.full_state_dimension
        )

        basis_original.reduced_state_dimensions == (rs := (10, 7, 3))
        basis_original.save(target, overwrite=True)
        basis = self.Basis.load(target, BasisClasses=BCs)
        assert basis.reduced_state_dimensions == rs

        os.remove(target)
