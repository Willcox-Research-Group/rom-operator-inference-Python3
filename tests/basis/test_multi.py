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
        def __init__(self, name=None):
            super().__init__(name)
            self.data = np.random.random(np.random.randint(1, 10, size=2))

        def fit(self, states):
            self.full_state_dimension = (n := states.shape[0])
            self.reduced_state_dimension = n // 2
            return self

        def compress(self, states):
            return states[: states.shape[0] // 2]

        def decompress(self, states_compressed, locs=None):
            states = np.concatenate(
                [
                    states_compressed,
                    np.zeros_like(states_compressed),
                ]
            )
            return states[locs] if locs is not None else states

    class Dummy2(Dummy):
        def __eq__(self, other):
            if self.data.shape != other.data.shape:
                return False
            return np.all(self.data == other.data)

        def save(self, savefile, overwrite=False):
            with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
                hf.create_dataset("data", data=self.data)
                if n := self.full_state_dimension:
                    hf.create_dataset("n", data=[n])
                if r := self.reduced_state_dimension:
                    hf.create_dataset("r", data=[r])

        @classmethod
        def load(cls, loadfile):
            dummy = cls()
            with opinf.utils.hdf5_loadhandle(loadfile) as hf:
                dummy.data = hf["data"][:]
                if "n" in hf:
                    dummy.full_state_dimension = hf["n"][0]
                if "r" in hf:
                    dummy.reduced_state_dimension = hf["r"][0]
            return dummy

    class Dummy3(Dummy2):
        pass

    def test_init(self):
        """Test BasisMulti.__init__(), bases, dimensions."""
        bases = [self.Dummy(), self.Dummy2(), self.Dummy3(name="third")]
        basis = self.Basis(bases)
        assert basis.num_variables == len(bases)
        assert hasattr(basis, "variable_names")
        for name in basis.variable_names:
            assert isinstance(name, str)
        assert basis.variable_names[-1] == "third"
        assert basis.full_state_dimension is None
        assert basis.reduced_state_dimension is None

        for i, bs in enumerate(bases):
            assert basis.bases[i] is bs

        bases[0].full_state_dimension = 12
        bases[1].full_state_dimension = 15
        assert basis.full_state_dimension is None
        bases[2].full_state_dimension = 18
        assert basis.full_variable_sizes == (12, 15, 18)
        assert basis.full_state_dimension == 45

        basis[0].reduced_state_dimension = 2
        basis[1].reduced_state_dimension = 3
        assert basis.reduced_state_dimension is None
        basis[2].reduced_state_dimension = 4
        assert basis.reduced_variable_sizes == (2, 3, 4)
        assert basis.reduced_state_dimension == 9
        assert basis.shape == (45, 9)

        assert len(basis) == len(bases)

        basis = self.Basis(bases, full_variable_sizes=(10, 11, 12))
        assert basis.bases[0].full_state_dimension == 10
        assert basis.bases[1].full_state_dimension == 11
        assert basis.bases[2].full_state_dimension == 12

        with pytest.raises(ValueError) as ex:
            self.Basis(bases, full_variable_sizes=(19, 23))
        assert ex.value.args[0] == "len(full_variable_sizes) != len(bases)"

        with pytest.raises(ValueError) as ex:
            self.Basis([])
        assert ex.value.args[0] == "at least one basis required"

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            self.Basis(bases[:1])
        assert wn[0].message.args[0] == "only one variable detected"

        class ExtraDummy:
            name = "nothing"

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            self.Basis([ExtraDummy(), ExtraDummy()])
        assert len(wn) == 2
        assert wn[0].message.args[0].startswith("bases[0] does not inherit")
        assert wn[1].message.args[0].startswith("bases[1] does not inherit")

    # Magic methods -----------------------------------------------------------
    def test_getitem(self):
        """Test BasisMulti.__getitem__()."""
        bases = [self.Dummy(), self.Dummy2(), self.Dummy()]
        basis = self.Basis(bases)
        for i, bs in enumerate(bases):
            assert basis[i] is bs

        for bs, name in zip(bases, "ABC"):
            bs.name = name
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
    def test_get_var(self, ns=(4, 5, 6), rs=(2, 3, 4), k=5):
        """Test BasisMulti.get_var()."""
        basis_A = self.Dummy(name="A")
        basis_B = self.Dummy(name="B")
        basis_C = self.Dummy(name="C")
        basis = self.Basis([basis_A, basis_B, basis_C])

        # No dimensions set.
        with pytest.raises(AttributeError) as ex:
            basis.get_var(0, None)
        assert ex.value.args[0] == "dimension attributes not set"

        basis_A.full_state_dimension = ns[0]
        basis_A.reduced_state_dimension = rs[0]
        basis_B.full_state_dimension = ns[1]
        basis_B.reduced_state_dimension = rs[1]
        basis_C.full_state_dimension = ns[2]
        basis_C.reduced_state_dimension = rs[2]

        # Full-order state selection.
        n = sum(ns)
        q = np.random.random(n)
        qvar = basis.get_var(1, q)
        assert qvar.shape == (ns[1],)
        assert np.all(qvar == q[ns[0] : -ns[2]])

        Q = np.random.random((n, k))
        Qvar = basis.get_var("C", Q)
        assert Qvar.shape == (ns[2], k)
        assert np.all(Qvar == Q[-ns[2] :])

        # Reduced-order state selection.
        r = sum(rs)
        q = np.random.random(r)
        qvar = basis.get_var("A", q)
        assert qvar.shape == (rs[0],)
        assert np.all(qvar == q[: rs[0]])

        Q = np.random.random((r, k))
        Qvar = basis.get_var(2, Q)
        assert Qvar.shape == (rs[2], k)
        assert np.all(Qvar == Q[-rs[2] :])

        # Bad dimensions.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            basis.get_var(2, np.random.random(n + 1))
        assert ex.value.args[0].startswith("states.shape[0] must be")

    def test_split(self, ns=(11, 13), rs=(5, 7), k=5):
        """Test BasisMulti.split()."""
        bases = [self.Dummy(), self.Dummy2()]
        basis = self.Basis(bases, ns)

        # Full-order splitting.
        q = np.random.random(sum(ns))
        qs = basis.split(q)
        assert len(qs) == 2
        for i, qvar in enumerate(qs):
            assert qvar.shape == (ns[i],)
        assert np.all(qs[0] == q[: ns[0]])
        assert np.all(qs[1] == q[ns[0] :])

        # Reduced-order splitting
        for bs, ri in zip(basis.bases, rs):
            bs.reduced_state_dimension = ri
        r = sum(rs)
        q = np.random.random(r)
        qs = basis.split(q)
        assert len(qs) == 2
        for i, qvar in enumerate(qs):
            assert qvar.shape == (rs[i],)
        assert np.all(qs[0] == q[: rs[0]])
        assert np.all(qs[1] == q[rs[0] :])

        # Bad dimensions.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            basis.split(np.random.random(r - 1))
        assert ex.value.args[0].startswith("states.shape[0] must be")

    # Main routines -----------------------------------------------------------
    def test_mains(self, ns=(16, 20), k=40):
        """Use BasisMulti.verify() to run tests."""
        basis = self.Basis([self.Dummy(), self.Dummy2()])
        n = sum(ns)
        Q = np.random.random((n, k))

        with pytest.raises(AttributeError) as ex:
            basis.compress(Q)
        assert ex.value.args[0] == "full_state_dimension not set, call fit()"

        with pytest.raises(AttributeError) as ex:
            basis.decompress(Q)
        assert ex.value.args[0] == (
            "reduced_state_dimension not set, call fit()"
        )

        # Assume equal full variable sizes.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            basis.fit(np.random.random((31, k)))
        assert ex.value.args[0] == (
            "len(states) must be evenly divisible by "
            "the number of variables n_q = 2"
        )

        print("EQUAL DIMS TEST")
        assert basis.fit(Q) is basis
        assert basis.full_state_dimension == n
        assert basis.full_variable_sizes[0] == (r := n // 2)
        assert basis.full_variable_sizes[1] == r
        assert basis.reduced_state_dimension == r
        assert basis.reduced_variable_sizes[0] == (r2 := r // 2)
        assert basis.reduced_variable_sizes[1] == r2
        basis.verify()

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            basis.compress(Q[1:, :])
        assert ex.value.args[0] == (
            f"states.shape[0] = {n - 1} != {n} = full_state_dimension"
        )

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            basis.decompress(Q[: r - 1, :])
        assert ex.value.args[0] == (
            f"states_compressed.shape[0] = {r - 1} "
            f"!= {r} = reduced_state_dimension"
        )

        # Nonequal full variable sizes.
        print("NONEQUAL DIMS TEST")
        basis[0].full_state_dimension = ns[0]
        basis[1].full_state_dimension = ns[1]
        assert basis.fit(Q) is basis
        assert basis[0].reduced_state_dimension == (ns[0] // 2)
        assert basis[1].reduced_state_dimension == (ns[1] // 2)
        basis.verify()

        Q_ = basis.compress(Q)
        with pytest.raises(ValueError) as ex:
            basis.decompress(Q_, locs=np.array([0, 2], dtype=int))
        assert ex.value.args[0].startswith("'locs != None' requires that")

        # Make sure calling fit() incorrectly doesn't wipe old data.
        with pytest.raises(Exception):
            basis.fit(100)
        assert basis.full_state_dimension == n

    # Persistence -------------------------------------------------------------
    def test_save(self):
        """Lightly test BasisMulti.save()."""
        target = "_savebasismultitest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        basis = self.Basis(
            [self.Dummy2(name="testbasis"), self.Dummy2(), self.Dummy3()]
        )
        basis[1].full_state_dimension = 20
        basis[2].reduced_state_dimension = 10
        basis.save(target)
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self):
        """Test BasisMulti.load()."""
        target = "_loadbasismultitest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Check that save() -> load() gives the same basis.
        basis_original = self.Basis(
            [self.Dummy2(), self.Dummy2(), self.Dummy3()]
        )
        BCs = [obj.__class__ for obj in basis_original.bases]

        basis_original.save(target)
        basis = self.Basis.load(target, BasisClasses=BCs)
        assert len(basis) == len(basis_original)
        for i, bs in enumerate(basis_original.bases):
            assert basis[i].__class__ is bs.__class__
            assert basis[i].data.shape == bs.data.shape
            assert np.all(basis[i].data == bs.data)
        assert basis.full_state_dimension is None

        basis_original[0].full_state_dimension = 4
        basis_original[1].full_state_dimension = 5
        basis_original[2].full_state_dimension = 7
        basis_original.save(target, overwrite=True)
        basis = self.Basis.load(target, BasisClasses=BCs)
        assert len(basis) == len(basis_original)
        for bs1, bs2 in zip(basis_original.bases, basis.bases):
            assert bs2.full_state_dimension == bs1.full_state_dimension
        assert (
            basis.full_state_dimension == basis_original.full_state_dimension
        )

        basis_original.bases = basis_original.bases[:-1] + (self.Dummy2(),)
        basis_original.bases[-1].full_state_dimension = 10
        basis_original[0].reduced_state_dimension = 2
        basis_original[1].reduced_state_dimension = 2
        basis_original[2].reduced_state_dimension = 3
        basis_original.save(target, overwrite=True)
        basis = self.Basis.load(target, BasisClasses=self.Dummy2)
        for bs1, bs2 in zip(basis_original.bases, basis.bases):
            assert bs2.full_state_dimension == bs1.full_state_dimension
            assert bs2.reduced_state_dimension == bs1.reduced_state_dimension
        assert (
            basis.full_state_dimension == basis_original.full_state_dimension
        )
        assert (
            basis.reduced_state_dimension
            == basis_original.reduced_state_dimension
        )

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Basis.load(target, BasisClasses=[self.Dummy2, self.Dummy])
        assert ex.value.args[0] == (
            "file contains 3 bases but 2 classes provided"
        )

        os.remove(target)
