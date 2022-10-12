# pre/basis/test_linear.py
"""Tests for rom_operator_inference.pre.basis._linear."""

import os
import h5py
import pytest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

import rom_operator_inference as opinf


class DummyTransformer:
    """Dummy object with all required transformer methods."""
    def transform(self, states):
        return states + 1

    def fit_transform(self, states):
        return self.transform(states)

    def inverse_transform(self, states):
        return states - 1

    def save(self, hf):
        hf.create_dataset("dummy", data=(1, 1, 1))

    def __str__(self):
        return "DummyTransformer"


class TestLinearBasis:
    """Test pre.basis._linear._LinearBasis."""
    LinearBasis = opinf.pre.basis.LinearBasis

    def test_init(self):
        """Test LinearBasis.__init__() and entries/transformer properties."""
        basis = self.LinearBasis()
        assert basis.entries is None
        assert basis.shape is None
        assert basis.r is None

        with pytest.raises(TypeError) as ex:
            basis.fit(10)
        assert ex.value.args[0].startswith("invalid basis")

        Vr = np.random.random((10, 3))
        basis.fit(Vr)
        assert basis.entries is Vr
        assert np.all(basis[:] == Vr)
        assert basis.shape == (10, 3)
        assert basis.r == 3

        out = basis.fit(Vr + 1)
        assert out is basis
        assert np.allclose(basis[:] - 1, Vr)

        transformer = DummyTransformer()
        basis = self.LinearBasis(transformer)
        assert basis.transformer is transformer

    def test_str(self):
        """Test LinearBasis.__str__() and LinearBasis.__repr__()."""
        basis = self.LinearBasis()
        assert str(basis) == "Empty LinearBasis"
        assert repr(basis).startswith("<LinearBasis object at ")

        basis.fit(np.empty((10, 4)))
        assert str(basis) == "LinearBasis" \
            "\nFull-order dimension    n = 10" \
            "\nReduced-order dimension r = 4"

        basis = self.LinearBasis(transformer=DummyTransformer())
        assert str(basis) == "Empty LinearBasis with DummyTransformer"

        basis.fit(np.empty((9, 5)))
        assert str(basis) == "LinearBasis with DummyTransformer" \
            "\nFull-order dimension    n = 9" \
            "\nReduced-order dimension r = 5"

    # Encoder / decoder -------------------------------------------------------
    def test_encode(self, n=9, r=4):
        """Test LinearBasis.encode()."""
        Vr = np.random.random((n, r))
        basis = self.LinearBasis().fit(Vr)
        q = np.random.random(n)

        # Encode without a transformer.
        q_ = Vr.T @ q
        assert np.allclose(basis.encode(q), q_)

        # Encode with a transformer.
        transformer = DummyTransformer()
        basis.transformer = transformer
        q_ = Vr.T @ (q + 1)
        assert np.allclose(basis.encode(q), q_)

    def test_decode(self, n=9, r=4):
        """Test LinearBasis.encode()."""
        Vr = np.random.random((n, r))
        basis = self.LinearBasis().fit(Vr)
        q_ = np.random.random(r)

        # Decode without a transformer.
        q = Vr @ q_
        assert np.allclose(basis.decode(q_), q)

        # Decode with a transformer.
        transformer = DummyTransformer()
        basis.transformer = transformer
        q = (Vr @ q_) - 1
        assert np.allclose(basis.decode(q_), q)

    # Persistence -------------------------------------------------------------
    def test_eq(self):
        """Test LinearBasis.__eq__()."""
        basis1 = self.LinearBasis()
        assert basis1 != 10

        basis2 = self.LinearBasis()
        basis1.fit(np.random.random((10, 4)))
        basis2.fit(np.random.random((10, 3)))
        assert basis1 != basis2

        basis1 = self.LinearBasis(transformer=DummyTransformer())
        basis1.fit(basis2.entries)
        assert basis1 != basis2

        basis1.transformer = None
        assert basis1 == basis2
        basis1.fit(basis2.entries + 1)
        assert basis1 != basis2

    def test_save(self, n=11, r=2):
        """Test LinearBasis.save()."""
        # Clean up after old tests.
        target = "_linearbasissavetest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        Vr = np.random.random((n, r))

        def _check_savefile(filename, hastransformer=False):
            with h5py.File(filename, 'r') as hf:
                assert "entries" in hf
                assert np.all(hf["entries"][:] == Vr)

                if hastransformer:
                    assert "meta" in hf
                    assert "TransformerClass" in hf["meta"].attrs
                    TClass = hf["meta"].attrs["TransformerClass"]
                    assert TClass == "DummyTransformer"
                    assert "transformer" in hf
                    assert "dummy" in hf["transformer"]
                    assert np.all(hf["transformer/dummy"][:] == (1, 1, 1))

        # Test 1: no transformer.
        basis = self.LinearBasis().fit(Vr)
        basis.save(target, save_transformer=False)
        _check_savefile(target, hastransformer=False)
        os.remove(target)
        basis.save(target, save_transformer=True)
        _check_savefile(target, hastransformer=False)
        os.remove(target)

        # Test 2: has a transformer.
        basis.transformer = DummyTransformer()
        basis.save(target, save_transformer=True)
        _check_savefile(target, hastransformer=True)
        os.remove(target)

    def test_load(self, n=10, r=5):
        """Test LinearBasis.load()."""
        # Clean up after old tests.
        target = "_linearbasisloadtest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        Vr = np.random.random((n, r))

        def _make_loadfile(loadfile, maketransformer=False, dometa=True):
            with h5py.File(loadfile, 'w') as hf:
                hf.create_dataset("entries", data=Vr)

                if maketransformer:
                    gp = hf.create_group("transformer")
                    meta = gp.create_dataset("meta", shape=(0,))
                    meta.attrs["center"] = False
                    meta.attrs["scaling"] = False
                    meta.attrs["byrow"] = False
                    meta.attrs["verbose"] = True
                    if dometa:
                        meta = hf.create_dataset("meta", shape=(0,))
                        meta.attrs["TransformerClass"] = "SnapshotTransformer"

        _make_loadfile(target, maketransformer=False)
        basis = self.LinearBasis.load(target)
        assert isinstance(basis, self.LinearBasis)
        assert np.all(basis.entries == Vr)
        assert basis.transformer is None
        os.remove(target)

        _make_loadfile(target, maketransformer=True, dometa=True)
        basis = self.LinearBasis.load(target)
        assert isinstance(basis, self.LinearBasis)
        assert np.all(basis.entries == Vr)
        assert isinstance(basis.transformer, opinf.pre.SnapshotTransformer)
        assert not basis.transformer.center
        assert not basis.transformer.scaling
        assert basis.transformer.verbose

        _make_loadfile(target, maketransformer=True, dometa=False)
        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.LinearBasis.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Check that save() and load() are inverses for an empty basis.
        basis1 = self.LinearBasis()
        basis1.save(target, overwrite=True)
        basis = self.LinearBasis.load(target)
        assert basis == basis1


class TestLinearBasisMulti:
    """Test pre.basis._linear._LinearBasis."""
    LinearBasisMulti = opinf.pre.basis.LinearBasisMulti

    def test_init(self):
        """Test LinearBasisMulti.__init__()."""
        basis = self.LinearBasisMulti(4, None, list("abcd"))
        for attr in ["num_variables", "variable_names", "transformer",
                     "r", "rs", "n", "ni", "entries", "bases"]:
            assert hasattr(basis, attr)

        assert basis.num_variables == 4
        assert basis.variable_names == list("abcd")
        assert basis.transformer is None
        assert basis.r is None
        assert basis.rs is None
        assert basis.n is None
        assert basis.ni is None
        assert basis.entries is None

        # Test individual bases.
        assert len(basis.bases) == 4
        for subbasis in basis.bases:
            assert isinstance(subbasis, self.LinearBasisMulti._BasisClass)
            assert subbasis.r is None
            assert subbasis.entries is None
            assert subbasis.transformer is None

    def test_properties(self, nvars=4, ni=10, k=27):
        """Test LinearBasisMulti properties r, rs, and entries."""
        basis = self.LinearBasisMulti(nvars)

        with pytest.raises(AttributeError) as ex:
            basis.r = 3
        assert ex.value.args[0] == "can't set attribute"

        # To test r, rs, and entries, we need to set the basis entries.
        rs = [i + 2 for i in range(nvars)]
        Vs = [la.qr(np.random.standard_normal((ni, r)), mode="economic")[0]
              for r in rs]
        for Vi, subbasis in zip(Vs, basis.bases):
            subbasis.fit(Vi)

        n = nvars * ni
        assert basis.rs == rs
        assert basis.r == sum(rs)
        for subbasis, r in zip(basis.bases, rs):
            assert subbasis.r == r
            assert subbasis.entries.shape == (ni, r)
        assert basis.entries is None
        basis._set_entries()
        assert basis.entries is not None
        assert basis.entries.shape == (n, basis.r)

    def test_eq(self, nvars=3):
        """Test LinearBasisMulti.__eq__()."""
        basis1 = self.LinearBasisMulti(nvars)
        assert basis1 != 100
        basis2 = self.LinearBasisMulti(nvars + 1)
        assert basis1 != basis2
        basis2 = self.LinearBasisMulti(nvars)
        assert basis1 == basis2
        basis2.bases[0] = 10
        assert basis1 != basis2

    def test_str(self):
        """Test LinearBasisMulti.__str__()."""
        names = list("ABCD")
        num = len(names)

        # Empty basis.
        basis = self.LinearBasisMulti(len(names), variable_names=names)
        bstr = str(basis)
        assert bstr.startswith(f"Empty {num:d}-variable LinearBasis")
        for name in "ABCD":
            assert f"\n* {name} : Empty LinearBasis" in bstr

        # Empty basis with transformer.
        basis.transformer = DummyTransformer()
        bstr = str(basis)
        assert bstr.startswith(
            f"Empty {num:d}-variable LinearBasis with DummyTransformer")
        for name in "ABCD":
            assert f"\n* {name} : Empty LinearBasis" in bstr
        assert bstr.count("DummyTransformer") == 1
        assert len(bstr.split('\n')) == 5

        # Non-empty basis.
        rs = np.random.randint(2, 10, num)
        Vs = [np.random.standard_normal((20, r)) for r in rs]
        basis.fit(Vs)
        bstr = str(basis)
        assert bstr.startswith(
            f"{num:d}-variable LinearBasis with DummyTransformer")
        for i, (name, r) in enumerate(zip("ABCD", rs)):
            assert f"\n* {name} : LinearBasis" in bstr
            assert f"Full-order dimension    n{i+1} = 20" in bstr
            assert f"Reduced-order dimension r{i+1} = {r:d}" in bstr
        n = sum(V.shape[0] for V in Vs)
        r = rs.sum()
        assert f"\nTotal full-order dimension    n = {n:d}" in bstr
        assert f"\nTotal reduced-order dimension r = {r:d}" in bstr

    # Main routines -----------------------------------------------------------
    def test_fit(self, nvars=4, ni=12, k=15):
        """Test LinearBasisMulti.fit() and _set_entries."""
        n = nvars * ni
        rs = [i + 2 for i in range(nvars)]
        rtotal = sum(rs)
        Vs = [la.qr(np.random.standard_normal((ni, r)), mode="economic")[0]
              for r in rs]
        basis = self.LinearBasisMulti(nvars).fit(Vs)
        assert isinstance(basis.entries, sparse.csc_matrix)
        assert basis.shape == (n, sum(rs))
        assert basis.n is not None
        assert basis.n == n
        assert basis.ni is not None
        assert basis.ni == ni
        assert basis.r is not None
        assert basis.r == rtotal
        for i, subbasis in enumerate(basis.bases):
            r = rs[i]
            assert subbasis.r == r
            assert isinstance(subbasis.entries, np.ndarray)
            assert subbasis.shape == (ni, r)
            assert np.allclose(subbasis.entries, Vs[i])
        topleft = basis.entries.toarray()[:ni, :rs[0]]
        assert np.all(topleft == basis.bases[0].entries)
        assert np.all(basis.entries.toarray()[ni:, :rs[0]] == 0)
        basis_prod = basis.entries.T @ basis.entries
        Id = np.eye(rtotal)
        assert basis_prod.shape == Id.shape
        assert np.allclose(basis_prod.toarray(), Id)

        # Try to use bases with different ni dimension.
        Vs = [np.random.standard_normal((ni + r, r)) for r in rs]
        basis = self.LinearBasisMulti(nvars)
        with pytest.raises(ValueError) as ex:
            basis.fit(Vs)
        assert ex.value.args[0] == "all bases must have the same row dimension"

    # Persistence -------------------------------------------------------------
    def test_save(self, nvars=4, ni=12, k=15):
        """Lightly test LinearBasisMulti.save()."""
        # Clean up after old tests.
        target = "_linearbasismultisavetest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Save an empty basis.
        basis = self.LinearBasisMulti(nvars)
        basis.save(target, overwrite=False)
        assert os.path.isfile(target)

        # Save a basis with a transformer.
        basis = self.LinearBasisMulti(nvars, transformer=DummyTransformer())
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)

        # Save a nonempty basis.
        Vs = [la.qr(np.random.standard_normal((ni, i+2)), mode="economic")[0]
              for i in range(nvars)]
        basis.fit(Vs)
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)

        # Clean up.
        os.remove(target)

    def test_load(self, nvars=4, ni=12, k=15):
        """Test LinearBasisMulti.load()."""
        # Clean up after old tests.
        target = "_linearbasismultiloadtest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Try to load a bad file.
        with h5py.File(target, "w"):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.LinearBasisMulti.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Test that save() and load() are inverses for an empty basis.
        basis1 = self.LinearBasisMulti(nvars)
        basis1.save(target, overwrite=True)
        basis2 = self.LinearBasisMulti.load(target)
        assert basis2.n is None
        assert basis2.ni is None
        assert basis2.r is None
        assert basis2.rs is None
        assert basis2.entries is None
        assert basis2.transformer is None
        assert basis1 == basis2

        # Save a basis to a temporary file (don't interrogate the file).
        rs = np.random.randint(2, 10, nvars)
        Vs = [np.random.standard_normal((ni, r)) for r in rs]
        basis1.fit(Vs)
        basis1.save(target, overwrite=True)

        # Test that save() and load() are inverses for a nonempty basis.
        basis2 = self.LinearBasisMulti.load(target)
        assert basis1.n == basis2.n
        assert basis1.ni == basis2.ni
        assert basis1.r == basis2.r
        assert basis1.entries.shape == basis2.entries.shape
        assert np.allclose(basis1.entries.toarray(), basis2.entries.toarray())
        for i, subbasis in enumerate(basis2.bases):
            assert subbasis.r == rs[i]
            assert subbasis.shape == Vs[i].shape
            assert np.all(subbasis.entries == Vs[i])
        assert basis1 == basis2

        # Repeat with a transformer.
        basis1.transformer = opinf.pre.SnapshotTransformer()
        basis1.save(target, overwrite=True)
        basis2 = self.LinearBasisMulti.load(target)
        assert basis2.transformer is not None
        assert basis2.transformer == basis1.transformer
        assert basis2 == basis1

        with h5py.File(target, 'a') as hf:
            hf["meta"].attrs["num_variables"] = nvars + 2
        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.LinearBasisMulti.load(target)
        assert ex.value.args[0] == \
            f"invalid save format (variable{nvars+1:d}/ not found)"

        # Clean up.
        os.remove(target)
