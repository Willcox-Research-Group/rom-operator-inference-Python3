# pre/basis/test_linear.py
"""Tests for rom_operator_inference.pre.basis._linear."""

import os
import h5py
import pytest
import numpy as np

import rom_operator_inference as opinf


class TestLinearBasis:
    """Test pre.basis._linear._LinearBasis."""
    LinearBasis = opinf.pre.basis.LinearBasis

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

        transformer = self.DummyTransformer()
        basis = self.LinearBasis(transformer)
        assert basis.transformer is transformer

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
        transformer = self.DummyTransformer()
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
        transformer = self.DummyTransformer()
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

        basis1 = self.LinearBasis(transformer=self.DummyTransformer())
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
        basis.transformer = self.DummyTransformer()
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
