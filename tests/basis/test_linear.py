# basis/test_linear.py
"""Tests for basis._linear."""

import os
import h5py
import pytest
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import opinf


class TestLinearBasis:
    """Test basis._linear.LinearBasis."""

    Basis = opinf.basis.LinearBasis

    def _orth(self, n, r):
        return la.svd(np.random.random((n, r)), full_matrices=False)[0]

    def test_init(self, n=10, r=3):
        """Test __init__(), fit(), and entries."""
        Vr = np.ones((n, r))
        with pytest.warns(opinf.errors.UsageWarning) as wn:
            self.Basis(Vr)
        assert wn[0].message.args[0] == "basis is not orthogonal"

        Vr = self._orth(n, r)
        basis = self.Basis(Vr)
        assert basis.entries is Vr
        assert np.all(basis[:] == Vr)
        assert basis.shape == (n, r)
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r

        assert basis.fit() is basis

    def test_str(self):
        """Test __str__() and __repr__()."""
        basis = self.Basis(self._orth(10, 4))
        assert str(basis) == (
            "LinearBasis"
            "\nFull state dimension    n = 10"
            "\nReduced state dimension r = 4"
        )

        basis = self.Basis(self._orth(9, 5))
        assert str(basis) == (
            "LinearBasis"
            "\nFull state dimension    n = 9"
            "\nReduced state dimension r = 5"
        )

    # Dimension reduction  ----------------------------------------------------
    def test_compress(self, n=9, r=4):
        """Test compress()."""
        Vr = self._orth(n, r)
        basis = self.Basis(Vr)
        q = np.random.random(n)
        q_ = Vr.T @ q
        assert np.allclose(basis.compress(q), q_)

    def test_decompress(self, n=9, r=4):
        """Test decompress()."""
        Vr = self._orth(n, r)
        basis = self.Basis(Vr)
        q_ = np.random.random(r)
        q = Vr @ q_
        assert np.allclose(basis.decompress(q_), q)

        # Get only a few coordinates.
        locs = np.array([0, 2], dtype=int)
        assert np.allclose(basis.decompress(q_, locs=locs), q[locs])

    # Visualization -----------------------------------------------------------
    def test_plot1D(self, n=20, r=4):
        """Lightly test plot1D()."""
        basis = self.Basis(self._orth(n, r))

        # Turn interactive mode on.
        _pltio = plt.isinteractive()
        plt.ion()

        # Call the plotting routine.
        x = np.linspace(0, 1, n)
        ax = basis.plot1D(x)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        # Restore interactive mode setting.
        plt.interactive(_pltio)

    # Persistence -------------------------------------------------------------
    def test_eq(self):
        """Test __eq__()."""
        basis1 = self.Basis(self._orth(10, 4))
        assert basis1 != 10
        basis2 = self.Basis(self._orth(10, 3))
        assert basis1 != basis2

        basis1 = self.Basis(basis2.entries)
        assert basis1 == basis2
        basis1 = self.Basis(self._orth(10, 10) @ basis2.entries)
        assert basis1 != basis2

    def test_save(self, n=11, r=2):
        """Test LinearBasis.save()."""
        # Clean up after old tests.
        target = "_linearbasissavetest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = self._orth(n, r)

        def _check_savefile(filename):
            with h5py.File(filename, "r") as hf:
                assert "entries" in hf
                assert np.all(hf["entries"][:] == Vr)

        basis = self.Basis(Vr)
        basis.save(target)
        _check_savefile(target)
        os.remove(target)
        basis.save(target)
        _check_savefile(target)
        os.remove(target)

    def test_load(self, n=10, r=5):
        """Test LinearBasis.load()."""
        # Clean up after old tests.
        target = "_linearbasisloadtest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = self._orth(n, r)
        basis1 = self.Basis(Vr)
        basis1.save(target)
        basis2 = self.Basis.load(target)
        assert basis2 == basis1

        # Clean up.
        os.remove(target)
