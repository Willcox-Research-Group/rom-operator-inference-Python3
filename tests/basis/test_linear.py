# basis/test_linear.py
"""Tests for basis._linear."""

import os
import pytest
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import opinf


def test_Wmult(n=10):
    """Test basis._linear._Wmult()."""
    func = opinf.basis._linear._Wmult

    w = np.random.random(n) + 0.1
    W = np.diag(w)

    # One-dimensional operand
    q = np.random.random(n)
    Wq = W @ q
    for ww in w, W:
        out = func(ww, q)
        assert out.shape == (n,)
        assert np.allclose(out, Wq)

    # Two-dimensional operand
    Q = np.random.random((n, n))
    WQ = W @ Q
    for ww in w, W:
        out = func(ww, Q)
        assert out.shape == (n, n)
        assert np.allclose(out, WQ)

    with pytest.raises(ValueError) as ex:
        func(w, np.random.random((2, 2, 2)))
    assert ex.value.args[0] == "expected one- or two-dimensional array"


def test_weighted_svd(n=100, k=10):
    """Test basis._linear.weighted_svd()."""
    func = opinf.basis._linear.weighted_svd

    with pytest.raises(ValueError) as ex:
        func(None, np.random.random((1, 1, 1)))
    assert ex.value.args[0] == (
        "expected one- or two-dimensional spatial weights"
    )

    Q = np.random.random((n, k))
    w = np.random.random(n) + 0.1
    W = np.diag(w)
    Id = np.eye(k)

    # Diagonal weights.
    for ww in w, W:
        Qw = func(Q, ww)
        assert Qw.shape == (n, k)
        assert np.allclose(Qw.T @ W @ Qw, Id)

    # Non-diagonal weights.
    W = np.random.random((n, n))
    u, s, _ = la.svd(W)
    W = u @ np.diag(s + 1) @ u.T
    QW = func(Q, W)
    assert QW.shape == (n, k)
    assert np.allclose(QW.T @ W @ QW, Id)


class TestLinearBasis:
    """Test basis._linear.LinearBasis."""

    Basis = opinf.basis.LinearBasis

    def _orth(self, n, r):
        return la.svd(np.random.random((n, r)), full_matrices=False)[0]

    def test_init(self, n=10, r=3):
        """Test __init__(), fit(), and entries."""
        # Empty basis.
        basis = self.Basis(None, name="something")
        assert basis.entries is None
        assert basis.weights is None
        assert basis.full_state_dimension is None
        assert basis.reduced_state_dimension is None
        assert basis.name == "something"

        # Check that compress() and decompress() are disabled.
        for method in basis.compress, basis.decompress:
            with pytest.raises(AttributeError) as ex:
                method(0)
            assert ex.value.args[0] == "basis entries not initialized"

        # Ensure the full_state_dimension cannot be set.
        with pytest.raises(AttributeError) as ex:
            basis.full_state_dimension = 10
        assert basis.full_state_dimension is None

        # Orthogonal basis.
        Vr = self._orth(n, r)
        basis = self.Basis(Vr)
        assert basis.entries is Vr
        assert basis.shape == (n, r)
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r
        assert np.all(basis[:] == Vr)
        assert basis.weights is None
        assert basis.name is None

        # Non-orthogonal basis.
        Vr = np.ones((n, r))
        with pytest.warns(opinf.errors.UsageWarning) as wn:
            basis = self.Basis(Vr)
        assert wn[0].message.args[0] == "basis not orthogonal"
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r
        assert np.all(basis.entries == 1)
        assert basis.weights is None
        assert basis.fit() is basis

        # Orthogonalize the basis.
        basis = self.Basis(Vr, orthogonalize=True)
        assert np.any(basis.entries != Vr)
        assert np.allclose(basis.entries.T @ basis.entries, np.eye(r))

        # Weighted basis without orthogonalizing.
        w = 1 + np.random.random(n)
        Vr = opinf.basis._linear.weighted_svd(np.random.random((n, r)), w)
        basis1 = self.Basis(
            entries=Vr,
            weights=w,
            orthogonalize=False,
            check_orthogonality=True,
        )
        basis2 = self.Basis(
            entries=Vr,
            weights=np.diag(w),
            orthogonalize=False,
            check_orthogonality=True,
        )
        assert np.all(basis1.entries == basis2.entries)
        assert np.all(basis2.entries == Vr)

        # Weighted basis with orthogonalizing.
        Vr = np.random.random((n, r))
        basis = self.Basis(Vr, w, orthogonalize=True)

    def test_str(self):
        """Test __str__() and __repr__()."""
        basis = self.Basis(self._orth(10, 4))
        assert str(basis) == (
            "LinearBasis"
            "\nFull state dimension    n = 10"
            "\nReduced state dimension r = 4"
        )

        basis = self.Basis(self._orth(9, 5), name="varname")
        assert str(basis) == (
            "LinearBasis for variable 'varname'"
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

        # TODO: test weighted compression

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

        basis2 = self.Basis(
            basis2.entries,
            weights=np.random.random(basis2.full_state_dimension) + 0.5,
            orthogonalize=True,
        )
        assert basis1 != basis2
        assert basis2 != basis1
        basis1 = self.Basis(basis2.entries, weights=basis2.weights)
        assert basis1 == basis2

    def test_save(self, n=11, r=2, target="_linearbasissavetest.h5"):
        """Lightly test LinearBasis.save()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = self._orth(n, r)
        w = np.random.random(n)

        basis = self.Basis(Vr)
        basis.save(target)
        assert os.path.isfile(target)
        os.remove(target)

        basis = self.Basis(Vr, w, orthogonalize=True)
        basis.save(target)
        assert os.path.isfile(target)
        os.remove(target)

    def test_load(self, n=10, r=5, target="_linearbasisloadtest.h5"):
        """Test LinearBasis.load()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = self._orth(n, r)
        basis1 = self.Basis(Vr)
        basis1.save(target)
        basis2 = self.Basis.load(target)
        assert basis2 == basis1

        w = np.random.random(n) + 0.5
        basis1 = self.Basis(Vr, w, orthogonalize=True)
        basis1.save(target, overwrite=True)
        basis2 = self.Basis.load(target)
        assert basis2 == basis1

        # Clean up.
        os.remove(target)
