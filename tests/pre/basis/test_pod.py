# pre/basis/test_pod.py
"""Tests for rom_operator_inference.pre.basis._pod."""

import os
import h5py
import pytest
import numpy as np
import scipy.sparse as sparse
from scipy import linalg as la
from matplotlib import pyplot as plt

import rom_operator_inference as opinf


class DummyTransformer(opinf.pre.transform._base._BaseTransformer):
    """Instantiable version of _BaseTransformer."""
    def fit_transform(self, states):
        return states + 1

    def transform(self, states):
        return self.fit_transform(states)

    def inverse_transform(self, states):
        return states - 1

    def save(self, hf):
        pass


class TestPODBasis:
    """Test pre.basis._pod.PODBasis."""
    PODBasis = opinf.pre.PODBasis

    def test_init(self):
        """Test PODBasis.init()."""
        basis = self.PODBasis()
        assert basis.transformer is None
        assert basis.r is None
        assert basis.n is None
        assert basis.shape is None
        assert basis.entries is None
        assert basis.svdvals is None
        assert basis.dual is None
        assert not basis.economize
        assert basis.rmax is None

    def test_dimensions(self, n=20, r=5):
        """Test PODBasis.r, economize, and _shrink_stored_entries_to()."""
        basis = self.PODBasis(economize=False)

        # Try setting the basis dimension before setting the entries.
        with pytest.raises(AttributeError) as ex:
            basis.r = 10
        assert ex.value.args[0] == "empty basis (call fit() first)"

        # Test _store_svd() real quick.
        V, s, Wt = la.svd(np.random.standard_normal((n, n)))
        Vr, sr, Wtr = V[:, :r], s[:r], Wt[:r]
        basis._store_svd(Vr, sr, Wtr)
        assert np.all(basis.entries == Vr)
        assert np.allclose(basis.dual, Wtr.T)
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        # Try setting the dimension too high.
        with pytest.raises(ValueError) as ex:
            basis.r = r + 2
        assert ex.value.args[0] == f"only {r} basis vectors stored"

        # Shrink dimension and blow it back up (economize=False).
        basis.r = r - 1
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        basis.r = r
        assert basis.shape == (n, r)
        assert np.all(basis.entries == Vr)
        assert np.all(basis.dual == Wtr.T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r

        # Shrink the dimension (economize=True).
        basis.economize = True
        basis.r = r - 1
        assert basis.shape == (n, r - 1)
        assert np.all(basis.entries == Vr[:, :-1])
        assert np.all(basis.dual == Wtr[:-1].T)
        assert basis.svdvals.shape == sr.shape
        assert np.all(basis.svdvals == sr)
        assert basis.rmax == r - 1

        # Try to recover forgotten columns.
        with pytest.raises(ValueError) as ex:
            basis.r = r
        assert ex.value.args[0] == f"only {r-1} basis vectors stored"

        # Ensure setting economize = True shrinks the dimension.
        basis.economize = False
        basis._store_svd(Vr, sr, Wtr)
        basis.r = r
        assert basis.rmax == r
        basis.r = r - 1
        assert basis.r == r - 1
        assert basis.rmax == r
        basis.economize = True
        assert basis.r == r - 1
        assert basis.rmax == r - 1
        with pytest.raises(ValueError) as ex:
            basis.r = r
        assert ex.value.args[0] == f"only {r-1} basis vectors stored"

        # Tests what happens when the dimension is set to 1.
        basis.economize = False
        basis._store_svd(Vr, sr, Wtr)
        basis.r = 1
        assert basis.shape == (n, 1)
        assert basis.dual.shape == (n, 1)
        assert basis.svdvals.shape == (r,)
        assert basis.encode(np.random.random(n,)).shape == (1,)
        assert basis.encode(np.random.random((n, n))).shape == (1, n)

    def test_set_dimension(self, n=20):
        """Test PODBasis.set_dimension()."""
        basis = self.PODBasis(economize=False)

        # Try setting dimension without singular values.
        with pytest.raises(AttributeError) as ex:
            basis.set_dimension(r=None, cumulative_energy=.9985)
        assert ex.value.args[0] == "no singular value data (call fit() first)"

        V, _, Wt = la.svd(np.random.standard_normal((n, n)))
        svdvals = np.sqrt([.9, .09, .009, .0009, .00009, .000009, .0000009])

        # Default: use all basis vectors.
        basis._store_svd(V, svdvals, Wt)
        basis.set_dimension()
        assert basis.r == n

        # Set specified dimension.
        basis._store_svd(V, svdvals, Wt)
        basis.set_dimension(n - 1)
        assert basis.r == n - 1

        # Choose dimension based on an energy criteria.
        basis._store_svd(V, svdvals, Wt)
        basis.set_dimension(cumulative_energy=.9999)
        assert basis.r == 4
        basis.set_dimension(residual_energy=.01)
        assert basis.r == 2

    def test_str(self):
        """Lightly test PODBasis.__str__() and LinearBasis.__repr__()."""
        basis = self.PODBasis()
        assert str(basis) == "Empty PODBasis"
        assert repr(basis).startswith("<PODBasis object at ")

    def test_fit(self, n=20, k=15, r=5):
        """Test PODBasis.fit()."""
        # First test PODBasis.validate_rank().
        states = np.empty((n, n))
        with pytest.raises(ValueError) as ex:
            self.PODBasis._validate_rank(states, n + 1)
        assert ex.value.args[0] == f"invalid POD rank r = {n + 1} " \
                                   f"(need 1 ≤ r ≤ {n})"

        self.PODBasis._validate_rank(states, n // 2)

        # Now test PODBasis.fit().
        states = np.random.standard_normal((n, k))
        U, vals, Wt = la.svd(states, full_matrices=False)
        basis = self.PODBasis().fit(states, r)
        assert basis.entries.shape == (n, r)
        assert basis.svdvals.shape == (min(n, k),)
        assert basis.dual.shape == (k, r)
        VrTVr = basis.entries.T @ basis.entries
        Ir = np.eye(r)
        assert np.allclose(VrTVr, Ir)
        WrTWr = basis.dual.T @ basis.dual
        assert np.allclose(WrTWr, Ir)
        assert basis.r == r
        assert np.allclose(basis.entries, U[:, :r])
        assert np.allclose(basis.svdvals, vals)
        assert np.allclose(basis.dual, Wt[:r, :].T)

        # Test with a transformer.
        basis = self.PODBasis(transformer=DummyTransformer())
        basis.fit(states, r)
        U, vals, Wt = la.svd(states + 1, full_matrices=False)
        assert np.allclose(basis.entries, U[:, :r])
        assert np.allclose(basis.svdvals, vals)
        assert np.allclose(basis.dual, Wt[:r, :].T)

        # TODO: weighted inner product matrix.

    def test_fit_randomized(self, n=20, k=14, r=5, tol=1e-6):
        """Test PODBasis.fit_randomized()."""
        options = dict(n_oversamples=30, n_iter=10, random_state=42)
        states = np.random.standard_normal((n, k))
        U, vals, Wt = la.svd(states, full_matrices=False)
        basis = self.PODBasis().fit_randomized(states, r, **options)
        assert basis.entries.shape == (n, r)
        assert basis.svdvals.shape == (r,)
        assert basis.dual.shape == (k, r)
        VrTVr = basis.entries.T @ basis.entries
        Ir = np.eye(r)
        assert np.allclose(VrTVr, Ir)
        WrTWr = basis.dual.T @ basis.dual
        assert np.allclose(WrTWr, Ir)
        assert basis.r == r
        # Flip the signs in U and W if needed so things will match.
        for i in range(r):
            if np.sign(U[0, i]) != np.sign(basis[0, i]):
                U[:, i] *= -1
            if np.sign(Wt[i, 0]) != np.sign(basis.dual[0, i]):
                Wt[i, :] *= -1
        assert la.norm(basis.entries - U[:, :r], ord=2) < tol
        assert la.norm(basis.svdvals - vals[:r]) / la.norm(basis.svdvals) < tol
        assert la.norm(basis.dual - Wt[:r, :].T, ord=2) < tol

        # Test with a transformer.
        states = np.random.standard_normal((n, n))
        basis = self.PODBasis(transformer=DummyTransformer())
        basis.fit_randomized(states, r, **options)
        U, vals, Wt = la.svd(states + 1, full_matrices=False)
        # Flip the signs in U and W if needed so things will match.
        for i in range(r):
            if np.sign(U[0, i]) != np.sign(basis[0, i]):
                U[:, i] *= -1
            if np.sign(Wt[i, 0]) != np.sign(basis.dual[0, i]):
                Wt[i, :] *= -1
        assert la.norm(basis.entries - U[:, :r], ord=2) < tol
        assert la.norm(basis.svdvals - vals[:r]) / la.norm(basis.svdvals) < tol
        assert la.norm(basis.dual - Wt[:r, :].T, ord=2) < tol

    # Visualization -----------------------------------------------------------
    def test_plots(self, n=40, k=25, r=4):
        """Lightly test PODBasis.plot_*()."""
        basis = self.PODBasis().fit(np.random.standard_normal((n, k)))

        # Turn interactive mode on.
        _pltio = plt.isinteractive()
        plt.ion()

        # Call each plotting routine.
        ax = basis.plot_svdval_decay(threshold=1e-3, normalize=True)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = basis.plot_residual_energy(threshold=1e-3)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = basis.plot_cumulative_energy(threshold=.999)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        fig, axes = basis.plot_energy()
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        for ax in axes.flat:
            assert isinstance(ax, plt.Axes)
        plt.close(fig)

        # Restore interactive mode setting.
        plt.interactive(_pltio)

    # Persistence -------------------------------------------------------------
    def test_save(self, n=20, k=14, r=6):
        """Lightly test PODBasis.save()."""
        # Clean up after old tests.
        target = "_podbasissavetest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Just save a basis to a temporary file, don't interrogate the file.
        basis = self.PODBasis().fit(np.random.random((n, k)), r)
        basis.save(target)
        assert os.path.isfile(target)

        # Repeat with a transformer.
        basis = self.PODBasis(transformer=DummyTransformer())
        basis.fit(np.random.random((n, k)), r)
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)
        os.remove(target)

    def test_load(self, n=20, k=14, r=6):
        """Test PODBasis.load()."""
        # Clean up after old tests.
        target = "_podbasisloadtest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Try to load a bad file.
        with h5py.File(target, "w"):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.PODBasis.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Test that save() and load() are inverses for an empty basis.
        basis1 = self.PODBasis(economize=True)
        basis1.save(target, overwrite=True)
        basis2 = self.PODBasis.load(target)
        assert isinstance(basis2, self.PODBasis)
        assert basis2.n is None
        assert basis2.r is None
        assert basis2.entries is None
        assert basis2.dual is None
        assert basis2.transformer is None
        assert basis2.economize is True
        assert basis1 == basis2

        # Just save a basis to a temporary file, don't interrogate the file.
        basis1 = self.PODBasis().fit(np.random.random((n, k)), r)
        basis1.save(target, overwrite=True)

        # Test that save() and load() are inverses.
        basis2 = self.PODBasis.load(target)
        assert basis1.r == basis2.r
        assert basis1.entries.shape == basis2.entries.shape
        assert np.allclose(basis1.entries, basis2.entries)
        assert basis1.svdvals.shape == basis2.svdvals.shape
        assert np.allclose(basis1.svdvals, basis2.svdvals)
        assert basis1.dual.shape == basis2.dual.shape
        assert np.allclose(basis1.dual, basis2.dual)
        assert basis1 == basis2

        # Repeat with a transformer.
        st = opinf.pre.transform.SnapshotTransformer()
        basis1 = self.PODBasis(transformer=st).fit(np.random.random((n, k)), r)
        basis1.save(target, overwrite=True)
        basis2 = self.PODBasis.load(target)
        assert basis1 == basis2

        # Clean up.
        os.remove(target)


class TestPODBasisMulti:
    """Test opinf.pre.basis._pod.PODBasisMulti."""
    PODBasisMulti = opinf.pre.PODBasisMulti

    def test_init(self):
        """Test PODBasisMulti.__init__()."""
        basis = self.PODBasisMulti(4, None, True, list("abcd"))
        for attr in ["num_variables", "variable_names", "transformer",
                     "r", "rs", "entries", "economize", "bases"]:
            assert hasattr(basis, attr)

        assert basis.num_variables == 4
        assert basis.variable_names == list("abcd")
        assert basis.transformer is None
        assert basis.r is None
        assert basis.rs is None
        assert basis.entries is None
        assert basis.economize is True

        # Test individual bases.
        assert len(basis.bases) == 4
        for subbasis in basis.bases:
            assert isinstance(subbasis, opinf.pre.PODBasis)
            assert subbasis.economize is True
            assert subbasis.r is None
            assert subbasis.entries is None
            assert subbasis.transformer is None

    def test_properties(self, nvars=4, ni=10, k=27):
        """Test PODBasisMulti properties r, rs, economize, and entries."""
        basis = self.PODBasisMulti(nvars, None, False)

        with pytest.raises(AttributeError) as ex:
            basis.r = 3
        assert ex.value.args[0] == "can't set attribute"

        with pytest.raises(AttributeError) as ex:
            basis.rs = [3] * basis.num_variables
        assert ex.value.args[0] == "empty basis (call fit() first)"

        # Test economize.
        for subbasis in basis.bases:
            assert subbasis.economize is False
        basis.economize = True
        for subbasis in basis.bases:
            assert subbasis.economize is True

        # To test r, rs, and entries, we need to set the basis entries.
        Qs = np.random.random((nvars, ni, k))
        rs = []
        for i, (Q, subbasis) in enumerate(zip(Qs, basis.bases)):
            r = i + 2
            subbasis.fit(Q, r=r)
            rs.append(r)

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

        # Change rs.
        newrs = [r - 1 for r in rs]
        basis.rs = newrs
        assert basis.rs == newrs
        assert basis.r == sum(newrs)
        for subbasis, r in zip(basis.bases, newrs):
            assert subbasis.r == r
            assert subbasis.entries.shape == (ni, r)
        assert basis.entries is not None
        assert basis.entries.shape == (n, basis.r)
        basis._set_entries()
        assert basis.entries.shape == (n, basis.r)

        # Try to set rs with wrong number of dimensions.
        with pytest.raises(ValueError) as ex:
            basis.rs = newrs[:-1]
        assert ex.value.args[0] == f"rs must have length {nvars}"

    def test_fit(self, nvars=4, ni=12, k=15):
        """Test PODBasisMulti.fit()."""
        n = nvars * ni
        states = np.random.standard_normal((n, k))
        Us = [la.svd(s, full_matrices=False)[0]
              for s in np.split(states, nvars, axis=0)]
        maxrank = min(ni, k)

        # Test full rank basis.
        basis = self.PODBasisMulti(nvars).fit(states, None)
        assert isinstance(basis.entries, sparse.csc_matrix)
        assert basis.shape == (n, nvars * maxrank)
        for i, subbasis in enumerate(basis.bases):
            assert subbasis.r == maxrank
            assert isinstance(subbasis.entries, np.ndarray)
            assert subbasis.shape == (ni, maxrank)
            assert np.allclose(subbasis.entries, Us[i])
        topleft = basis.entries.toarray()[:ni, :maxrank]
        assert np.all(topleft == basis.bases[0].entries)
        assert np.all(basis.entries.toarray()[ni:, :maxrank] == 0)
        basis_prod = basis.entries.T @ basis.entries
        Id = np.eye(n)
        assert basis_prod.shape == Id.shape
        assert np.allclose(basis_prod.toarray(), Id)

        # Test with specified ranks.
        rs = [i + 2 for i in range(nvars)]
        basis = self.PODBasisMulti(nvars).fit(states, rs)
        assert isinstance(basis.entries, sparse.csc_matrix)
        assert basis.shape == (n, sum(rs))
        for i, subbasis in enumerate(basis.bases):
            r = rs[i]
            assert subbasis.r == r
            assert isinstance(subbasis.entries, np.ndarray)
            assert subbasis.shape == (ni, r)
            assert np.allclose(subbasis.entries, Us[i][:, :r])
        topleft = basis.entries.toarray()[:ni, :rs[0]]
        assert np.all(topleft == basis.bases[0].entries)
        assert np.all(basis.entries.toarray()[ni:, :rs[0]] == 0)
        basis_prod = basis.entries.T @ basis.entries
        Id = np.eye(sum(rs))
        assert basis_prod.shape == Id.shape
        assert np.allclose(basis_prod.toarray(), Id)

        # Test with cumulative / residual energy criteria.
        basis1 = self.PODBasisMulti(nvars).fit(states, cumulative_energy=.99)
        basis2 = self.PODBasisMulti(nvars).fit(states, residual_energy=1e-3)
        for basis in (basis1, basis2):
            assert isinstance(basis.entries, sparse.csc_matrix)
            r = sum(sb.r for sb in basis.bases)
            assert basis.shape == (n, r)
            r0 = basis.bases[0].r
            topleft = basis.entries.toarray()[:ni, :r0]
            assert np.all(topleft == basis.bases[0].entries)
            assert np.all(basis.entries.toarray()[ni:, :r0] == 0)
            basis_prod = basis.entries.T @ basis.entries
            Id = np.eye(r)
            assert basis_prod.shape == Id.shape
            assert np.allclose(basis_prod.toarray(), Id)

        # Test with a transformer.
        basis = self.PODBasisMulti(nvars, transformer=DummyTransformer())
        basis.fit(states, rs=[3]*nvars)
        Us = [la.svd(s + 1, full_matrices=False)[0]
              for s in np.split(states, nvars, axis=0)]
        assert isinstance(basis.entries, sparse.csc_matrix)
        assert basis.shape == (n, nvars * 3)
        for i, subbasis in enumerate(basis.bases):
            assert subbasis.r == 3
            assert isinstance(subbasis.entries, np.ndarray)
            assert subbasis.shape == (ni, 3)
            assert np.allclose(subbasis.entries, Us[i][:, :3])
        topleft = basis.entries.toarray()[:ni, :3]
        assert np.all(topleft == basis.bases[0].entries)
        assert np.all(basis.entries.toarray()[ni:, :3] == 0)
        basis_prod = basis.entries.T @ basis.entries
        Id = np.eye(3 * nvars)
        assert basis_prod.shape == Id.shape
        assert np.allclose(basis_prod.toarray(), Id)

    def test_fit_randomized(self, nvars=3, ni=13, k=11, tol=1e-6):
        """Test PODBasisMulti.fit_randomized()."""
        options = dict(n_oversamples=30, n_iter=10, random_state=42)
        n = nvars * ni
        states = np.random.standard_normal((n, k))
        Us = [la.svd(s, full_matrices=False)[0]
              for s in np.split(states, nvars, axis=0)]

        # Try fitting with bad dimension.
        basis = self.PODBasisMulti(nvars)
        with pytest.raises(TypeError) as ex:
            basis.fit_randomized(states, None)
        assert ex.value.args[0] == f"rs must be list of length {nvars}"

        # Test with specified ranks.
        rs = [i + 2 for i in range(nvars)]
        basis.fit_randomized(states, rs, **options)
        assert isinstance(basis.entries, sparse.csc_matrix)
        assert basis.shape == (n, sum(rs))
        for i, subbasis in enumerate(basis.bases):
            r = rs[i]
            assert subbasis.r == r
            assert isinstance(subbasis.entries, np.ndarray)
            assert subbasis.shape == (ni, r)
            for j in range(subbasis.r):
                # Flip signs for comparison as needed.
                if np.sign(Us[i][0, j]) != np.sign(subbasis.entries[0, j]):
                    Us[i][:, j] *= -1
            assert np.allclose(subbasis.entries, Us[i][:, :r], atol=tol)
        topleft = basis.entries.toarray()[:ni, :rs[0]]
        assert np.all(topleft == basis.bases[0].entries)
        assert np.all(basis.entries.toarray()[ni:, :rs[0]] == 0)
        basis_prod = basis.entries.T @ basis.entries
        Id = np.eye(sum(rs))
        assert basis_prod.shape == Id.shape
        assert np.allclose(basis_prod.toarray(), Id)

        # Test with a transformer.
        basis = self.PODBasisMulti(nvars, transformer=DummyTransformer())
        options.pop("random_state")
        basis.fit_randomized(states, rs=[3]*nvars, **options)
        Us = [la.svd(s + 1, full_matrices=False)[0]
              for s in np.split(states, nvars, axis=0)]
        assert isinstance(basis.entries, sparse.csc_matrix)
        assert basis.shape == (n, nvars * 3)
        for i, subbasis in enumerate(basis.bases):
            assert subbasis.r == 3
            assert isinstance(subbasis.entries, np.ndarray)
            assert subbasis.shape == (ni, 3)
            for j in range(subbasis.r):
                # Flip signs for comparison as needed.
                if np.sign(Us[i][0, j]) != np.sign(subbasis.entries[0, j]):
                    Us[i][:, j] *= -1
            assert np.allclose(subbasis.entries, Us[i][:, :3], atol=tol)
        topleft = basis.entries.toarray()[:ni, :3]
        assert np.all(topleft == basis.bases[0].entries)
        assert np.all(basis.entries.toarray()[ni:, :3] == 0)
        basis_prod = basis.entries.T @ basis.entries
        Id = np.eye(3 * nvars)
        assert basis_prod.shape == Id.shape
        assert np.allclose(basis_prod.toarray(), Id)

    def test_save(self, nvars=4, ni=12, k=15):
        """Lightly test PODBasisMulti.save()."""
        # Clean up after old tests.
        target = "_podbasismultisavetest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Save an empty basis.
        basis = self.PODBasisMulti(nvars)
        basis.save(target, overwrite=False)
        assert os.path.isfile(target)

        # Save a basis with a transformer.
        basis = self.PODBasisMulti(nvars, transformer=DummyTransformer())
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)

        # Save a nonempty basis.
        states = np.random.standard_normal((nvars * ni, k))
        basis.fit(states)
        basis.save(target, overwrite=True)
        assert os.path.isfile(target)

        # Clean up.
        os.remove(target)

    def test_load(self, nvars=4, ni=12, k=15):
        """Test PODBasisMulti.load()."""
        # Clean up after old tests.
        target = "_podbasismultiloadtest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Try to load a bad file.
        with h5py.File(target, "w"):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.PODBasisMulti.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Test that save() and load() are inverses for an empty basis.
        basis1 = self.PODBasisMulti(nvars, economize=True)
        basis1.save(target, overwrite=True)
        basis2 = self.PODBasisMulti.load(target)
        assert isinstance(basis2, self.PODBasisMulti)
        assert basis2.n is None
        assert basis2.ni is None
        assert basis2.r is None
        assert basis2.rs is None
        assert basis2.entries is None
        assert basis2.transformer is None
        assert basis2.economize is True
        assert basis1 == basis2

        # Save a basis to a temporary file (don't interrogate the file).
        rs = np.random.randint(2, 10, nvars)
        states = np.random.standard_normal((nvars * ni, k))
        basis1.fit(states, rs)
        basis1.save(target, overwrite=True)

        # Test that save() and load() are inverses for a nonempty basis.
        basis2 = self.PODBasisMulti.load(target)
        assert basis1.n == basis2.n
        assert basis1.ni == basis2.ni
        assert basis1.r == basis2.r
        assert basis1.entries.shape == basis2.entries.shape
        assert np.allclose(basis1.entries.toarray(), basis2.entries.toarray())
        for i, subbasis in enumerate(basis2.bases):
            assert subbasis.r == rs[i]
            assert subbasis.shape == (ni, rs[i])
        assert basis1 == basis2

        # Clean up.
        os.remove(target)


# Basis computation ===========================================================
def test_pod_basis(set_up_basis_data):
    """Test pre.basis._pod.pod_basis()."""
    Q = set_up_basis_data
    n, k = Q.shape

    # Try with an invalid rank.
    rmax = min(n, k)
    with pytest.raises(ValueError) as exc:
        opinf.pre.pod_basis(Q, rmax+1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = {rmax+1} (need 1 ≤ r ≤ {rmax})"

    with pytest.raises(ValueError) as exc:
        opinf.pre.pod_basis(Q, -1)
    assert exc.value.args[0] == \
        f"invalid POD rank r = -1 (need 1 ≤ r ≤ {rmax})"

    # Try with an invalid mode.
    with pytest.raises(NotImplementedError) as exc:
        opinf.pre.pod_basis(Q, None, mode="full")
    assert exc.value.args[0] == "invalid mode 'full'"

    U, vals, Wt = la.svd(Q, full_matrices=False)
    for r in [2, 10, rmax]:
        Ur = U[:, :r]
        vals_r = vals[:r]
        Wr = Wt[:r, :].T
        Id = np.eye(r)

        for mode in ("dense", "sparse", "randomized"):

            print(r, mode)
            basis, svdvals = opinf.pre.pod_basis(Q, r, mode=mode)
            _, _, W = opinf.pre.pod_basis(Q, r, mode=mode, return_W=True)
            assert basis.shape == (n, r)
            assert np.allclose(basis.T @ basis, Id)
            assert W.shape == (k, r)
            assert np.allclose(W.T @ W, Id)

            if mode == "dense":
                assert svdvals.shape == (rmax,)
            if mode in ("sparse", "randomized"):
                assert svdvals.shape == (r,)
                # Make sure the basis vectors have the same sign.
                for j in range(r):
                    if not np.isclose(basis[0, j], Ur[0, j]):
                        basis[:, j] *= -1
                    if not np.isclose(W[0, j], Wr[0, j]):
                        W[:, j] *= -1

            if mode != "randomized":
                # Accuracy tests (none for randomized SVD).
                assert np.allclose(basis, Ur)
                assert np.allclose(svdvals[:r], vals_r)
                assert np.allclose(W, Wr)


# Reduced dimension selection =================================================
def test_svdval_decay(set_up_basis_data):
    """Test pre.basis._pod.svdval_decay()."""
    Q = set_up_basis_data
    svdvals = la.svdvals(Q)

    # Single cutoffs.
    r = opinf.pre.svdval_decay(svdvals, 1e-14, plot=False)
    assert isinstance(r, int) and r >= 1

    # Multiple cutoffss.
    rs = opinf.pre.svdval_decay(svdvals, [1e-10, 1e-12], plot=False)
    assert isinstance(rs, list)
    for r in rs:
        assert isinstance(r, int) and r >= 1
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = opinf.pre.svdval_decay(svdvals, .0001, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = opinf.pre.svdval_decay(svdvals, [1e-4, 1e-8, 1e-12], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")

    # Specific test.
    svdvals = [.9, .09, .009, .0009, .00009, .000009, .0000009]
    rs = opinf.pre.svdval_decay(svdvals, [.8, .1, .0004],
                                normalize=False, plot=False)
    assert len(rs) == 3
    assert rs == [1, 1, 4]

    svdvals = np.array([1e1, 1e2, 1e3, 1e0, 1e-2]) - 1e-3
    rs = opinf.pre.svdval_decay(svdvals, [9e-1, 9e-2, 5e-4, 0],
                                normalize=True, plot=False)
    assert len(rs) == 4
    assert rs == [1, 2, 4, 5]


def test_cumulative_energy(set_up_basis_data):
    """Test pre.basis._pod.cumulative_energy()."""
    Q = set_up_basis_data
    svdvals = la.svdvals(Q)
    energy = np.cumsum(svdvals**2)/np.sum(svdvals**2)

    def _test(r, thresh):
        assert isinstance(r, int)
        assert r >= 1
        assert energy[r-1] >= thresh
        assert np.all(energy[:r-2] < thresh)

    # Single threshold.
    thresh = .9
    r = opinf.pre.cumulative_energy(svdvals, thresh, plot=False)
    _test(r, thresh)

    # Multiple thresholds.
    thresh = [.9, .99, .999]
    rs = opinf.pre.cumulative_energy(svdvals, thresh, plot=False)
    assert isinstance(rs, list)
    for r, t in zip(rs, thresh):
        _test(r, t)
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = opinf.pre.cumulative_energy(svdvals, .999, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = opinf.pre.cumulative_energy(svdvals, [.9, .99, .999], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")

    # Specific test.
    svdvals = np.sqrt([.9, .09, .009, .0009, .00009, .000009, .0000009])
    rs = opinf.pre.cumulative_energy(svdvals, [.9, .99, .999], plot=False)
    assert len(rs) == 3
    assert rs == [1, 2, 3]


def test_residual_energy(set_up_basis_data):
    """Test pre.basis._pod.residual_energy()."""
    Q = set_up_basis_data
    svdvals = la.svdvals(Q)
    resid = 1 - np.cumsum(svdvals**2)/np.sum(svdvals**2)

    def _test(r, tol):
        assert isinstance(r, int)
        assert r >= 1
        assert resid[r-1] <= tol
        assert np.all(resid[:r-2] > tol)

    # Single tolerance.
    tol = 1e-2
    r = opinf.pre.residual_energy(svdvals, tol, plot=False)
    _test(r, tol)

    # Multiple tolerances.
    tols = [1e-2, 1e-4, 1e-6]
    rs = opinf.pre.residual_energy(svdvals, tols, plot=False)
    assert isinstance(rs, list)
    for r, t in zip(rs, tols):
        _test(r, t)
    assert rs == sorted(rs)

    # Plotting.
    status = plt.isinteractive()
    plt.ion()
    rs = opinf.pre.residual_energy(svdvals, 1e-3, plot=True)
    assert len(plt.gcf().get_axes()) == 1
    rs = opinf.pre.cumulative_energy(svdvals, [1e-2, 1e-4, 1e-6], plot=True)
    assert len(plt.gcf().get_axes()) == 1
    plt.interactive(status)
    plt.close("all")


def test_projection_error(set_up_basis_data):
    """Test pre.basis._pod.projection_error()."""
    Q = set_up_basis_data
    Vr = la.svd(Q, full_matrices=False)[0][:, :Q.shape[1]//3]

    abserr, relerr = opinf.pre.projection_error(Q, Vr)
    assert np.isscalar(abserr)
    assert abserr >= 0
    assert np.isscalar(relerr)
    assert relerr >= 0
    assert np.isclose(abserr, relerr * la.norm(Q))
