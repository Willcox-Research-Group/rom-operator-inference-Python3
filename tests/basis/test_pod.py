# basis/test_pod.py
"""Tests for basis._pod."""

import os
import pytest
import numpy as np

# from scipy import linalg as la
from matplotlib import pyplot as plt

import opinf


class TestPODBasis:
    """Test basis._pod.PODBasis."""

    Basis = opinf.basis.PODBasis

    # Constructors ------------------------------------------------------------
    def test_init(self):
        """Test __init__() andd properties."""
        # Valid instantiation.
        basis = self.Basis(num_vectors=10, name="testbasis")
        for attr in (
            "reduced_state_dimension",
            "full_state_dimension",
            "entries",
            "weights",
            "leftvecs",
            "svdvals",
            "rightvecs",
            "residual_energy",
            "cumulative_energy",
        ):
            assert getattr(basis, attr) is None
        assert basis.name == "testbasis"
        assert isinstance(basis.solver_options, dict)
        assert len(basis.solver_options) == 0

        # Setter for mode.
        with pytest.raises(AttributeError) as ex:
            basis.mode = "smartly"
        assert ex.value.args[0].startswith("invalid mode 'smartly', options: ")
        basis.mode = "randomized"

        # Setter for solver_options.
        with pytest.raises(TypeError) as ex:
            basis.solver_options = 10
        assert ex.value.args[0] == "solver_options must be a dictionary"
        basis.solver_options["full_matrices"] = False

    def test_from_svd(self):
        """Test from_svd() pseudoconstructor."""
        raise NotImplementedError

    # Dimension management ----------------------------------------------------
    def test_set_dimension(self):
        basis = self.Basis(num_vectors=10)

        # Dimension selection criteria
        with pytest.raises(opinf.errors.UsageWarning) as wn:
            basis._set_dimension_from_criterion(
                num_vectors=20,
                cumulative_energy=0.999,
                residual_energy=0.01,
            )
        assert wn[0].message.args[0] == (
            "received multiple dimension selection criteria, "
            "using num_vectors=20"
        )

        with pytest.raises(ValueError) as ex:
            basis._set_dimension_selection_criterion()
        assert ex.value.args[0] == (
            "exactly one dimension selection criterion must be provided"
        )

        basis._set_dimension_selection_criterion(residual_energy=1e-6)
        criterion = basis._PODBasis__criterion
        assert criterion[0] == "residual_energy"
        assert criterion[1] == 1e-6

        # Dimension setting (empty basis).
        basis.reduced_state_dimension = 5
        criterion = basis._PODBasis__criterion
        assert criterion[0] == "num_vectors"
        assert criterion[1] == 5
        assert basis.reduced_state_dimension == 5

        # Dimension setting (existing basis).
        basis = self.Basis.from_svd(None, None)  # TODO

    # Visualization -----------------------------------------------------------
    def test_plots(self, n=40, k=25, r=4):
        """Lightly test plot_*()."""
        basis = self.Basis(num_vectors=r).fit(
            np.random.standard_normal((n, k))
        )

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

        ax = basis.plot_cumulative_energy(threshold=0.999)
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
    def test_save(self, n=20, k=14, r=6, target="_podbasissavetest.h5"):
        """Lightly test save()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Just save a basis to a temporary file, don't interrogate the file.
        basis = self.Basis(num_vectors=r).fit(np.random.random((n, k)))
        basis.save(target)
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self, n=20, k=14, r=6):
        """Test load()."""
        # Clean up after old tests.
        target = "_podbasisloadtest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Test that save() and load() are inverses for an empty basis.
        basis1 = self.Basis(num_vectors=r)
        basis1.save(target, overwrite=True)
        basis2 = self.Basis.load(target)
        assert basis1 == basis2

        # Test that save() and load() are inverses for a nonempty basis.
        basis1.fit(np.random.random((n, k)))
        basis1.save(target, overwrite=True)
        basis2 = self.Basis.load(target)
        assert basis1 == basis2

        # Test max_vectors gives a smaller basis.
        rnew = basis1.reduced_state_dimension - 2
        basis2 = self.Basis.load(target, max_vectors=rnew)
        assert basis2.reduced_state_dimension == rnew
        assert basis2.entries.shape == (basis1.entries.shape[0], rnew)
        assert basis2.max_vectors == rnew
        assert basis2.svdvals.shape == (k,)
        assert np.allclose(basis2.svdvals, basis1.svdvals)
        assert np.allclose(basis1.entries[:, :-2], basis2.entries)

        # Clean up.
        os.remove(target)
