# roms/test_bayes.py
"""Tests for roms._bayes.py."""

import os
import pytest
import numpy as np
import scipy.stats
import scipy.linalg as la

import opinf

try:
    from .test_nonparametric import TestROM
except ImportError:
    from test_nonparametric import TestROM


_module = opinf.roms


class TestOperatorPosterior:
    """Test roms._bayes.OperatorPosterior."""

    Posterior = opinf.roms.OperatorPosterior

    def test_init(self, r=4, d=6):
        """Test __init__()."""
        Id = np.eye(d)

        with pytest.raises(ValueError) as ex:
            self.Posterior([1, 2, 3], [1, 2])
        assert ex.value.args[0] == "len(means) = 3 != 2 = len(precisions)"

        with pytest.raises(ValueError) as ex:
            self.Posterior([np.ones(d), "a" * d], [Id] * 2)
        assert ex.value.args[0] == "means[1] should be a 1D ndarray"

        with pytest.raises(ValueError) as ex:
            self.Posterior(np.ones((r, d)), np.ones(r))
        assert ex.value.args[0] == "precisions[0] should be a 2D ndarray"

        with pytest.raises(ValueError) as ex:
            self.Posterior(np.ones((r, d)), np.ones((r, d, d + 1)))
        assert ex.value.args[0] == "means[0] and precisions[0] not aligned"

        # Homogenous dimensions.
        P = la.inv(scipy.stats.wishart.rvs(d, Id))
        post = self.Posterior(np.ones((r, d)), [P] * r)
        assert post.nrows == r
        assert np.allclose(post.means, 1)
        for X in post.covs:
            assert X.shape == (d, d)
            assert np.allclose(la.inv(X), P)
        assert len(post.randomvariables) == r

        post2 = self.Posterior(post.means, post.covs, alreadyinverted=True)
        for X2, X1 in zip(post2.covs, post.covs):
            assert np.allclose(X1, X2)

        # Different dimensions for different rows.
        post = self.Posterior(
            [np.ones(3), np.ones(2), np.ones(4)],
            [np.eye(3) * 10, np.eye(2) * 5, np.eye(4) * 100],
        )
        assert post.nrows == 3

    def test_eq(self, r=6, d=3):
        """Test __eq__()."""
        P = scipy.stats.wishart.rvs(d, np.eye(d))
        Ps = [P] * r
        post1 = self.Posterior(np.ones((r, d)), Ps)

        post2 = self.Posterior(np.ones((r - 1, d)), Ps[:-1])
        assert post1 != post2

        post2 = self.Posterior(2 * np.ones((r, d)), Ps)
        assert post1 != post2

        post2 = self.Posterior(np.ones((r, d)), [2 * P] * r)
        assert post1 != post2

        post2 = self.Posterior(np.ones((r, d)), Ps)
        assert post1 == post2

    def test_rvs(self, r=5, d=6):
        """Test rvs()."""
        # Homogeneous dimensions.
        Id = np.eye(d)
        P = la.inv(scipy.stats.wishart.rvs(d, Id))
        post = self.Posterior(np.ones((r, d)), [P] * r)
        sample = post.rvs()
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (r, d)

        # Different dimensions for different rows.
        post = self.Posterior(
            [np.ones(3), np.ones(2), np.ones(4)],
            [np.eye(3) * 10, np.eye(2) * 5, np.eye(4) * 100],
        )
        sample = post.rvs()
        assert isinstance(sample, list)
        assert len(sample) == 3
        for row in sample:
            assert isinstance(row, np.ndarray)
            assert row.ndim == 1
        assert sample[0].shape == (3,)
        assert sample[1].shape == (2,)
        assert sample[2].shape == (4,)

    def test_save_and_load(self, target="_operatorposterior_saveloadtest.h5"):
        """Test save() and load()."""
        if os.path.isfile(target):
            os.remove(target)

        post = self.Posterior(
            [
                np.random.random(3),
                np.random.random(2),
                np.random.random(4),
            ],
            [
                np.eye(3) * np.random.random(),
                np.eye(2) * np.random.random(),
                np.eye(4) * np.random.random(),
            ],
        )

        post.save(target)
        assert os.path.isfile(target)
        post2 = self.Posterior.load(target)
        assert post2 == post

        os.remove(target)


class TestBayesianROM(TestROM):
    """Test roms._bayes.BayesianROM."""

    ROM = _module.BayesianROM
    check_regselect_solver = False
    kwargs = {"num_posterior_draws": 2}

    def test_init(self):
        model = opinf.models.ContinuousModel("A", solver=None)
        with pytest.raises(AttributeError) as ex:
            self.ROM(model)
        assert ex.value.args[0].startswith("'model' must have a 'solver'")

    def test_fit_regselect_continuous(self):
        for model in self._get_models():
            rom = self.ROM(model)
            if not rom._iscontinuous:
                with pytest.raises(AttributeError) as ex:
                    rom.fit_regselect_continuous(None, None, None)
                assert ex.value.args[0] == (
                    "this method is for time-continuous models only, "
                    "use fit_regselect_discrete()"
                )
        return super().test_fit_regselect_continuous()

    def test_fit_regselect_discrete(self):
        for model in self._get_models():
            rom = self.ROM(model)
            if rom._iscontinuous:
                with pytest.raises(AttributeError) as ex:
                    rom.fit_regselect_discrete(None, None)
                assert ex.value.args[0] == (
                    "this method is for fully discrete models only, "
                    "use fit_regselect_continuous()"
                )
        return super().test_fit_regselect_discrete()


if __name__ == "__main__":
    pytest.main([__file__])
