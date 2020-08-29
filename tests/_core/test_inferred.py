# _core/test_inferred.py
"""Tests for rom_operator_inference._core._inferred.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi

from . import MODEL_KEYS, MODEL_FORMS, LSTSQ_REPORTS, _get_data


# Mixins (private) ============================================================
class TestInferredMixin:
    """Test _core._inferred._InferredMixin."""

    class Dummy(roi._core._base._BaseROM, roi._core._inferred._InferredMixin):
        def __init__(self, modelform):
            self.modelform = modelform

    def test_check_training_data_shapes(self):
        """Test _core._inferred._InferredMixin._check_training_data_shapes().
        """
        # Get test data.
        n, k, m, r = 60, 50, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        model = roi._core._inferred._InferredMixin()
        model.n = n
        model.m = m
        model.r = r
        labels = ["X", "Xdot", "U"]

        # Try to fit the model with a single snapshot.
        with pytest.raises(ValueError) as ex:
            model._check_training_data_shapes([X[:,0], Xdot[:,0]], labels[:2])
        assert ex.value.args[0] == "X must be two-dimensional"

        # Try to fit the model with misaligned X and Xdot.
        with pytest.raises(ValueError) as ex:
            model._check_training_data_shapes([X, Xdot[:,1:-1]], labels[:2])
        assert ex.value.args[0] == \
            "training data not aligned (Xdot.shape[1] != X.shape[1])"

        # Try to fit the model with misaligned X and U.
        with pytest.raises(ValueError) as ex:
            model._check_training_data_shapes([X, Xdot, U[:,:-1]], labels)
        assert ex.value.args[0] == \
            "training data not aligned (U.shape[1] != X.shape[1])"

        # Try with misaligned inputs (bad number of rows).
        with pytest.raises(ValueError) as ex:
            model._check_training_data_shapes([X[:-1,:], Xdot, U], labels)
        assert ex.value.args[0] == \
            f"invalid training set (X.shape[0] != n={n} or r={r})"

        with pytest.raises(ValueError) as ex:
            model._check_training_data_shapes([X, Xdot, U[:-1,:]], labels)
        assert ex.value.args[0] == \
            f"invalid training input (U.shape[0] != m={m})"

        # Correct usage.
        model._check_training_data_shapes([X, Xdot], ["X", "Xdot"])
        model._check_training_data_shapes([X, Xdot, U], ["X", "Xdot", "U"])

    def test_process_fit_arguments(self):
        """Test _core._inferred._InferredMixin._process_fit_arguments()."""
        # Get test data.
        n, k, m, r = 60, 500, 20, 10
        X, rhs, U = _get_data(n, k, m)
        U1d = U[0,:]
        Vr = la.svd(X)[0][:,:r]

        # With basis and input.
        model = self.Dummy("AB")
        X_, rhs_, U_ = model._process_fit_arguments(Vr, X, rhs, U)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr
        assert model.m == m
        assert np.allclose(X_, Vr.T @ X)
        assert np.allclose(rhs_, Vr.T @ rhs)
        assert U_ is U

        # Without basis and with a one-dimensional input.
        model.modelform = "cHB"
        X_, rhs_, U_ = model._process_fit_arguments(None, X, rhs, U1d)
        assert model.n is None
        assert model.r == n
        assert model.Vr is None
        assert model.m == 1
        assert X_ is X
        assert rhs_ is rhs
        assert U_.shape == (1,k)
        assert np.allclose(U_.reshape(-1), U)

        # With basis and no input.
        model.modelform = "cA"
        X_, rhs_, U_ = model._process_fit_arguments(Vr, X, rhs, None)
        assert model.n == n
        assert model.r == r
        assert model.Vr is Vr
        assert model.m is None
        assert np.allclose(X_, Vr.T @ X)
        assert np.allclose(rhs_, Vr.T @ rhs)
        assert U_ is None

    def test_construct_data_matrix(self):
        """Test _core._inferred._InferredMixin._construct_data_matrix()."""
        # Get test data.
        k, m, r = 500, 20, 10
        X_, _, U = _get_data(r, k, m)

        model = self.Dummy("c")
        for form in MODEL_FORMS:
            model.modelform = form
            D = model._construct_data_matrix(X_, U)
            d = roi.utils.get_least_squares_size(form, r,
                                                 m if 'B' in form else 0)
            assert D.shape == (k,d)

            # Spot check.
            if form == "c":
                assert np.allclose(D, np.ones((k,1)))
            elif form == "H":
                assert np.allclose(D, roi.utils.kron2c(X_).T)
            elif form == "G":
                assert np.allclose(D, roi.utils.kron3c(X_).T)
            elif form == "AB":
                assert np.allclose(D[:,:r], X_.T)
                assert np.allclose(D[:,r:], U.T)

    def test_solve_opinf_lstsq(self):
        """Test _core._inferred._InferredMixin._solve_opinf_lstsq()."""
        k, r = 500, 10
        d = 1 + r + r * (r+1) // 2
        D = np.random.random((k,d))
        R = np.random.random((r,k))

        model = self.Dummy("c")  # Model form doesn't matter here

        def _check_model(mdl, OO):
            for attr in LSTSQ_REPORTS:
                assert hasattr(mdl, attr)
                assert getattr(mdl, attr) >= 0
            assert OO.shape == (r,d)

        # Without regularization.
        O = model._solve_opinf_lstsq(D, R, 0)
        _check_model(model, O)
        assert np.allclose(model.datacond_, model.dataregcond_)

        # With regularization.
        O = model._solve_opinf_lstsq(D, R, 1)
        _check_model(model, O)
        assert model.datacond_ >= model.dataregcond_

    def test_extract_operators(self):
        """Test _core._inferred._InferredMixin._extract_operators()."""
        r, m = 10, 2
        shapes = {
                    "c_":  (r,),
                    "A_":  (r,r),
                    "Hc_": (r,r*(r+1)//2),
                    "Gc_": (r,r*(r+1)*(r+2)//6),
                    "B_":  (r,m),
                 }

        model = self.Dummy("c")
        model.r = r
        for form in MODEL_FORMS:
            model.modelform = form
            model.m = m if 'B' in form else 0
            d = roi.utils.get_least_squares_size(form, r, model.m)
            O = np.random.random((r,d))
            model._extract_operators(O)
            for prefix in MODEL_KEYS:
                attr = prefix+'c_' if prefix in "HG" else prefix+'_'
                assert hasattr(model, attr)
                value = getattr(model, attr)
                if prefix in form:
                    assert isinstance(value, np.ndarray)
                    assert value.shape == shapes[attr]
                else:
                    assert value is None

    def _test_fit(self, ModelClass):
        """Test _core._inferred._InferredMixin.fit(), the parent method for
        _core._inferred.InferredDiscreteROM.fit() and
        _core._inferred.InferredContinuousROM.fit().
        """
        model = ModelClass("cAH")

        # Get test data.
        n, k, m, r = 60, 500, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        U1d = U[0,:]
        Vr = la.svd(X)[0][:,:r]
        args_n = [X]
        args_r = [Vr.T @ X]
        if issubclass(ModelClass, roi._core._inferred._ContinuousROM):
            args_n.append(Xdot)
            args_r.append(Vr.T @ Xdot)

        # Fit the model with each modelform.
        for form in MODEL_FORMS:
            model.modelform = form
            if "B" in form:
                # Two-dimensional inputs.
                model.fit(Vr, *args_n, U)           # With basis.
                model.fit(None, *args_r, U)         # Without basis.
                # One-dimensional inputs.
                model.fit(Vr, *args_n, U1d)         # With basis.
                model.fit(None, *args_r, U1d)       # Without basis.
            else:
                # No inputs.
                model.fit(Vr, *args_n)              # With basis.
                model.fit(None, *args_r)            # Without basis.


# Useable classes (public) ====================================================
class TestInferredDiscreteROM:
    """Test _core._inferred.InferredDiscreteROM."""
    def test_fit(self):
        """Test _core._inferred.InferredDiscreteROM.fit()."""
        TestInferredMixin()._test_fit(roi.InferredDiscreteROM)


class TestInferredContinuousROM:
    """Test _core._inferred.InferredContinuousROM."""
    def test_fit(self):
        """Test _core._inferred.InferredContinuousROM.fit()."""
        TestInferredMixin()._test_fit(roi.InferredContinuousROM)
