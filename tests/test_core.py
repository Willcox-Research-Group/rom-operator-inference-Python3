# test_core.py
"""Tests for rom_operator_inference._core.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


def _get_data(n=200, k=100, m=20):
    X = np.random.random((n,k))
    Xdot = np.zeros((n,k))
    U = np.ones((m,k))

    return X, Xdot, U


class TestReducedModel:
    """Test rom_operator_inference._core.ReducedModel."""


    @pytest.fixture
    def set_up_fresh_model(self):
        return roi.ReducedModel("LQc", has_inputs=False)


    @pytest.fixture
    def set_up_trained_model(self, r=15):
        X, Xdot, U = _get_data()
        Vr = la.svd(X)[0][:,:r]
        return roi.ReducedModel("LQc", has_inputs=True).fit(X, Xdot, Vr, U)


    def test_init(self):
        """Test ReducedModel.__init__()."""
        with pytest.raises(TypeError) as exc:
            roi.ReducedModel()
        assert exc.value.args[0] == \
            "__init__() missing 1 required positional argument: 'modelform'"

        with pytest.raises(TypeError) as exc:
            roi.ReducedModel("LQc", False, None)
        assert exc.value.args[0] == \
            "__init__() takes from 2 to 3 positional arguments but 4 were given"

        model = roi.ReducedModel("LQc")
        assert hasattr(model, "modelform")


    def test_fit(self, set_up_fresh_model):
        model = set_up_fresh_model

        # Get test data.
        n, k, m, r = 200, 100, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        Vr = la.svd(X)[0][:,:r]

        # Try to use an invalid modelform.
        model.modelform = "LLL"
        with pytest.raises(ValueError) as exc:
            model.fit(X, Xdot, Vr)
        assert exc.value.args[0] == \
            f"invalid modelform 'LLL'. Options are {model._VALID_MODEL_FORMS}."
        model.modelform = "LQc"

        model.has_inputs = True
        with pytest.raises(ValueError) as exc:
            model.fit(X, Xdot, Vr)
        assert exc.value.args[0] == \
            "argument 'U' required since has_inputs=True"

        model.has_inputs = False
        with pytest.raises(ValueError) as exc:
            model.fit(X, Xdot, Vr, U=U)
        assert exc.value.args[0] == \
            "argument 'U' invalid since has_inputs=False"

        # Try to fit the model with misaligned X and Xdot.
        with pytest.raises(ValueError) as exc:
            model.fit(X, Xdot[:,1:-1], Vr)
        assert exc.value.args[0] == \
            f"X and Xdot different shapes ({(n,k)} != {(n,k-2)})"

        # Try to fit the model with misaligned X and Vr.
        with pytest.raises(ValueError) as exc:
            model.fit(X, Xdot, Vr[1:-1,:])
        assert exc.value.args[0] == \
            f"X and Vr not aligned, first dimension {n} != {n-2}"

        # Fit the model with each possible modelform.
        model.has_inputs = False
        for form in model._VALID_MODEL_FORMS:
            model.modelform = form
            model.fit(X, Xdot, Vr)

        # Test fit output sizes.
        model.modelform = "LQc"
        model.has_inputs = True
        model.fit(X, Xdot, Vr, U=U)
        assert model.n == n
        assert model.r == r
        assert model.m == m
        assert model.A_.shape == (r,r)
        assert model.F_.shape == (r,r*(r+1)//2)
        assert model.H_.shape == (r,r**2)
        assert model.c_.shape == (r,)
        assert model.B_.shape == (r,m)
        assert hasattr(model, "residual_")

        # Try again with one-dimensional inputs.
        m = 1
        U = np.ones(k)
        model.fit(X, Xdot, Vr, U=U)
        n, r, m = model.n, model.r, model.m
        assert model.n == n
        assert model.r == r
        assert model.m == 1
        assert model.A_.shape == (r,r)
        assert model.F_.shape == (r,r*(r+1)//2)
        assert model.H_.shape == (r,r**2)
        assert model.c_.shape == (r,)
        assert model.B_.shape == (r,1)
        assert hasattr(model, "residual_")


    def test_predict(self, set_up_fresh_model):
        model = set_up_fresh_model

        # Get test data.
        n, k, m, r = 200, 100, 20, 10
        X, Xdot, U = _get_data(n, k, m)
        Vr = la.svd(X)[0][:,:r]
        U1d = np.ones(k)

        nt = 30
        x0 = X[:,0]
        t = np.linspace(0, 1, nt)
        u = lambda t: np.ones(m)
        Upred = np.ones((m, nt))

        # Try to predict before fitting.
        model = roi.ReducedModel("LQc", has_inputs=True)
        with pytest.raises(AttributeError) as exc:
            model.predict(x0, t, u)
        assert exc.value.args[0] == "model not trained (call fit() first)"
        model.fit(X, Xdot, Vr, U)

        # Try to predict with invalid initial condition.
        x0_ = Vr.T @ x0
        with pytest.raises(ValueError) as exc:
            model.predict(x0_, t, u)
        assert exc.value.args[0] == f"invalid initial state size ({r} != {n})"

        # Try to predict with weird time array.
        with pytest.raises(ValueError) as exc:
            model.predict(x0, np.vstack((t,t)), u)
        assert exc.value.args[0] == "time 't' must be one-dimensional"

        # Try to predict without inputs when required and vice versa
        model.has_inputs = True
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t)
        assert exc.value.args[0] == \
            "argument 'u' required since has_inputs=True"

        model.has_inputs = False
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t, u)
        assert exc.value.args[0] == \
            "argument 'u' invalid since has_inputs=False"

        # Change has_inputs between fit() and predict().
        model.has_inputs = True
        model.fit(X, Xdot, Vr, U)
        model.has_inputs = False
        with pytest.raises(AttributeError) as exc:
            model.predict(x0, t)
        assert exc.value.args[0] == \
            "`has_inputs` attribute altered between fit()" \
            " and predict(); call fit() again to retrain"

        model.has_inputs = False
        model.fit(X, Xdot, Vr)
        model.has_inputs = True
        with pytest.raises(AttributeError) as exc:
            model.predict(x0, t, u)
        assert exc.value.args[0] == \
            "`has_inputs` attribute altered between fit()" \
            " and predict(); call fit() again to retrain"

        # Predict without inputs.
        model.has_inputs = False
        model.fit(X, Xdot, Vr)
        model.predict(x0, t)

        # Try to predict with badly-shaped discrete inputs.
        model.has_inputs = True
        model.fit(X, Xdot, Vr, U)
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t, np.random.random((m-1, nt)))
        assert exc.value.args[0] == \
            f"invalid input shape ({(m-1,nt)} != {(m,nt)}"

        model.fit(X, Xdot, Vr, U1d)
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t, np.random.random((2, nt)))
        assert exc.value.args[0] == \
            f"invalid input shape ({(2,nt)} != {(1,nt)}"

        # Try to predict with badly-shaped continuous inputs.
        model.has_inputs = True
        model.fit(X, Xdot, Vr, U)                       # 2D case
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t, lambda t: np.ones(m-1))
        assert exc.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t, lambda t: 1)
        assert exc.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        model.fit(X, Xdot, Vr, U1d)                     # 1D case
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t, u)
        assert exc.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(1,)}" \
            " or scalar"

        # Try to predict with continuous inputs with bad return type
        model.has_inputs = True
        model.fit(X, Xdot, Vr, U)                       # 2D case
        with pytest.raises(ValueError) as exc:
            model.predict(x0, t, lambda t: set([5]))
        assert exc.value.args[0] == \
            f"input function u() must return ndarray of shape (m,)={(m,)}"

        # Successes
        model.has_inputs = True
        for form in model._VALID_MODEL_FORMS:
            model.modelform = form

            # Predict with 2D inputs.
            model.fit(X, Xdot, Vr, U)
            out = model.predict(x0, t, u)                 # continuous case
            assert isinstance(out, np.ndarray)
            assert out.shape == (n,nt)
            out = model.predict(x0, t, Upred)             # discrete case
            assert isinstance(out, np.ndarray)
            assert out.shape == (n,nt)

            # Predict with 1D inputs.
            model.fit(X, Xdot, Vr, U1d)
            out = model.predict(x0, t, lambda t: 1)       # continuous cases
            assert isinstance(out, np.ndarray)
            assert out.shape == (n,nt)
            out = model.predict(x0, t, lambda t: np.array([1]))
            assert isinstance(out, np.ndarray)
            assert out.shape == (n,nt)
            out = model.predict(x0, t, np.ones_like(t))   # discrete case
            assert isinstance(out, np.ndarray)
            assert out.shape == (n,nt)


    def test_getitem(self, set_up_trained_model):
        """Test ReducedModel.__getitem__() (indexing, e.g., model["A_"])."""
        model = set_up_trained_model

        assert model.A_ is model["A_"]
        assert model.F_ is model["F_"]
        assert model.c_ is model["c_"]
        assert model.B_ is model["B_"]
        assert np.allclose(model.H_, model["H_"])

        with pytest.raises(KeyError) as exc:
            model["G_"]
        assert exc.value.args[0] == f"valid keys are {model._VALID_KEYS}"


    def test_str(self, set_up_fresh_model):
        """Test ReducedModel.__str__() (string representation)."""
        model = set_up_fresh_model

        model.modelform = "L"
        assert str(model) == "Reduced-order model structure: " \
                             "dx / dt = Ax(t)"
        model.modelform = "Lc"
        assert str(model) == "Reduced-order model structure: " \
                             "dx / dt = Ax(t) + c"
        model.modelform = "Q"
        assert str(model) == "Reduced-order model structure: " \
                             "dx / dt = H(x ⊗ x)(t)"
        model.modelform = "Qc"
        assert str(model) == "Reduced-order model structure: " \
                             "dx / dt = H(x ⊗ x)(t) + c"
        model.modelform = "LQ"
        assert str(model) == "Reduced-order model structure: " \
                             "dx / dt = Ax(t) + H(x ⊗ x)(t)"
        model.modelform = "LQc"
        assert str(model) == "Reduced-order model structure: " \
                             "dx / dt = Ax(t) + H(x ⊗ x)(t) + c"

        model.modelform = "bad_form"
        with pytest.raises(ValueError) as exc:
            str(model)
        assert exc.value.args[0] == \
            "invalid modelform 'bad_form'; " \
            f"valid options: {model._VALID_MODEL_FORMS}"
