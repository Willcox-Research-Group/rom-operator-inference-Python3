# core/nonparametric/test_public.py
"""Tests for core.nonparametric._public."""

import pytest
import numpy as np
import scipy.linalg as la

import opinf

from .. import (MODEL_FORMS,
                _get_data, _get_operators, _trainedmodel, _isoperator)


class TestSteadyOpInfROM:
    """Test core.nonparametric._public.SteadyOpInfROM."""
    ModelClass = opinf.core.nonparametric._public.SteadyOpInfROM

    def test_evaluate(self, r=10):
        """Test core.nonparametric._public.SteadyOpInfROM.evaluate()."""
        c_, A_, H_, G_, _ = _get_operators(r, 2)

        rom = self.ModelClass("cA")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        q_ = np.random.random(r)
        y_ = c_ + A_ @ q_
        assert np.allclose(rom.evaluate(q_), y_)

        kron2c, kron3c = opinf.utils.kron2c, opinf.utils.kron3c
        rom = self.ModelClass("HG")
        rom.r = r
        rom.H_, rom.G_ = H_, G_
        q_ = np.random.random(r)
        y_ = H_ @ kron2c(q_) + G_ @ kron3c(q_)
        assert np.allclose(rom.evaluate(q_), y_)

    def test_fit(self, n=50, k=400, r=10):
        """Test core.nonparametric._public.SteadyOpInfROM.fit()."""
        Q, F, _ = _get_data(n, k, 2)
        Vr = la.svd(Q)[0][:, :r]
        args_n = [Q, F]
        args_r = [Vr.T @ Q, Vr.T @ F]

        # Fit the rom with each modelform.
        rom = self.ModelClass("c")
        for form in MODEL_FORMS:
            if "B" in form:
                continue
            rom.modelform = form
            rom.fit(Vr, *args_n)
            rom.fit(None, *args_r)

        # Special case: fully intrusive.
        rom.modelform = "cA"
        c, A, _, _, _ = _get_operators(n, 2)
        rom.fit(Vr, None, None, known_operators={"c": c, "A": A})
        assert rom.solver_ is None
        assert _isoperator(rom.c_)
        assert _isoperator(rom.A_)
        assert np.allclose(rom.c_.entries, Vr.T @ c)
        assert np.allclose(rom.A_.entries, Vr.T @ A @ Vr)

    # def test_predict(self):
    #     """Test core.nonparametric._public.SteadyOpInfROM.predict()."""
    #     raise NotImplementedError


class TestDiscreteOpInfROM:
    """Test core.nonparametric._public.DiscreteOpInfROM."""
    ModelClass = opinf.core.nonparametric._public.DiscreteOpInfROM

    def test_evaluate(self, r=6, m=3):
        """Test core.nonparametric._public.DiscreteOpInfROM.evaluate()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        rom = self.ModelClass("cA")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        q_ = np.random.random(r)
        y_ = c_ + A_ @ q_
        assert np.allclose(rom.evaluate(q_), y_)
        assert np.allclose(rom.evaluate(q_, -1), y_)

        kron2c, kron3c = opinf.utils.kron2c, opinf.utils.kron3c
        rom = self.ModelClass("HGB")
        rom.r, rom.m = r, m
        rom.H_, rom.G_, rom.B_ = H_, G_, B_
        u = np.random.random(m)
        q_ = np.random.random(r)
        y_ = H_ @ kron2c(q_) + G_ @ kron3c(q_) + B_ @ u
        assert np.allclose(rom.evaluate(q_, u), y_)

        rom = self.ModelClass("AB")
        rom.r, rom.m = r, 1
        B1d_ = B_[:, 0]
        rom.A_, rom.B_ = A_, B1d_
        u = np.random.random()
        q_ = np.random.random(r)
        y_ = A_ @ q_ + (B1d_ * u)
        assert np.allclose(rom.evaluate(q_, u), y_)

    def test_fit(self, n=20, k=500, r=5, m=3):
        """Test core.nonparametric._public.DiscreteOpInfROM.fit()."""
        Q, Qnext, U = _get_data(n, k, m)
        U1d = U[0, :]
        Vr = la.svd(Q)[0][:, :r]
        Q_ = Vr.T @ Q

        # Fit the rom with each modelform.
        rom = self.ModelClass("c")
        for form in MODEL_FORMS:
            rom.modelform = form
            if "B" in form:
                # Two-dimensional inputs.
                rom.fit(Vr, Q, inputs=U)                # With basis.
                rom.fit(Vr, Q, Qnext, U)
                rom.fit(None, Q_, inputs=U)             # Without basis.
                # One-dimensional inputs.
                rom.fit(Vr, Q, inputs=U1d)              # With basis.
                rom.fit(Vr, Q, Qnext, U1d)
                rom.fit(None, Q_, inputs=U1d)           # Without basis.
            else:
                # No inputs.
                rom.fit(Vr, Q, inputs=None)             # With basis.
                rom.fit(Vr, Q, Qnext, None)       # With basis.
                rom.fit(None, Q_, inputs=None)          # Without basis.

        # Special case: fully intrusive.
        rom.modelform = "BA"
        _, A, _, _, B = _get_operators(n, m)
        rom.fit(Vr, None, None, known_operators={"A": A, "B": B})
        assert rom.solver_ is None
        assert _isoperator(rom.A_)
        assert _isoperator(rom.B_)
        assert np.allclose(rom.A_.entries, Vr.T @ A @ Vr)
        assert np.allclose(rom.B_.entries, Vr.T @ B)

    def test_predict(self, n=60, k=40, m=6, r=4):
        """Test core.nonparametric._public.DiscreteOpInfROM.predict()."""
        # Get test data.
        Q = _get_data(n, k, m)[0]
        Vr = la.svd(Q)[0][:, :r]
        niters = 5
        q0 = Q[:, 0]
        U = np.ones((m, niters-1))

        # Try to predict with invalid initial condition.
        q0_ = Vr.T @ q0
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0_[:-1], niters, U)
        assert ex.value.args[0] == "state0 not aligned with basis"

        # Try to predict with bad niters argument.
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, -18, U)
        assert ex.value.args[0] == \
            "argument 'niters' must be a positive integer"

        # Try to predict with badly-shaped discrete inputs.
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, niters, np.random.random((m-1, niters-1)))
        assert ex.value.args[0] == \
            f"inputs.shape = ({(m-1, niters-1)} != {(m, niters-1)}"

        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, niters, np.random.random((2, niters-1)))
        assert ex.value.args[0] == \
            f"inputs.shape = ({(2, niters-1)} != {(1, niters-1)}"

        # Try to predict with continuous inputs.
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(TypeError) as ex:
            rom.predict(q0, niters, lambda t: np.ones(m-1))
        assert ex.value.args[0] == "inputs must be NumPy array, not callable"

        for form in MODEL_FORMS:
            if "B" not in form:             # No control inputs.
                rom = _trainedmodel(self.ModelClass, form, Vr, None)
                out = rom.predict(q0, niters, decode=True)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, niters)
                out2 = rom.predict(q0, niters, decode=False)
                assert isinstance(out2, np.ndarray)
                assert out2.shape == (r, niters)
                assert np.allclose(Vr @ out2, out)

            else:                           # Has Control inputs.
                # Predict with 2D inputs.
                rom = _trainedmodel(self.ModelClass, form, Vr, m)
                out = rom.predict(q0, niters, U, decode=True)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, niters)

                # Predict with 1D inputs.
                rom = _trainedmodel(self.ModelClass, form, Vr, 1)
                out = rom.predict(q0, niters, np.ones(niters),
                                  decode=False)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, niters)

        # Predict with no basis gives result in low-dimensional space.
        rom = _trainedmodel(self.ModelClass, "cA", Vr, None)
        rom.basis = None
        out = rom.predict(Vr.T @ q0, niters, decode=True)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)


class TestContinuousOpInfROM:
    """Test core.nonparametric._public.ContinuousOpInfROM."""
    ModelClass = opinf.core.nonparametric._public.ContinuousOpInfROM

    def test_evaluate(self, r=5, m=2):
        """Test core.nonparametric._public.ContinuousOpInfROM.evaluate()."""
        c_, A_, H_, G_, B_ = _get_operators(r, m)

        rom = self.ModelClass("cA")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        q_ = np.random.random(r)
        y_ = c_ + A_ @ q_
        assert np.allclose(rom.evaluate(0, q_), y_)
        assert np.allclose(rom.evaluate(1, q_), y_)
        assert np.allclose(rom.evaluate(1, q_, -1), y_)

        kron2c, kron3c = opinf.utils.kron2c, opinf.utils.kron3c
        rom = self.ModelClass("HGB")
        rom.r, rom.m = r, m
        rom.H_, rom.G_, rom.B_ = H_, G_, B_
        uu = np.random.random(m)

        def input_func(t):
            return uu + t

        q_ = np.random.random(r)
        y_ = H_ @ kron2c(q_) + G_ @ kron3c(q_) + B_ @ uu
        assert np.allclose(rom.evaluate(0, q_, input_func), y_)
        y_ = H_ @ kron2c(q_) + G_ @ kron3c(q_) + B_ @ input_func(10)
        assert np.allclose(rom.evaluate(10, q_, input_func), y_)

        rom = self.ModelClass("AB")
        rom.r, rom.m = r, 1
        B1d_ = B_[:, 0]
        rom.A_, rom.B_ = A_, B1d_
        input_func = np.sin
        q_ = np.random.random(r)
        y_ = A_ @ q_ + (B1d_ * input_func(5))
        assert np.allclose(rom.evaluate(5, q_, input_func), y_)

        with pytest.raises(TypeError) as ex:
            rom.evaluate(5, q_, 10)
        assert "object is not callable" in ex.value.args[0]

    def test_fit(self, n=200, k=500, m=3, r=4):
        """Test core.nonparametric._public.ContinuousOpInfROM.fit()."""
        Q, Qdot, U = _get_data(n, k, m)
        U1d = U[0, :]
        Vr = la.svd(Q)[0][:, :r]
        args_n = [Q, Qdot]
        args_r = [Vr.T @ Q, Vr.T @ Qdot]

        # Fit the rom with each modelform.
        rom = self.ModelClass("c")
        for form in MODEL_FORMS:
            rom.modelform = form
            if "B" in form:
                # Two-dimensional inputs.
                rom.fit(Vr, *args_n, inputs=U)          # With basis.
                rom.fit(None, *args_r, inputs=U)        # Without basis.
                # One-dimensional inputs.
                rom.fit(Vr, *args_n, inputs=U1d)        # With basis.
                rom.fit(None, *args_r, inputs=U1d)      # Without basis.
            else:
                # No inputs.
                rom.fit(Vr, *args_n, inputs=None)       # With basis.
                rom.fit(None, *args_r, inputs=None)     # Without basis.

        # Special case: fully intrusive.
        rom.modelform = "BA"
        _, A, _, _, B = _get_operators(n, m)
        rom.fit(Vr, None, None, known_operators={"A": A, "B": B})
        assert rom.solver_ is None
        assert _isoperator(rom.A_)
        assert _isoperator(rom.B_)
        assert np.allclose(rom.A_.entries, Vr.T @ A @ Vr)
        assert np.allclose(rom.B_.entries, Vr.T @ B)

    def test_predict(self, n=60, k=50, m=10, r=6):
        """Test core.nonparametric._public.ContinuousOpInfROM.predict()."""
        # Get test data.
        Q = _get_data(n, k, m)[0]
        Vr = la.svd(Q)[0][:, :r]
        nt = 5
        q0 = Q[:, 0]
        t = np.linspace(0, .01*nt, nt)

        def input_func(tt):
            return tt*np.ones(m)

        Upred = np.column_stack([input_func(tt) for tt in t])

        # Try to predict with invalid initial condition.
        q0_ = Vr.T @ q0
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0_[1:], t, input_func)
        assert ex.value.args[0] == "state0 not aligned with basis"

        # Try to predict with bad time array.
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, np.vstack((t, t)), input_func)
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Predict without inputs.
        for form in MODEL_FORMS:
            if "B" not in form:
                rom = _trainedmodel(self.ModelClass, form, Vr, None)
                out = rom.predict(q0, t, decode=True)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, t.size)
                out = rom.predict(q0, t, decode=False)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, t.size)

        # Predict with no basis gives result in low-dimensional space.
        rom = _trainedmodel(self.ModelClass, "cA", Vr, None)
        rom.basis = None
        out = rom.predict(Vr.T @ q0, t, decode=True)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, t.size)

        # Try to predict with badly-shaped discrete inputs.
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, np.random.random((m-1, nt)))
        assert ex.value.args[0] == \
            f"input_func.shape = {(m-1, nt)} != {(m, nt)}"

        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, np.random.random((2, nt)))
        assert ex.value.args[0] == \
            f"input_func.shape = {(2, nt)} != {(1, nt)}"

        # Try to predict with badly-shaped continuous inputs.
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, lambda t: np.ones(m-1))
        assert ex.value.args[0] == \
            f"input_func() must return ndarray of shape (m,) = {(m,)}"
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, lambda t: 1)
        assert ex.value.args[0] == \
            f"input_func() must return ndarray of shape (m,) = {(m,)}"

        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, input_func)
        assert ex.value.args[0] == \
            "input_func() must return ndarray of shape (m,) = (1,) or scalar"

        # Try to predict with continuous inputs with bad return type
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, lambda t: set([5]))
        assert ex.value.args[0] == \
            f"input_func() must return ndarray of shape (m,) = {(m,)}"

        for form in MODEL_FORMS:
            if "B" in form:
                # Predict with 2D inputs.
                rom = _trainedmodel(self.ModelClass, form, Vr, m)
                # continuous input.
                out = rom.predict(q0, t, input_func, decode=True)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, nt)
                # discrete input.
                out = rom.predict(q0, t, Upred, decode=False)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)

                # Predict with 1D inputs.
                rom = _trainedmodel(self.ModelClass, form, Vr, 1)
                # continuous input.
                out = rom.predict(q0, t, lambda t: 1, decode=True)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, nt)
                out = rom.predict(q0, t, lambda t: np.array([1]),
                                  decode=False)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                # discrete input.
                out = rom.predict(q0, t, np.ones_like(t))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, nt)
                assert hasattr(rom, "predict_result_")
