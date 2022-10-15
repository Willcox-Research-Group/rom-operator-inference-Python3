# core/interpolate/test_public.py
"""Tests for core.interpolate._public."""

# import pytest
import numpy as np
import scipy.interpolate
import scipy.linalg as la

import opinf

from .. import MODEL_FORMS, _get_data, _get_operators


def _get_interp_operators(s, r, m):
    """Get s 1D parameters and dummy interpolated operators."""
    params = np.sort(np.random.uniform(size=s))
    c_, A_, H_, G_, B_ = _get_operators(r, m)
    c_ = opinf.core.operators.InterpolatedConstantOperator(
        params, [c_ + p for p in params], scipy.interpolate.CubicSpline)
    A_ = opinf.core.operators.InterpolatedLinearOperator(
        params, [A_ + 2*p for p in params], scipy.interpolate.CubicSpline)
    H_ = opinf.core.operators.InterpolatedQuadraticOperator(
        params, [H_ + p**2 for p in params], scipy.interpolate.CubicSpline)
    G_ = opinf.core.operators.InterpolatedCubicOperator(
        params, [G_ + p**3 for p in params], scipy.interpolate.CubicSpline)
    B_ = opinf.core.operators.InterpolatedLinearOperator(
        params, [B_ + 2*p for p in params], scipy.interpolate.CubicSpline)
    return params, c_, A_, H_, G_, B_


# # TODO: Copy TestSteadyOpInfROM from tests/core/nonparametric/test_public.py.
# class TestInterpolatedSteadyOpInfROM:
#     """Test core.interpolate._public.InterpolatedSteadyOpInfROM."""
#     ModelClass = opinf.core.interpolate._public.InterpolatedSteadyOpInfROM


class TestInterpolatedDiscreteOpInfROM:
    """Test core.interpolate._public.InterpolatedDiscreteOpInfROM."""
    ModelClass = opinf.core.interpolate._public.InterpolatedDiscreteOpInfROM

    def test_evaluate(self, r=6, m=3, s=4):
        """Test InterpolatedDiscreteOpInfROM.evaluate()."""
        params, c_, A_, H_, G_, B_ = _get_interp_operators(s, r, m)
        B1d_ = opinf.core.operators.InterpolatedLinearOperator(
            params, [B[:, 0] for B in B_.matrices],
            scipy.interpolate.CubicSpline)

        rom = self.ModelClass("cA", "cubicspline")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        q_ = np.random.random(r)
        p = np.random.uniform()
        y_ = c_(p).evaluate() + A_(p).evaluate(q_)
        assert np.allclose(rom.evaluate(p, q_), y_)
        assert np.allclose(rom.evaluate(p, q_, -1), y_)

        rom = self.ModelClass("HGB")
        rom.r, rom.m = r, m
        rom.H_, rom.G_, rom.B_ = H_, G_, B_
        u = np.random.random(m)
        q_ = np.random.random(r)
        p = np.random.uniform()
        y_ = H_(p).evaluate(q_) + G_(p).evaluate(q_) + B_(p).evaluate(u)
        assert np.allclose(rom.evaluate(p, q_, u), y_)

        rom = self.ModelClass("AB")
        rom.r, rom.m = r, 1
        rom.A_, rom.B_ = A_, B1d_
        p = np.random.uniform()
        u = np.random.random()
        q_ = np.random.random(r)
        y_ = A_(p).evaluate(q_) + B1d_(p).evaluate(u)
        assert np.allclose(rom.evaluate(p, q_, u), y_)

    def test_fit(self, n=20, m=3, s=4, k=500, r=2):
        """Test InterpolatedDiscreteOpInfROM.fit()."""
        params = np.sort(np.random.uniform(size=s))
        Q, Qnext, U = _get_data(n, k, m)
        Qs = [Q + p for p in params]
        Qnexts = [Qnext + 2*p for p in params]
        Us = [U - p for p in params]
        U1ds = [U[0, :] for U in Us]
        Vr = la.svd(np.hstack(Qs))[0][:, :r]
        Qs_ = [Vr.T @ Q for Q in Qs]

        # Fit the rom with each modelform.
        rom = self.ModelClass("c")
        for form in MODEL_FORMS:
            rom.modelform = form
            if "B" in form:
                # Two-dimensional inputs.
                rom.fit(Vr, params, Qs, inputs=Us)          # With basis.
                rom.fit(Vr, params, Qs, Qnexts, Us)
                rom.fit(None, params, Qs_, inputs=Us)       # Without basis.
                # One-dimensional inputs.
                rom.fit(Vr, params, Qs, inputs=U1ds)        # With basis.
                rom.fit(Vr, params, Qs, Qnexts, U1ds)
                rom.fit(None, params, Qs_, inputs=U1ds)     # Without basis.
            else:
                # No inputs.
                rom.fit(Vr, params, Qs, inputs=None)        # With basis.
                rom.fit(Vr, params, Qs, Qnexts, None)       # With basis.
                rom.fit(None, params, Qs_, inputs=None)     # Without basis.

        # Special case: fully intrusive, still parametric.
        c, A, _, _, B = _get_operators(n, m)
        cs = [c + p for p in params]
        As = [A + 2*p for p in params]

        rom.modelform = "cA"
        rom.fit(Vr, params, None, known_operators=dict(c=cs, A=As))
        assert isinstance(rom.c_,
                          opinf.core.operators.InterpolatedConstantOperator)
        assert isinstance(rom.A_,
                          opinf.core.operators.InterpolatedLinearOperator)
        assert np.allclose(rom.A_.matrices, [Vr.T @ A @ Vr for A in As])
        assert np.allclose(rom.c_.matrices, [Vr.T @ c for c in cs])

        # Special case: fully intrusive, fully nonparametric.
        rom.modelform = "BA"
        rom.fit(Vr, params, None, None, known_operators={"A": A, "B": B})
        assert isinstance(rom.A_, opinf.core.operators.LinearOperator)
        assert isinstance(rom.B_, opinf.core.operators.LinearOperator)
        assert np.allclose(rom.A_.entries, Vr.T @ A @ Vr)
        assert np.allclose(rom.B_.entries, Vr.T @ B)

    def test_predict(self, r=4, s=5, niters=5, ntrials=10):
        """Test InterpolatedDiscreteOpInfROM.predict()."""
        # Construct a dummy model.
        params = np.sort(np.random.uniform(size=s))
        A_ = np.random.random((r, r))
        A_ = opinf.core.operators.InterpolatedLinearOperator(
            params, [A_/(1 + p)**2 for p in params],
            scipy.interpolate.CubicSpline)
        rom = self.ModelClass("A")
        rom._set_operators(basis=None, A_=A_)

        # Test prediction.
        for _ in range(ntrials):
            q0 = np.random.random(r)
            p = np.random.uniform()
            A = A_(p).entries
            actual = np.array([np.linalg.matrix_power(A, i) @ q0
                               for i in range(niters)]).T
            predicted = rom.predict(p, q0, niters)
            assert np.allclose(predicted, actual)


class TestInterpolatedContinuousOpInfROM:
    """Test core.interpolate._public.InterpolatedContinuousOpInfROM."""
    ModelClass = opinf.core.interpolate._public.InterpolatedContinuousOpInfROM

    def test_evaluate(self, r=6, m=3, s=4):
        """Test InterpolatedContinuousOpInfROM.evaluate()."""
        params, c_, A_, H_, G_, B_ = _get_interp_operators(s, r, m)

        rom = self.ModelClass("cA", "cubicspline")
        rom.r = r
        rom.c_, rom.A_ = c_, A_
        q_ = np.random.random(r)
        p = np.random.uniform()
        y_ = c_(p).evaluate() + A_(p).evaluate(q_)
        assert np.allclose(rom.evaluate(p, 0, q_), y_)
        assert np.allclose(rom.evaluate(p, 1, q_), y_)
        assert np.allclose(rom.evaluate(p, 1, q_, -1), y_)

        rom = self.ModelClass("HGB")
        rom.r, rom.m = r, m
        rom.H_, rom.G_, rom.B_ = H_, G_, B_
        q_ = np.random.random(r)
        p = np.random.uniform()
        uu = np.random.random(m)

        def input_func(t):
            return uu + t

        ybar = H_(p).evaluate(q_) + G_(p).evaluate(q_)
        y_ = ybar + B_(p).evaluate(uu)
        assert np.allclose(rom.evaluate(p, 0, q_, input_func), y_)
        y_ = ybar + B_(p).evaluate(input_func(10))
        assert np.allclose(rom.evaluate(p, 10, q_, input_func), y_)

    def test_fit(self, n=20, m=3, s=4, k=500, r=2):
        """Test InterpolatedContinuousOpInfROM.fit()."""
        params = np.sort(np.random.uniform(size=s))
        Q, Qdot, U = _get_data(n, k, m)
        Qs = [Q + p for p in params]
        Qdots = [Qdot + 2*p for p in params]
        Us = [U - p for p in params]
        U1ds = [U[0, :] for U in Us]
        Vr = la.svd(np.hstack(Qs))[0][:, :r]
        Qs_ = [Vr.T @ Q for Q in Qs]
        Qdots_ = [Vr.T @ Qdot for Qdot in Qdots]

        # Fit the rom with each modelform.
        rom = self.ModelClass("c")
        for form in MODEL_FORMS:
            rom.modelform = form
            if "B" in form:
                # Two-dimensional inputs.
                rom.fit(Vr, params, Qs, Qdots, Us)
                rom.fit(None, params, Qs_, Qdots_, inputs=Us)
                # One-dimensional inputs.
                rom.fit(Vr, params, Qs, Qdots, U1ds)
                rom.fit(None, params, Qs_, Qdots_, inputs=U1ds)
            else:
                # No inputs.
                rom.fit(Vr, params, Qs, Qdots, None)
                rom.fit(None, params, Qs_, Qdots_, inputs=None)

        # Special case: fully intrusive, still parametric.
        c, A, _, _, B = _get_operators(n, m)
        cs = [c + p for p in params]
        As = [A + 2*p for p in params]

        rom.modelform = "cA"
        rom.fit(Vr, params, None, None, known_operators=dict(c=cs, A=As))
        assert isinstance(rom.c_,
                          opinf.core.operators.InterpolatedConstantOperator)
        assert isinstance(rom.A_,
                          opinf.core.operators.InterpolatedLinearOperator)
        assert np.allclose(rom.A_.matrices, [Vr.T @ A @ Vr for A in As])
        assert np.allclose(rom.c_.matrices, [Vr.T @ c for c in cs])

        # Special case: fully intrusive, fully nonparametric.
        rom.modelform = "BA"
        rom.fit(Vr, params, None, None, known_operators={"A": A, "B": B})
        assert isinstance(rom.A_, opinf.core.operators.LinearOperator)
        assert isinstance(rom.B_, opinf.core.operators.LinearOperator)
        assert np.allclose(rom.A_.entries, Vr.T @ A @ Vr)
        assert np.allclose(rom.B_.entries, Vr.T @ B)

    def test_predict(self, r=4, s=5, nt=5, ntrials=10):
        """Test InterpolatedContinuousOpInfROM.predict()."""
        # Construct a dummy model.
        params = np.sort(np.random.uniform(size=s))
        A_ = np.eye(r)
        A_ = opinf.core.operators.InterpolatedLinearOperator(
            params, [A_/(1 + p)**2 for p in params],
            scipy.interpolate.CubicSpline)
        rom = self.ModelClass("A")
        rom._set_operators(basis=None, A_=A_)
        t = np.linspace(0, .01*nt, nt)

        # Test prediction.
        for _ in range(ntrials):
            q0 = np.random.random(r)
            p = np.random.uniform()
            A = A_(p).entries
            actual = np.array([la.expm(A * tt) @ q0
                               for tt in t]).T
            predicted = rom.predict(p, q0, t)
            assert np.allclose(predicted, actual)
