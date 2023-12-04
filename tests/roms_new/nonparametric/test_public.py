# roms/nonparametric/test_public.py
"""Tests for roms.nonparametric._public."""

import pytest
import numpy as np
import scipy.linalg as la

import opinf

from .. import MODEL_FORMS, _get_data, _get_operators, _trainedmodel


class TestSteadyROM:
    """Test roms.nonparametric._public.SteadyROM."""

    ModelClass = opinf.roms_new.nonparametric._public.SteadyROM

    def test_evaluate(self, r=10):
        """Lightly test SteadyROM.evaluate().
        Stronger tests in test_base.TestNonparametricROM.test_evaluate().
        """
        rom = _trainedmodel(self.ModelClass, "cAH", np.empty((r, r)), 0)
        rom.evaluate(np.random.random(r))

    def test_jacobian(self, r=6, m=3):
        """Lightly test DiscreteROM.jacobian().
        Stronger tests in test_base.TestNonparametricROM.test_jacobian().
        """
        rom = _trainedmodel(self.ModelClass, "A", np.empty((r, r)), 0)
        rom.jacobian(np.random.random(r))

    def test_fit(self, n=50, k=400, r=10):
        """Lightly test SteadyROM.fit().
        Stronger tests in test_base.TestNonparametricROM.test_fit().
        """
        Q, F, _ = _get_data(r, k, 2)
        rom = self.ModelClass("A")
        rom.fit(Q, F)

    # def test_predict(self):
    #     """Test SteadyROM.predict()."""
    #     raise NotImplementedError


class TestDiscreteROM:
    """Test roms.nonparametric._public.DiscreteROM."""

    ModelClass = opinf.roms_new.nonparametric._public.DiscreteROM

    def test_stack_trajectories(self, n=10, k=20, m=5, num_trajectories=4):
        """Test DiscreteROM.stack_trajectories()."""
        statelist, inputlist = [], []
        for _ in range(num_trajectories):
            Q, _, U = _get_data(n, k, m)
            statelist.append(Q)
            inputlist.append(U)

        Qs, Qnexts = self.ModelClass.stack_trajectories(statelist)
        assert Qs.shape == (n, (k - 1) * num_trajectories)
        assert Qnexts.shape == Qs.shape
        Qs_split = np.split(Qs, num_trajectories, axis=1)
        Qnexts_split = np.split(Qnexts, num_trajectories, axis=1)
        for i in range(num_trajectories):
            assert np.all(Qs_split[i][:, 1:] == Qnexts_split[i][:, :-1])
            assert np.all(Qs_split[i] == statelist[i][:, :-1])
            assert np.all(Qnexts_split[i] == statelist[i][:, 1:])

        Qs2, Qnexts2, Us = self.ModelClass.stack_trajectories(
            statelist,
            inputlist,
        )
        assert Qs2.shape == Qs.shape
        assert np.all(Qs2 == Qs)
        assert Qnexts2.shape == Qnexts.shape
        assert np.all(Qnexts2 == Qnexts)
        assert Us.shape == (m, (k - 1) * num_trajectories)
        for i, Usplit in enumerate(np.split(Us, num_trajectories, axis=1)):
            assert np.all(Usplit == inputlist[i][:, : (k - 1)])

        # 1D inputs
        inputlist_1d = [np.random.random(k) for _ in range(num_trajectories)]
        Qs3, Qnexts3, Us_1d = self.ModelClass.stack_trajectories(
            statelist,
            inputlist_1d,
        )
        assert Qs3.shape == Qs.shape
        assert np.all(Qs3 == Qs)
        assert Qnexts3.shape == Qnexts.shape
        assert np.all(Qnexts3 == Qnexts)
        assert Us_1d.shape == ((k - 1) * num_trajectories,)
        for i, Usplit in enumerate(np.split(Us_1d, num_trajectories, axis=0)):
            assert np.all(Usplit == inputlist_1d[i][: (k - 1)])

    def test_evaluate(self, r=6, m=3):
        """Lightly test DiscreteROM.evaluate().
        Stronger tests in test_base.TestNonparametricROM.test_evaluate().
        """
        rom = _trainedmodel(self.ModelClass, "cG", np.empty((r, r)), 0)
        rom.evaluate(np.random.random(r))

    def test_jacobian(self, r=6, m=3):
        """Lightly test DiscreteROM.jacobian().
        Stronger tests in test_base.TestNonparametricROM.test_jacobian().
        """
        rom = _trainedmodel(self.ModelClass, "cG", np.empty((r, r)), 0)
        rom.jacobian(np.random.random(r))

    def test_fit(self, n=20, k=500, r=5, m=3):
        """Test DiscreteROM.fit()."""
        Q, _, U = _get_data(n, k, m)
        Qnext = Q[:, 1:]
        rom = self.ModelClass("A").fit(Q)
        rom2 = self.ModelClass("A").fit(Q[:, :-1], Qnext)
        assert rom.A_ == rom2.A_

        rom = self.ModelClass("AB").fit(Q, inputs=U)
        rom2 = self.ModelClass("AB").fit(Q[:, :-1], Qnext, inputs=U)
        assert rom.A_ == rom2.A_
        assert rom.B_ == rom2.B_

    def test_predict(self, n=10, k=20, m=6, r=4):
        """Test DiscreteROM.predict()."""
        # Get test data.
        Q = _get_data(r, k, m)[0]
        niters = 5
        q0 = Q[:, 0]
        U = np.ones((m, niters - 1))
        Vr = la.svd(np.random.random((n, k)))[0][:, :r]

        ops = A_, B_ = _get_operators("AB", r, m)
        B1d_ = _get_operators("B", r, m=1)[0]
        rom = self.ModelClass(ops)

        # Try to predict with bad niters argument.
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, -18, U)
        assert (
            ex.value.args[0] == "argument 'niters' must be a positive integer"
        )

        # Try to predict with badly-shaped discrete inputs.
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, niters, np.random.random((m - 1, niters - 1)))
        assert (
            ex.value.args[0] == f"inputs.shape = ({(m-1, niters-1)} "
            f"!= {(m, niters-1)} = (m, niters-1)"
        )

        rom_m1 = self.ModelClass([A_, B1d_])
        with pytest.raises(ValueError) as ex:
            rom_m1.predict(q0, niters, np.random.random((2, niters - 1)))
        assert (
            ex.value.args[0] == f"inputs.shape = ({(2, niters-1)} "
            f"!= {(1, niters-1)} = (m, niters-1)"
        )

        # Try to predict with continuous inputs.
        with pytest.raises(TypeError) as ex:
            rom.predict(q0, niters, lambda t: np.ones(m - 1))
        assert ex.value.args[0] == "inputs must be NumPy array, not callable"

        # No control inputs.
        rom = self.ModelClass([A_], basis=Vr)
        out = rom.predict(q0, niters, decompress=True)
        assert isinstance(out, np.ndarray)
        assert out.shape == (n, niters)
        out2 = rom.predict(q0, niters, decompress=False)
        assert isinstance(out2, np.ndarray)
        assert out2.shape == (r, niters)
        assert np.allclose(Vr @ out2, out)

        # With 2D inputs.
        rom = self.ModelClass([A_, B_])
        out = rom.predict(q0, niters, U)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)

        # With 1D inputs.
        rom = self.ModelClass([A_, B1d_])
        out = rom.predict(q0, niters, np.ones(niters))
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)


class TestContinuousROM:
    """Test roms.nonparametric._public.ContinuousROM."""

    ModelClass = opinf.roms_new.nonparametric._public.ContinuousROM

    def test_evaluate(self, r=5, m=2):
        """Test ContinuousROM.evaluate()."""
        A_, B_ = _get_operators("AB", r, m)

        rom = self.ModelClass([A_])
        q_ = np.random.random(r)
        rom.evaluate(10, q_)

        rom = self.ModelClass([A_, B_])
        with pytest.raises(TypeError) as ex:
            rom.evaluate(5, q_, 10)
        assert "object is not callable" in ex.value.args[0]

        def input_func(t):
            return np.random.random(m)

        rom.evaluate(np.pi, q_, input_func)

    def test_jacobian(self, r=6, m=3, ntrials=10):
        """Test ContinuousROM.jacobian()."""
        A_, B_ = _get_operators("AB", r, m)

        rom = self.ModelClass([A_])
        q_ = np.random.random(r)
        rom.jacobian(8, q_)

        rom = self.ModelClass([A_, B_])
        with pytest.raises(TypeError) as ex:
            rom.jacobian(5, q_, 10)
        assert "object is not callable" in ex.value.args[0]

        def input_func(t):
            return np.random.random(m)

        rom.jacobian(2, q_, input_func)

    def test_fit(self, k=20, m=3, r=4):
        """Lightly test ContinuousROM.fit().
        Stronger tests in test_base.TestNonparametricROM.test_fit().
        """
        Q, Qdot, U = _get_data(r, k, m)
        self.ModelClass("AB").fit(Q, Qdot, U)

    def test_predict(self, n=60, k=50, m=10, r=6):
        """Test ContinuousROM.predict()."""
        # Get test data.
        Q = _get_data(n, k, m)[0]
        Vr = la.svd(Q)[0][:, :r]
        nt = 5
        q0 = Q[:, 0]
        t = np.linspace(0, 0.01 * nt, nt)

        def input_func(tt):
            return tt * np.ones(m)

        Upred = np.column_stack([input_func(tt) for tt in t])

        # Try to predict with invalid initial condition.
        q0_ = Vr.T @ q0
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            rom.predict(q0_[1:], t, input_func)
        assert ex.value.args[0] == "state0 not aligned with basis"

        # Try to predict with bad time array.
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, np.vstack((t, t)), input_func)
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Predict without inputs.
        for form in MODEL_FORMS:
            if "B" not in form and "N" not in form:
                rom = _trainedmodel(self.ModelClass, form, Vr, None)
                out = rom.predict(q0, t, decompress=True)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, t.size)
                out = rom.predict(q0, t, decompress=False)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, t.size)

        # Predict with no basis gives result in low-dimensional space.
        rom = _trainedmodel(self.ModelClass, "cA", Vr, None)
        rom.basis = None
        out = rom.predict(Vr.T @ q0, t, decompress=True)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, t.size)

        # Try to predict with badly-shaped discrete inputs.
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, np.random.random((m - 1, nt)))
        assert (
            ex.value.args[0] == f"input_func.shape = {(m-1, nt)} "
            f"!= {(m, nt)} = (m, len(t))"
        )

        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, np.random.random((2, nt)))
        assert (
            ex.value.args[0] == f"input_func.shape = {(2, nt)} "
            f"!= {(1, nt)} = (m, len(t))"
        )

        # Try to predict with badly-shaped continuous inputs.
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, lambda t: np.ones(m - 1))
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            f"of shape (m,) = {(m,)}"
        )
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, lambda t: 1)
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            f"of shape (m,) = {(m,)}"
        )

        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m=1)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, input_func)
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            "of shape (m,) = (1,) or scalar"
        )

        # Try to predict with continuous inputs with bad return type
        rom = _trainedmodel(self.ModelClass, "cAHB", Vr, m)
        with pytest.raises(ValueError) as ex:
            rom.predict(q0, t, lambda t: set([5]))
        assert (
            ex.value.args[0] == "input_func() must return ndarray of "
            f"shape (m,) = {(m,)}"
        )

        for form in MODEL_FORMS:
            if "B" in form or "N" in form:
                # Predict with 2D inputs.
                rom = _trainedmodel(self.ModelClass, form, Vr, m)
                # continuous input.
                for method in ["RK45", "BDF"]:
                    out = rom.predict(
                        q0, t, input_func, decompress=True, method=method
                    )
                    assert isinstance(out, np.ndarray)
                    assert out.shape == (n, nt)
                # discrete input.
                out = rom.predict(q0, t, Upred, decompress=False)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)

                # Predict with 1D inputs.
                rom = _trainedmodel(self.ModelClass, form, Vr, 1)
                # continuous input.
                out = rom.predict(q0, t, lambda t: 1, decompress=True)
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, nt)
                out = rom.predict(
                    q0, t, lambda t: np.array([1]), decompress=False
                )
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                # discrete input.
                out = rom.predict(q0, t, np.ones_like(t))
                assert isinstance(out, np.ndarray)
                assert out.shape == (n, nt)
                assert hasattr(rom, "predict_result_")
