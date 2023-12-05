# models/nonparametric/test_public.py
"""Tests for models.nonparametric._public."""

import pytest
import numpy as np

import opinf

from .. import MODEL_FORMS, _get_data, _get_operators, _trainedmodel


class TestSteadyModel:
    """Test models.nonparametric._public.SteadyModel."""

    ModelClass = opinf.models.nonparametric._public.SteadyModel

    def test_rhs(self, r=10):
        """Lightly test SteadyModel.rhs().
        Stronger tests in test_base.TestNonparametricModel.test_rhs().
        """
        model = _trainedmodel(self.ModelClass, "cAH", r, 0)
        model.rhs(np.random.random(r))

    def test_jacobian(self, r=6, m=3):
        """Lightly test DiscreteModel.jacobian().
        Stronger tests in test_base.TestNonparametricModel.test_jacobian().
        """
        model = _trainedmodel(self.ModelClass, "A", r, 0)
        model.jacobian(np.random.random(r))

    def test_fit(self, k=400, r=10):
        """Lightly test SteadyModel.fit().
        Stronger tests in test_base.TestNonparametricModel.test_fit().
        """
        Q, F, _ = _get_data(r, k, 2)
        model = self.ModelClass("A")
        model.fit(Q, F)

    # def test_predict(self):
    #     """Test SteadyModel.predict()."""
    #     raise NotImplementedError


class TestDiscreteModel:
    """Test models.nonparametric._public.DiscreteModel."""

    ModelClass = opinf.models.nonparametric._public.DiscreteModel

    def test_stack_trajectories(self, r=10, k=20, m=5, num_trajectories=4):
        """Test DiscreteModel.stack_trajectories()."""
        statelist, inputlist = [], []
        for _ in range(num_trajectories):
            Q, _, U = _get_data(r, k, m)
            statelist.append(Q)
            inputlist.append(U)

        Qs, Qnexts = self.ModelClass.stack_trajectories(statelist)
        assert Qs.shape == (r, (k - 1) * num_trajectories)
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

    def test_rhs(self, r=6, m=3):
        """Lightly test DiscreteModel.rhs().
        Stronger tests in test_base.TestNonparametricModel.test_rhs().
        """
        model = _trainedmodel(self.ModelClass, "cG", r, 0)
        model.rhs(np.random.random(r))

    def test_jacobian(self, r=6, m=3):
        """Lightly test DiscreteModel.jacobian().
        Stronger tests in test_base.TestNonparametricModel.test_jacobian().
        """
        model = _trainedmodel(self.ModelClass, "cG", r, 0)
        model.jacobian(np.random.random(r))

    def test_fit(self, k=50, r=5, m=3):
        """Test DiscreteModel.fit()."""
        Q, _, U = _get_data(r, k, m)
        Qnext = Q[:, 1:]
        model = self.ModelClass("A").fit(Q)
        model2 = self.ModelClass("A").fit(Q[:, :-1], Qnext)
        assert model.A_ == model2.A_

        model = self.ModelClass("AB").fit(Q, inputs=U)
        model2 = self.ModelClass("AB").fit(Q[:, :-1], Qnext, inputs=U)
        assert model.A_ == model2.A_
        assert model.B_ == model2.B_

    def test_predict(self, k=20, m=6, r=4):
        """Test DiscreteModel.predict()."""
        # Get test data.
        Q = _get_data(r, k, m)[0]
        niters = 5
        q0 = Q[:, 0]
        U = np.ones((m, niters - 1))

        ops = A_, B_ = _get_operators("AB", r, m)
        B1d_ = _get_operators("B", r, m=1)[0]
        model = self.ModelClass(ops)

        # Try to predict with invalid initial condition.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0[1:], None)
        assert ex.value.args[0] == (
            "initial condition not aligned with model "
            f"(state0.shape = ({r-1},) != ({r},) = (r,))"
        )

        # Try to predict with bad niters argument.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, -18, U)
        assert (
            ex.value.args[0] == "argument 'niters' must be a positive integer"
        )

        # Try to predict with badly-shaped discrete inputs.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, niters, np.random.random((m - 1, niters - 1)))
        assert (
            ex.value.args[0] == f"inputs.shape = ({(m-1, niters-1)} "
            f"!= {(m, niters-1)} = (m, niters-1)"
        )

        model_m1 = self.ModelClass([A_, B1d_])
        with pytest.raises(ValueError) as ex:
            model_m1.predict(q0, niters, np.random.random((2, niters - 1)))
        assert (
            ex.value.args[0] == f"inputs.shape = ({(2, niters-1)} "
            f"!= {(1, niters-1)} = (m, niters-1)"
        )

        # Try to predict with continuous inputs.
        with pytest.raises(TypeError) as ex:
            model.predict(q0, niters, lambda t: np.ones(m - 1))
        assert ex.value.args[0] == "inputs must be NumPy array, not callable"

        # No control inputs.
        model = self.ModelClass([A_])
        out = model.predict(q0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)

        # With 2D inputs.
        model = self.ModelClass([A_, B_])
        out = model.predict(q0, niters, U)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)

        # With 1D inputs.
        model = self.ModelClass([A_, B1d_])
        out = model.predict(q0, niters, np.ones(niters))
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)


class TestContinuousModel:
    """Test models.nonparametric._public.ContinuousModel."""

    ModelClass = opinf.models.nonparametric._public.ContinuousModel

    def test_rhs(self, r=5, m=2):
        """Test ContinuousModel.rhs()."""
        A_, B_ = _get_operators("AB", r, m)

        model = self.ModelClass([A_])
        q_ = np.random.random(r)
        model.rhs(10, q_)

        model = self.ModelClass([A_, B_])
        with pytest.raises(TypeError) as ex:
            model.rhs(5, q_, 10)
        assert "object is not callable" in ex.value.args[0]

        def input_func(t):
            return np.random.random(m)

        model.rhs(np.pi, q_, input_func)

    def test_jacobian(self, r=6, m=3):
        """Test ContinuousModel.jacobian()."""
        A_, B_ = _get_operators("AB", r, m)

        model = self.ModelClass([A_])
        q_ = np.random.random(r)
        model.jacobian(8, q_)

        model = self.ModelClass([A_, B_])
        with pytest.raises(TypeError) as ex:
            model.jacobian(5, q_, 10)
        assert "object is not callable" in ex.value.args[0]

        def input_func(t):
            return np.random.random(m)

        model.jacobian(2, q_, input_func)

    def test_fit(self, k=20, m=3, r=4):
        """Lightly test ContinuousModel.fit().
        Stronger tests in test_base.TestNonparametricModel.test_fit().
        """
        Q, Qdot, U = _get_data(r, k, m)
        self.ModelClass("AB").fit(Q, Qdot, U)

    def test_predict(self, k=50, m=10, r=6):
        """Test ContinuousModel.predict()."""
        # Get test data.
        Q = _get_data(r, k, m)[0]
        nt = 5
        q0 = Q[:, 0]
        t = np.linspace(0, 0.01 * nt, nt)

        def input_func(tt):
            return tt * np.ones(m)

        Upred = np.column_stack([input_func(tt) for tt in t])

        # Try to predict with invalid initial condition.
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0[1:], t, input_func)
        assert ex.value.args[0] == (
            "initial condition not aligned with model "
            f"(state0.shape = ({r-1},) != ({r},) = (r,))"
        )

        # Try to predict with bad time array.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, np.vstack((t, t)), input_func)
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Predict without inputs.
        for form in MODEL_FORMS:
            if "B" not in form and "N" not in form:
                model = _trainedmodel(self.ModelClass, form, r, None)
                out = model.predict(q0, t)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, t.size)

        # Predict with no basis gives result in low-dimensional space.
        model = _trainedmodel(self.ModelClass, "cA", r, None)
        model.basis = None
        out = model.predict(q0, t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, t.size)

        # Try to predict with badly-shaped discrete inputs.
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, np.random.random((m - 1, nt)))
        assert (
            ex.value.args[0] == f"input_func.shape = {(m-1, nt)} "
            f"!= {(m, nt)} = (m, len(t))"
        )

        model = _trainedmodel(self.ModelClass, "cAHB", r, m=1)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, np.random.random((2, nt)))
        assert (
            ex.value.args[0] == f"input_func.shape = {(2, nt)} "
            f"!= {(1, nt)} = (m, len(t))"
        )

        # Try to predict with badly-shaped continuous inputs.
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, lambda t: np.ones(m - 1))
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            f"of shape (m,) = {(m,)}"
        )
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, lambda t: 1)
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            f"of shape (m,) = {(m,)}"
        )

        model = _trainedmodel(self.ModelClass, "cAHB", r, m=1)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, input_func)
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            "of shape (m,) = (1,) or scalar"
        )

        # Try to predict with continuous inputs with bad return type
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, lambda t: set([5]))
        assert (
            ex.value.args[0] == "input_func() must return ndarray of "
            f"shape (m,) = {(m,)}"
        )

        for form in MODEL_FORMS:
            if "B" in form or "N" in form:
                # Predict with 2D inputs.
                model = _trainedmodel(self.ModelClass, form, r, m)
                # continuous input.
                for method in ["RK45", "BDF"]:
                    out = model.predict(q0, t, input_func, method=method)
                    assert isinstance(out, np.ndarray)
                    assert out.shape == (r, nt)
                # discrete input.
                out = model.predict(q0, t, Upred)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)

                # Predict with 1D inputs.
                model = _trainedmodel(self.ModelClass, form, r, 1)
                # continuous input.
                out = model.predict(q0, t, lambda t: 1)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                out = model.predict(q0, t, lambda t: np.array([1]))
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                # discrete input.
                out = model.predict(q0, t, np.ones_like(t))
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                assert hasattr(model, "predict_result_")
