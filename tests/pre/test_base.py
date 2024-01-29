# pre/test_base.py
"""Tests for pre._base.py."""

import pytest
import numpy as np

import opinf


class TestTransformerTemplate:
    """Test pre._base.TransformerTemplate."""

    class Dummy(opinf.pre.TransformerTemplate):
        def fit_transform(self, states):
            return states

        def transform(self, states):
            return states

        def inverse_transform(self, states, locs=None):
            return states

    def test_fit(self):
        """Test TransformerTemplate.fit()."""
        tf = self.Dummy()
        out = tf.fit(10)
        assert out is tf

    def test_verify(self, n=30, k=16):
        """Test TransformerTemplate.verify()."""
        Q = np.random.random((n, k))
        self.Dummy().verify(Q)

        class Dummy2(self.Dummy):
            def transform(self, states):
                return states[:-1]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2().verify(Q)
        assert ex.value.args[0] == "transform(states).shape != states.shape"

        class Dummy3(self.Dummy):
            def inverse_transform(self, states_transformed):
                return states_transformed[:-1]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)).shape "
            "!= transform(states).shape"
        )

        class Dummy4(self.Dummy):
            def inverse_transform(self, states_transformed, locs=None):
                if locs is None:
                    return states_transformed
                return states_transformed[:-1]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy4().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)[locs], locs).shape "
            "!= states[locs].shape"
        )

        class Dummy5(self.Dummy):
            def inverse_transform(self, states_transformed, locs=None):
                return states_transformed + 1

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5().verify(Q)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses"
        )

        class Dummy6(self.Dummy):
            def inverse_transform(self, states_transformed, locs=None):
                if locs is not None:
                    return states_transformed + 1
                return states_transformed

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6().verify(Q)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses "
            "(locs != None)"
        )

        class Dummy7(self.Dummy):
            def transform_ddts(self, ddts):
                return ddts + 1

        with pytest.raises(ValueError) as ex:
            Dummy7().verify(Q)
        assert ex.value.args[0] == (
            "time domain 't' required for finite difference check"
        )

        t = np.linspace(0, 1, k)
        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy7().verify(Q, t)
        assert ex.value.args[0].startswith(
            "transform_ddts() failed finite difference check"
        )

        class Dummy8(self.Dummy):
            def transform_ddts(self, ddts):
                return ddts

        Dummy8().verify(Q, t)


class TestUnivarMixin:
    """Tests for pre._base._UnivarMixin."""

    Mixin = opinf.pre._base._UnivarMixin

    def test_state_dimension(self):
        """Test _UnivarMixin.state_dimension."""
        mixin = self.Mixin()
        assert mixin.state_dimension is None
        mixin.state_dimension = 10.0
        n = mixin.state_dimension
        assert isinstance(n, int)
        assert mixin.state_dimension == n


class TestMultivarMixin:
    """Tests for pre._base._MultivarMixin."""

    Mixin = opinf.pre._base._MultivarMixin

    def test_init(self, nvar=4):
        """Test _MultivarMixin.__init__()."""
        with pytest.raises(TypeError) as ex:
            self.Mixin(-1)
        assert ex.value.args[0] == "'num_variables' must be a positive integer"

        mix = self.Mixin(nvar)
        assert mix.num_variables == nvar
        assert mix.state_dimension is None
        assert mix.variable_size is None

        assert len(mix) == nvar

    def test_variable_names(self, nvar=3):
        """Test _MultivarMixin.variable_names."""
        vnames = list("abcdefghijklmnopqrstuvwxyz")[:nvar]
        mix = self.Mixin(nvar, vnames)
        assert len(mix.variable_names) == nvar
        for i in range(nvar):
            assert mix.variable_names[i] == vnames[i]

        mix.variable_names = None
        assert len(mix.variable_names) == nvar
        for name in mix.variable_names:
            assert name.startswith("variable ")

        with pytest.raises(ValueError) as ex:
            mix.variable_names = vnames[:-1]
        assert ex.value.args[0] == f"variable_names must have length {nvar}"

    def test_properties(self, nvar=5):
        """Test _MultivarMixin.state_dimension and variable_size."""
        mix = self.Mixin(nvar)
        assert mix.state_dimension is None
        assert mix.variable_size is None

        with pytest.raises(ValueError) as ex:
            mix.state_dimension = 2 * nvar - 1
        assert ex.value.args[0] == (
            "'state_dimension' must be evenly divisible by 'num_variables'"
        )

        for size in np.random.randint(3, 10, size=5):
            mix.state_dimension = (n := nvar * size)
            assert mix.state_dimension == n
            assert mix.variable_size == size

    # Convenience methods -----------------------------------------------------
    def test_check_shape(self, nvar=12, nx=10, k=20):
        """Test _MultivarMixin._check_shape()."""
        mix = self.Mixin(nvar)
        mix.state_dimension = (n := nvar * nx)
        X = np.random.randint(0, 100, (n, k)).astype(float)
        mix._check_shape(X)

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            mix._check_shape(X[:-1])
        assert ex.value.args[0] == (
            f"states.shape[0] = {n - 1:d} != {nvar:d} * {nx:d} "
            "= num_variables * variable_size = state_dimension"
        )

    def test_get_var(self, nvar=4, nx=9, k=3):
        """Test _MultivarMixin.get_var()."""
        mix = self.Mixin(nvar, variable_names="abcdefghijklmnop"[:nvar])
        mix.state_dimension = (n := nvar * nx)
        q = np.random.random(n)

        q0 = mix.get_var(0, q)
        assert q0.shape == (nx,)
        assert np.all(q0 == q[:nx])

        q1 = mix.get_var(1, q)
        assert q1.shape == (nx,)
        assert np.all(q1 == q[nx : (2 * nx)])

        q2 = mix.get_var("c", q)
        assert q2.shape == (nx,)
        assert np.all(q2 == q[2 * nx : 3 * nx])
