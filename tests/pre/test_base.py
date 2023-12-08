# pre/test_base.py
"""Tests for pre._base.py."""

import pytest
import numpy as np

import opinf


class TestBaseTransformer:
    """Test pre._base._BaseTransformer."""

    class Dummy(opinf.pre._base._BaseTransformer):
        def fit_transform(self, states):
            self.n = states.shape[0]
            return states

        def transform(self, states):
            return states

        def inverse_transform(self, states):
            return states

    def test_fit(self):
        """Test pre._base._BaseTransformer.fit()."""
        bt = self.Dummy()
        states = np.random.random((10, 5))
        out = bt.fit(states)
        assert out is bt
        assert hasattr(bt, "n")
        assert bt.n == 10

    def test_save(self):
        """Test pre._base._BaseTransformer.save()."""
        with pytest.raises(NotImplementedError) as ex:
            self.Dummy().save("test")
        assert ex.value.args[0] == "use pickle/joblib"

    def test_load(self):
        """Test pre._base._BaseTransformer.load()."""
        with pytest.raises(NotImplementedError) as ex:
            self.Dummy.load("test")
        assert ex.value.args[0] == "use pickle/joblib"


class TestMultivarMixin:
    """Test for pre._base._MultivarMixin."""

    Mixin = opinf.pre._base._MultivarMixin

    def test_init(self, nvar=4):
        """Test _MultivarMixin.__init__()."""
        with pytest.raises(ValueError) as ex:
            self.Mixin(-1)
        assert ex.value.args[0] == "num_variables must be a positive integer"

        mix = self.Mixin(nvar)
        assert mix.num_variables == nvar
        assert mix.n is None
        assert mix.ni is None

    def test_variable_names(self, nvar=3):
        """Test _MultivarMixin.variable_names."""
        vnames = list("abcdefghijklmnopqrstuvwxyz")[:nvar]
        mix = self.Mixin(nvar, vnames)
        assert len(mix.variable_names) == nvar
        assert mix.variable_names == vnames

        mix.variable_names = None
        assert len(mix.variable_names) == nvar
        for name in mix.variable_names:
            assert name.startswith("variable ")

        with pytest.raises(TypeError) as ex:
            mix.variable_names = 1
        assert (
            ex.value.args[0]
            == f"variable_names must be a list of length {nvar}"
        )

        with pytest.raises(TypeError) as ex:
            mix.variable_names = vnames[:-1]
        assert (
            ex.value.args[0]
            == f"variable_names must be a list of length {nvar}"
        )

    def test_n_properties(self, nvar=5):
        """Test _MultivarMixin.n and ni."""
        mix = self.Mixin(nvar)
        assert mix.n is None
        assert mix.ni is None

        with pytest.raises(ValueError) as ex:
            mix.n = 2 * nvar - 1
        assert (
            ex.value.args[0] == "n must be evenly divisible by num_variables"
        )

        mix.n = nvar * 12
        assert mix.n == nvar * 12
        assert mix.ni == 12

        mix.n = nvar * 3
        assert mix.n == nvar * 3
        assert mix.ni == 3

    # Convenience methods -----------------------------------------------------
    def test_get_varslice(self):
        """Test _MultivarMixin.get_varslice()."""
        mix = self.Mixin(4, variable_names=list("abcd"))
        mix.n = 12
        s0 = mix.get_varslice(0)
        assert isinstance(s0, slice)
        assert s0.start == 0
        assert s0.stop == mix.ni
        s1 = mix.get_varslice(1)
        assert isinstance(s1, slice)
        assert s1.start == mix.ni
        assert s1.stop == 2 * mix.ni
        s2 = mix.get_varslice("c")
        assert isinstance(s2, slice)
        assert s2.start == 2 * mix.ni
        assert s2.stop == 3 * mix.ni

    def test_get_var(self):
        """Test _MultivarMixin.get_var()."""
        mix = self.Mixin(4, variable_names=list("abcd"))
        mix.n = 12
        q = np.random.random(mix.n)
        q0 = mix.get_var(0, q)
        assert q0.shape == (mix.ni,)
        assert np.all(q0 == q[: mix.ni])
        q1 = mix.get_var(1, q)
        assert q1.shape == (mix.ni,)
        assert np.all(q1 == q[mix.ni : 2 * mix.ni])
        q2 = mix.get_var("c", q)
        assert q2.shape == (mix.ni,)
        assert np.all(q2 == q[2 * mix.ni : 3 * mix.ni])

    def test_check_shape(self):
        """Test _MultivarMixin._check_shape()."""
        mix = self.Mixin(12)
        mix.n = 120
        X = np.random.randint(0, 100, (120, 23)).astype(float)
        mix._check_shape(X)

        with pytest.raises(ValueError) as ex:
            mix._check_shape(X[:-1])
        assert (
            ex.value.args[0]
            == "states.shape[0] = 119 != 12 * 10 = num_variables * n_i"
        )
