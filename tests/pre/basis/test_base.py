# pre/basis/test_base.py
"""Tests for rom_operator_inference.pre.basis._base."""

import pytest

import rom_operator_inference as opinf


class TestBaseBasis:
    """Test pre.basis._base._BaseBasis."""

    class Dummy(opinf.pre.basis._base._BaseBasis):
        """Instantiable version of _BaseBasis."""
        def fit(self):
            pass

        def encode(self, state):
            return state + 2

        def decode(self, state_):
            return state_ - 1

    class DummyTransformer:
        """Dummy object with all required transformer methods."""
        def fit_transform(self, states):
            return states

        def transform(self, states):
            return states

        def inverse_transform(self, states):
            return states

    def test_init(self):
        """Test _BaseBasis.__init__() and transformer properties."""
        self.Dummy()

        with pytest.raises(TypeError) as ex:
            self.Dummy(10)
        assert ex.value.args[0].startswith("transformer missing required meth")

        transformer = self.DummyTransformer()
        basis = self.Dummy(transformer)
        assert basis.transformer is transformer

    def test_project(self):
        """Test _BaseBasis.project() and _BaseBasis.projection_error()."""
        basis = self.Dummy()
        state = 5
        assert basis.project(state) == (state + 1)
        assert basis.projection_error(state, relative=False) == 1
        assert basis.projection_error(state, relative=True) == 1/state
