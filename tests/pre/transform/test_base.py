# pre/transform/test_base.py
"""Tests for rom_operator_inference.pre._base.py."""

import pytest
import numpy as np

import rom_operator_inference as opinf


class TestBaseTransformer:
    """Test pre.transform._base._BaseTransformer."""

    class Dummy(opinf.pre.transform._base._BaseTransformer):
        def fit_transform(self, states):
            self.n = states.shape[0]
            return states

        def transform(self, states):
            return states

        def inverse_transform(self, states):
            return states

    def test_fit(self):
        """Test pre.transform._base._BaseTransformer.fit()."""
        bt = self.Dummy()
        states = np.random.random((10, 5))
        out = bt.fit(states)
        assert out is bt
        assert hasattr(bt, "n")
        assert bt.n == 10

    def test_save(self):
        """Test pre.transform._base._BaseTransformer.save()."""
        with pytest.raises(NotImplementedError) as ex:
            self.Dummy().save("test")
        assert ex.value.args[0] == "use pickle/joblib"

    def test_load(self):
        """Test pre.transform._base._BaseTransformer.load()."""
        with pytest.raises(NotImplementedError) as ex:
            self.Dummy.load("test")
        assert ex.value.args[0] == "use pickle/joblib"


def test_check_is_transformer():
    """Test pre.transform._base._check_is_transformer()."""

    class Dummy:
        def transform(self):
            pass

        def fit_transform(self):
            pass

    with pytest.raises(TypeError) as ex:
        opinf.pre.transform._base._check_is_transformer(Dummy())
    assert ex.value.args[0] == \
        "transformer missing required method inverse_transform()"

    dummy = TestBaseTransformer.Dummy()
    opinf.pre.transform._base._check_is_transformer(dummy)
