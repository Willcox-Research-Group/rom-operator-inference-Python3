# pre/test_base.py
"""Tests for pre._base.py."""

import pytest
import numpy as np

import opinf


class TestTransformerTemplate:
    """Test pre._base.TransformerTemplate."""

    class Dummy(opinf.pre.TransformerTemplate):
        def fit_transform(self, states, inplace=False):
            return states if inplace else states.copy()

        def transform(self, states, inplace=False):
            return states if inplace else states.copy()

        def inverse_transform(self, states, inplace=False, locs=None):
            return states if inplace else states.copy()

    def test_name(self):
        """Test __init__(), name."""
        tf = self.Dummy()
        assert tf.name is None

        s1 = "the name"
        tf = self.Dummy(name=s1)
        assert tf.name == s1

        s2 = "new name"
        tf.name = s2
        assert tf.name == s2

        tf.name = None
        assert tf.name is None

    def test_state_dimension(self):
        """Test state_dimension."""
        tf = self.Dummy()
        assert tf.state_dimension is None
        tf.state_dimension = 10.0
        n = tf.state_dimension
        assert isinstance(n, int)
        assert tf.state_dimension == n
        tf.state_dimension = None
        assert tf.state_dimension is None

    def test_str(self):
        """Lightly test __str__()."""
        tf = self.Dummy()
        str(tf)
        tf.state_dimension = 10
        assert str(tf) in repr(tf)

    def test_fit(self):
        """Test fit()."""
        tf = self.Dummy()
        out = tf.fit(np.random.random((2, 5)))
        assert out is tf

    def test_verify(self, n=30):
        """Test verify()."""
        dummy = self.Dummy()

        with pytest.raises(AttributeError) as ex:
            dummy.verify()
        assert ex.value.args[0] == (
            "transformer not trained (state_dimension not set), "
            "call fit() or fit_transform()"
        )

        dummy.state_dimension = n
        dummy.verify()

        class Dummy1(self.Dummy):
            def __init__(self, name=None):
                super().__init__(name=name)
                self.state_dimension = n

        class Dummy2a(Dummy1):
            def transform(self, states, inplace=False):
                return states[:-1]

        class Dummy2b(Dummy1):
            def transform(self, states, inplace=False):
                return states

        class Dummy2c(Dummy1):
            def transform(self, states, inplace=False):
                return states.copy()

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2a().verify()
        assert ex.value.args[0] == "transform(states).shape != states.shape"

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2b().verify()
        assert ex.value.args[0] == "transform(states, inplace=False) is states"

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2c().verify()
        assert ex.value.args[0] == (
            "transform(states, inplace=True) is not states"
        )

        class Dummy3a(Dummy1):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed[:-1]

        class Dummy3b(Dummy1):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed

        class Dummy3c(Dummy1):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed.copy()

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3a().verify()
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)).shape != states.shape"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3b().verify()
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed, inplace=False) "
            "is states_transformed"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3c().verify()
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed, inplace=True) "
            "is not states_transformed"
        )

        class Dummy4(Dummy1):
            def inverse_transform(
                self,
                states_transformed,
                inplace=False,
                locs=None,
            ):
                if locs is None:
                    if inplace:
                        return states_transformed
                    return states_transformed.copy()
                return states_transformed[:-1]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy4().verify()
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)[locs], locs).shape "
            "!= states[locs].shape"
        )

        class Dummy5a(Dummy1):
            def inverse_transform(
                self,
                states_transformed,
                inplace=False,
                locs=None,
            ):
                Q = states_transformed
                if not inplace:
                    Q = Q.copy()
                Q += 1
                return Q

        class Dummy5b(Dummy1):
            def inverse_transform(
                self,
                states_transformed,
                inplace=False,
                locs=None,
            ):
                Q = states_transformed
                if not inplace:
                    Q = Q.copy()
                if locs is not None:
                    Q += 1
                return Q

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5a().verify()
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5b().verify()
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses "
            "(locs != None)"
        )

        class Dummy6a(Dummy1):
            def transform_ddts(self, ddts, inplace=False):
                return ddts

        class Dummy6b(Dummy1):
            def transform_ddts(self, ddts, inplace=False):
                dQ = ddts if inplace else ddts.copy()
                dQ += 1e16
                return dQ

        class Dummy6c(Dummy1):
            def transform_ddts(self, ddts, inplace=False):
                return ddts.copy()

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6a().verify()
        assert ex.value.args[0].startswith(
            "transform_ddts(ddts, inplace=False) is ddts"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6b().verify()
        assert ex.value.args[0].startswith(
            "transform_ddts() failed finite difference check"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6c().verify()
        assert ex.value.args[0].startswith(
            "transform_ddts(ddts, inplace=True) is not ddts"
        )

        class Dummy7(Dummy1):
            def transform_ddts(self, ddts, inplace=False):
                return ddts if inplace else ddts.copy()

        Dummy7().verify()
