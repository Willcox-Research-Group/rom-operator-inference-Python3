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
        """Test TransformerTemplate.__init__(), name."""
        mixin = self.Dummy()
        assert mixin.name is None

        s1 = "the name"
        mixin = self.Dummy(name=s1)
        assert mixin.name == s1

        s2 = "new name"
        mixin.name = s2
        assert mixin.name == s2

        mixin.name = None
        assert mixin.name is None

    def test_state_dimension(self):
        """Test TransformerTemplate.state_dimension."""
        mixin = self.Dummy()
        assert mixin.state_dimension is None
        mixin.state_dimension = 10.0
        n = mixin.state_dimension
        assert isinstance(n, int)
        assert mixin.state_dimension == n
        mixin.state_dimension = None
        assert mixin.state_dimension is None

    def test_fit(self):
        """Test TransformerTemplate.fit()."""
        tf = self.Dummy()
        out = tf.fit(np.random.random((2, 5)))
        assert out is tf

    def test_verify(self, n=30):
        """Test TransformerTemplate.verify()."""
        Dummy = self.Dummy
        dummy = Dummy()

        with pytest.raises(AttributeError) as ex:
            dummy.verify()
        assert ex.value.args[0] == (
            "transformer not trained (state_dimension not set), "
            "call fit() or fit_transform()"
        )

        dummy.state_dimension = n
        dummy.verify()

        class Dummy2a(Dummy):
            def __init__(self):
                Dummy.__init__(self)
                self.state_dimension = n

            def transform(self, states, inplace=False):
                return states[:-1]

        class Dummy2b(Dummy2a):
            def transform(self, states, inplace=False):
                return states

        class Dummy2c(Dummy2a):
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

        class Dummy3a(Dummy):
            def __init__(self):
                Dummy.__init__(self)
                self.state_dimension = n

            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed[:-1]

        class Dummy3b(Dummy3a):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed

        class Dummy3c(Dummy3a):
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

        class Dummy4(Dummy):
            def __init__(self):
                Dummy.__init__(self)
                self.state_dimension = n

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

        class Dummy5a(Dummy):
            def __init__(self):
                Dummy.__init__(self)
                self.state_dimension = n

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

        class Dummy5b(Dummy5a):
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

        class Dummy6a(Dummy):
            def __init__(self):
                Dummy.__init__(self)
                self.state_dimension = n

            def transform_ddts(self, ddts, inplace=False):
                return ddts

        class Dummy6b(Dummy6a):
            def transform_ddts(self, ddts, inplace=False):
                dQ = ddts if inplace else ddts.copy()
                dQ += 1e16
                return dQ

        class Dummy6c(Dummy6a):
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

        class Dummy7(Dummy):
            def __init__(self):
                Dummy.__init__(self)
                self.state_dimension = n

            def transform_ddts(self, ddts, inplace=False):
                return ddts if inplace else ddts.copy()

        Dummy7().verify()
