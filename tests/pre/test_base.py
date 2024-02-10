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

    def test_verify(self, n=30, k=16):
        """Test TransformerTemplate.verify()."""
        dummy = self.Dummy()
        q = np.random.random(n)

        with pytest.raises(ValueError) as ex:
            dummy.verify(q)
        assert ex.value.args[0] == (
            "two-dimensional states required for verification"
        )

        Q = np.random.random((n, k))
        dummy.verify(Q)

        class Dummy2a(self.Dummy):
            def transform(self, states, inplace=False):
                return states[:-1]

        class Dummy2b(self.Dummy):
            def transform(self, states, inplace=False):
                return states

        class Dummy2c(self.Dummy):
            def transform(self, states, inplace=False):
                return states.copy()

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2a().verify(Q)
        assert ex.value.args[0] == "transform(states).shape != states.shape"

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2b().verify(Q)
        assert ex.value.args[0] == "transform(states, inplace=False) is states"

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2c().verify(Q)
        assert ex.value.args[0] == (
            "transform(states, inplace=True) is not states"
        )

        class Dummy3a(self.Dummy):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed[:-1]

        class Dummy3b(self.Dummy):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed

        class Dummy3c(self.Dummy):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed.copy()

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3a().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)).shape != states.shape"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3b().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed, inplace=False) "
            "is states_transformed"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3c().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed, inplace=True) "
            "is not states_transformed"
        )

        class Dummy4(self.Dummy):
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
            Dummy4().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)[locs], locs).shape "
            "!= states[locs].shape"
        )

        class Dummy5a(self.Dummy):
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

        class Dummy5b(self.Dummy):
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
            Dummy5a().verify(Q)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5b().verify(Q)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses "
            "(locs != None)"
        )

        class Dummy6a(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                return ddts

        class Dummy6b(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                dQ = ddts if inplace else ddts.copy()
                dQ += 1e16
                return dQ

        class Dummy6c(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                return ddts.copy()

        with pytest.raises(ValueError) as ex:
            Dummy6a().verify(Q)
        assert ex.value.args[0] == (
            "time domain 't' required for finite difference check"
        )

        t = np.linspace(0, 1, k)
        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6a().verify(Q, t)
        assert ex.value.args[0].startswith(
            "transform_ddts(ddts, inplace=False) is ddts"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6b().verify(Q, t)
        assert ex.value.args[0].startswith(
            "transform_ddts() failed finite difference check"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6c().verify(Q, t)
        assert ex.value.args[0].startswith(
            "transform_ddts(ddts, inplace=True) is not ddts"
        )

        class Dummy7(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                return ddts if inplace else ddts.copy()

        Dummy7().verify(Q, t)
