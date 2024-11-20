# pre/test_shiftscale.py
"""Tests for pre._shiftscale.py."""

import pytest
import itertools
import numpy as np

import opinf

try:
    from .test_base import _TestTransformer
except ImportError:
    from test_base import _TestTransformer


# Functions ===================================================================
def test_shift(set_up_transformer_data):
    """Test pre._shift_scale.shift()."""
    X = set_up_transformer_data

    # Try with bad data shape.
    with pytest.raises(ValueError) as ex:
        opinf.pre.shift(np.random.random((3, 3, 3)))
    assert ex.value.args[0] == "'states' must be two-dimensional"

    # Try with bad shift vector.
    with pytest.raises(ValueError) as ex:
        opinf.pre.shift(X, X)
    assert ex.value.args[0] == "'shift_by' must be one-dimensional"

    # Correct usage.
    Xshifted, xbar = opinf.pre.shift(X)
    assert xbar.shape == (X.shape[0],)
    assert Xshifted.shape == X.shape
    assert np.allclose(np.mean(Xshifted, axis=1), np.zeros(X.shape[0]))
    for j in range(X.shape[1]):
        assert np.allclose(Xshifted[:, j], X[:, j] - xbar)

    Y = np.random.random(X.shape)
    Yshifted = opinf.pre.shift(Y, xbar.reshape(-1, 1))
    for j in range(Y.shape[1]):
        assert np.allclose(Yshifted[:, j], Y[:, j] - xbar)

    # Verify inverse shifting.
    assert np.allclose(X, opinf.pre.shift(Xshifted, -xbar))


def test_scale(set_up_transformer_data):
    """Test pre._shift_scale.scale()."""
    X = set_up_transformer_data

    # Try with bad scales.
    with pytest.raises(ValueError) as ex:
        opinf.pre.scale(X, (1, 2, 3), (4, 5))
    assert ex.value.args[0] == "scale_to must have exactly 2 elements"

    with pytest.raises(ValueError) as ex:
        opinf.pre.scale(X, (1, 2), (3, 4, 5))
    assert ex.value.args[0] == "scale_from must have exactly 2 elements"

    # Scale X to [-1, 1] and then scale Y with the same transformation.
    Xscaled, scaled_to, scaled_from = opinf.pre.scale(X, (-1, 1))
    assert Xscaled.shape == X.shape
    assert scaled_to == (-1, 1)
    assert isinstance(scaled_from, tuple)
    assert len(scaled_from) == 2
    assert round(scaled_from[0], 8) == round(X.min(), 8)
    assert round(scaled_from[1], 8) == round(X.max(), 8)
    assert round(Xscaled.min(), 8) == -1
    assert round(Xscaled.max(), 8) == 1

    # Verify inverse scaling.
    assert np.allclose(opinf.pre.scale(Xscaled, scaled_from, scaled_to), X)


# Transformer classes =========================================================
class TestShiftTransformer(_TestTransformer):
    Transformer = opinf.pre.ShiftTransformer
    requires_training = False
    statedim = 20

    def get_transformers(self, name=None):
        yield self.Transformer(np.random.random(self.statedim), name=name)

    def test_init(self):
        """Test __init__() and the reference property."""

        with pytest.raises(TypeError) as ex:
            self.Transformer("moose")
        assert ex.value.args[0] == (
            "reference snapshot must be a one-dimensional array"
        )

        tf = self.get_transformer()
        assert tf.state_dimension == self.statedim
        assert isinstance(tf.reference, np.ndarray)
        assert tf.reference.shape == (tf.state_dimension,)

        with pytest.raises(AttributeError) as ex:
            tf.state_dimension = self.statedim + 2
        assert ex.value.args[0] == (
            "can't set attribute 'state_dimension' to "
            f"{self.statedim + 2} != {self.statedim} = reference.size"
        )


class TestScaleTransformer(_TestTransformer):
    Transformer = opinf.pre.ScaleTransformer
    requires_training = None
    statedim = 21

    def get_transformers(self, name=None):
        self.requires_training = True
        yield self.Transformer(np.random.random(), name=name)
        self.requires_training = False
        yield self.Transformer(np.random.random(self.statedim), name=name)

    def test_init(self):
        """Test __init__() and the scale property."""

        with pytest.raises(TypeError) as ex:
            self.Transformer("bison")
        assert ex.value.args[0] == (
            "scaler must be a nonzero scalar or one-dimensional array"
        )

        tf = self.Transformer(10)
        assert tf.state_dimension is None
        assert tf.scaler == 10

        tf = self.Transformer(np.random.random(self.statedim))
        assert tf.state_dimension == self.statedim
        assert isinstance(tf.scaler, np.ndarray)

        with pytest.raises(AttributeError) as ex:
            tf.state_dimension = self.statedim - 1
        assert ex.value.args[0] == (
            "can't set attribute 'state_dimension' to "
            f"{self.statedim - 1} != {self.statedim} = scaler.size"
        )


class TestShiftScaleTransformer(_TestTransformer):
    """Test pre.ShiftScaleTransformer."""

    Transformer = opinf.pre.ShiftScaleTransformer
    requires_training = True

    def get_transformers(self, name=None):
        for scaling, centering in itertools.product(
            {None, *self.Transformer._VALID_SCALINGS},
            (True, False),
        ):
            if scaling is None and centering is False:
                self.requires_training = False
            yield self.Transformer(
                centering=centering,
                scaling=scaling,
                byrow=False,
                name=name,
                verbose=False,
            )
            self.requires_training = True
            if scaling is not None and scaling != "maxnorm":
                # "maxnorm" scaling is incompatible with byrow=True
                yield self.Transformer(
                    centering=centering,
                    scaling=scaling,
                    byrow=True,
                    name=name,
                    verbose=True,
                )

    def test_init(self, n=10):
        """Test ShiftScaleTransformer.__init__()."""
        st = self.Transformer()
        for attr in [
            "scaling",
            "centering",
            "verbose",
            "state_dimension",
        ]:
            assert hasattr(st, attr)
        assert st.state_dimension is None

        # Test centering.
        st = self.Transformer(centering=False)
        with pytest.raises(AttributeError) as ex:
            st.mean_ = 100
        assert ex.value.args[0] == "cannot set mean_ (centering=False)"

        st = self.Transformer(centering=True, scaling=None)
        with pytest.raises(ValueError) as ex:
            st.mean_ = 10
        assert ex.value.args[0] == "expected one-dimensional mean_"
        st.mean_ = (qbar := np.random.random(n))
        assert st.state_dimension == n
        assert np.all(st.mean_ == qbar)
        with pytest.raises(ValueError) as ex:
            st.mean_ = 100
        assert ex.value.args[0] == f"expected mean_ to be ({n:d},) ndarray"

        # Test scaling.
        for attr in "scale_", "shift_":
            with pytest.raises(AttributeError) as ex:
                setattr(st, attr, 100)
            assert ex.value.args[0] == f"cannot set {attr} (scaling=None)"

        with pytest.raises(TypeError) as ex:
            self.Transformer(scaling=[1, 2])
        assert ex.value.args[0] == "'scaling' must be None or of type 'str'"

        with pytest.raises(ValueError) as ex:
            self.Transformer(scaling="minimusmaximus")
        assert ex.value.args[0].startswith("invalid scaling 'minimusmaximus'")

        for s in st._VALID_SCALINGS:
            st = self.Transformer(scaling=s)
            st.scale_ = 10
            st.shift_ = 1

        # Test byrow.
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Transformer(scaling=None, byrow=True)
        assert wn[0].message.args[0] == (
            "scaling=None --> byrow=True will have no effect"
        )

        for attr in "scale_", "shift_":
            st = self.Transformer(scaling="standard", byrow=True)
            with pytest.raises(ValueError) as ex:
                setattr(st, attr, 10)
            assert ex.value.args[0] == f"expected one-dimensional {attr}"
            setattr(st, attr, np.random.random(n))
            assert st.state_dimension == n
            with pytest.raises(ValueError) as ex:
                setattr(st, attr, 100)
            assert ex.value.args[0] == f"expected {attr} to be ({n},) ndarray"

        # Test verbose.
        st.verbose = 0
        assert st.verbose is False
        st.verbose = "sure"
        assert st.verbose is True

    def test_eq(self, n=200):
        """Test ShiftScaleTransformer.__eq__()."""
        µ = np.random.randint(0, 100, (n,))
        a, b = 10, -3

        # Null transformers.
        st1 = self.Transformer()
        st2 = self.Transformer()
        assert st1 == st2
        assert st1 != 100

        # Mismatched attributes.
        st1 = self.Transformer(centering=True)
        st2 = self.Transformer(centering=False)
        assert not (st1 == st2)
        assert st1 != st2
        st2 = self.Transformer(centering=True)

        # Mismatched dimensions.
        st1.state_dimension = n
        st2.state_dimension = n + 2
        assert not (st1 == st2)
        assert st1 != st2
        st2.state_dimension = n

        # Centering attributes.
        st1.mean_ = µ
        assert st1 != st2
        st2.mean_ = µ
        assert st1 == st2
        st2.mean_ = µ - 5
        assert st1 != st2
        st2.mean_ = µ

        # Scaling attributes.
        st1 = self.Transformer(scaling="standard")
        st2 = self.Transformer(scaling=None)
        assert st1 != st2
        st2 = self.Transformer(scaling="minmax")
        assert st1 != st2
        st2 = self.Transformer(scaling="standard")
        assert st1 == st2
        st1.scale_, st1.shift_ = a, b
        assert st1 != st2
        st2.scale_, st2.shift_ = a - 1, b + 1
        assert st1 != st2
        st2.scale_, st2.shift_ = a, b
        assert st1 == st2

    def test_statistics_report(self):
        """Test ShiftScaleTransformer._statistics_report()."""
        X = np.arange(10) - 4
        report = self.Transformer._statistics_report(X)
        assert report == "-4.000e+00 |  5.000e-01 |  5.000e+00 |  2.872e+00"

    def test_transformation_types(self, n=80, k=39):
        """Test ShiftScaleTransformer.fit_transform()."""

        def fit_transform_copy(st, A):
            """Assert A and B are not the same object but do have the same
            type and shape.
            """
            B = st.fit_transform(A, inplace=False)
            assert B is not A
            assert type(B) is type(A)
            assert B.shape == A.shape
            return B

        # Test dimension check.
        st = self.Transformer(centering=False, scaling=None, byrow=False)
        with pytest.raises(ValueError) as ex:
            st.fit_transform(np.zeros(10))
        assert ex.value.args[0] == "2D array required to fit transformer"

        # Test null transformation.
        X = np.random.randint(0, 100, (n, k)).astype(float)
        Y = st.fit_transform(X, inplace=True)
        assert Y is X
        Y = fit_transform_copy(st, X)
        assert np.all(Y == X)

        # Test centering.
        st = self.Transformer(centering=True, name="VaRiAbLe", verbose=True)
        Y = fit_transform_copy(st, X)
        assert isinstance(st.mean_, np.ndarray)
        assert st.mean_.shape == (X.shape[0],)
        assert np.allclose(np.mean(Y, axis=1), 0)

        # Test scaling (without and with centering).
        for centering in (False, True):
            st = self.Transformer(centering=centering, scaling="standard")

            # Test standard scaling.
            Y = fit_transform_copy(st, X)
            for attr in "scale_", "shift_":
                assert isinstance(getattr(st, attr), float)
            assert np.isclose(np.mean(Y), 0)
            assert np.isclose(np.std(Y), 1)

            # Test min-max scaling.
            st = self.Transformer(centering=centering, scaling="minmax")
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.min(Y), 0)
            assert np.isclose(np.max(Y), 1)

            # Test symmetric min-max scaling.
            st = self.Transformer(centering=centering, scaling="minmaxsym")
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.min(Y), -1)
            assert np.isclose(np.max(Y), 1)

            # Test maximum absolute scaling.
            st = self.Transformer(centering=centering, scaling="maxabs")
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.max(np.abs(Y)), 1)

            # Test minimum-maximum absolute scaling.
            st = self.Transformer(centering=centering, scaling="maxabssym")
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.mean(Y), 0)
            assert np.isclose(np.max(np.abs(Y)), 1)

            # Test norm scaling.
            st = self.Transformer(centering=centering, scaling="maxnorm")
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.max(np.linalg.norm(Y, axis=0)), 1)

        # Test scaling by row (without and with centering).
        for centering in (False, True):
            # Test standard scaling.
            st = self.Transformer(
                centering=centering,
                scaling="standard",
                byrow=True,
            )
            Y = fit_transform_copy(st, X)
            for attr in "scale_", "shift_":
                assert isinstance(getattr(st, attr), np.ndarray)
            assert np.allclose(np.mean(Y, axis=1), 0)
            assert np.allclose(np.std(Y, axis=1), 1)

            # Test min-max scaling.
            st = self.Transformer(
                centering=centering,
                scaling="minmax",
                byrow=True,
            )
            Y = fit_transform_copy(st, X)
            assert np.allclose(np.min(Y, axis=1), 0)
            assert np.allclose(np.max(Y, axis=1), 1)

            # Test symmetric min-max scaling.
            st = self.Transformer(
                centering=centering,
                scaling="minmaxsym",
                byrow=True,
            )
            Y = fit_transform_copy(st, X)
            assert np.allclose(np.min(Y, axis=1), -1)
            assert np.allclose(np.max(Y, axis=1), 1)

            # Test maximum absolute scaling.
            st = self.Transformer(
                centering=centering,
                scaling="maxabs",
                byrow=True,
            )
            Y = fit_transform_copy(st, X)
            assert np.allclose(np.max(np.abs(Y), axis=1), 1)

            # Test minimum-maximum absolute scaling.
            st = self.Transformer(
                centering=centering,
                scaling="maxabssym",
                byrow=True,
            )
            Y = fit_transform_copy(st, X)
            assert np.allclose(np.mean(Y, axis=1), 0)
            assert np.allclose(np.max(np.abs(Y), axis=1), 1)

            # Test norm scaling.
            with pytest.raises(ValueError) as ex:
                self.Transformer(
                    centering=centering,
                    scaling="maxnorm",
                    byrow=True,
                )
            assert ex.value.args[0] == (
                "scaling 'maxnorm' is invalid when byrow=True"
            )

    def test_mains(self, n=11, k=21):
        """Test fit(), fit_transform(), transform(), transform_ddts(), and
        inverse_transform().
        """

        Q = np.random.random((n, k))
        for tf in (
            self.Transformer(centering=True),
            self.Transformer(scaling="standard"),
        ):
            for method in (
                tf.transform,
                tf.inverse_transform,
                tf.transform_ddts,
            ):
                with pytest.raises(AttributeError) as ex:
                    method(Q)
                assert ex.value.args[0] == (
                    "transformer not trained, call fit() or fit_transform()"
                )

        return super().test_mains(n, k)


if __name__ == "__main__":
    pytest.main([__file__])
