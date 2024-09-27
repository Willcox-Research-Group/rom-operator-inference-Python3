# pre/test_shiftscale.py
"""Tests for pre._shiftscale.py."""

import os
import h5py
import pytest
import itertools
import numpy as np

import opinf


# Data preprocessing: shifting and MinMax scaling / unscaling =================
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


# Transformer classes for centering and scaling ===============================
class TestShiftScaleTransformer:
    """Test pre.ShiftScaleTransformer."""

    Transformer = opinf.pre.ShiftScaleTransformer

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

    # Printing ----------------------------------------------------------------
    def test_str(self):
        """Test ShiftScaleTransformer.__str__()."""
        st = self.Transformer()
        assert str(st) == "ShiftScaleTransformer"

        st = self.Transformer(centering=True)
        trn = "(call fit() or fit_transform() to train)"
        msc = "ShiftScaleTransformer with mean-snapshot centering"
        assert str(st) == f"{msc} {trn}"
        for s in st._VALID_SCALINGS:
            st = self.Transformer(centering=True, scaling=s)
            assert str(st) == f"{msc} and '{s}' scaling {trn}"

        for s in st._VALID_SCALINGS:
            st = self.Transformer(centering=False, scaling=s)
            assert str(st) == f"ShiftScaleTransformer with '{s}' scaling {trn}"

        st = self.Transformer(centering=False, scaling=None)
        st.state_dimension = 100
        assert str(st) == "ShiftScaleTransformer (state_dimension = 100)"

        assert str(hex(id(st))) in repr(st)

    def test_statistics_report(self):
        """Test ShiftScaleTransformer._statistics_report()."""
        X = np.arange(10) - 4
        report = self.Transformer._statistics_report(X)
        assert report == "-4.000e+00 |  5.000e-01 |  5.000e+00 |  2.872e+00"

    # Persistence -------------------------------------------------------------
    def test_save(self, n=200, k=50):
        """Test ShiftScaleTransformer.save()."""
        # Clean up after old tests.
        target = "_savetransformertest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        def _checkfile(filename, st):
            assert os.path.isfile(filename)
            with h5py.File(filename, "r") as hf:
                # Check transformation metadata.
                assert "meta" in hf
                assert len(hf["meta"]) == 0
                for attr in ("centering", "scaling", "byrow", "verbose"):
                    assert attr in hf["meta"].attrs
                    if attr == "scaling" and st.scaling is None:
                        assert not hf["meta"].attrs[attr]
                    else:
                        assert hf["meta"].attrs[attr] == getattr(st, attr)

                # Check transformation parameters.
                if st.centering and hasattr(st, "mean_"):
                    assert "transformation/mean_" in hf
                    assert np.all(hf["transformation/mean_"][:] == st.mean_)
                if st.scaling and hasattr(st, "scale_"):
                    assert "transformation/scale_" in hf
                    assert np.all(hf["transformation/scale_"][:] == st.scale_)
                    assert "transformation/shift_" in hf
                    assert np.all(hf["transformation/shift_"][:] == st.shift_)

        # Check file creation and overwrite protocol on null transformation.
        st = self.Transformer()
        st.save(target)
        _checkfile(target, st)

        with pytest.raises(FileExistsError) as ex:
            st.save(target, overwrite=False)
        ex.value.args[0] == f"{target} (overwrite=True to ignore)"

        st.save(target, overwrite=True)
        _checkfile(target, st)

        # Check non-null transformations.
        X = np.random.randint(0, 100, (n, k)).astype(float)
        for scaling, centering, byrow in itertools.product(
            *[{None, *st._VALID_SCALINGS}, (True, False), (True, False)]
        ):
            st = self.Transformer(
                centering=centering,
                scaling=scaling,
                byrow=byrow if scaling else False,
                verbose=centering,
            )
            st.fit_transform(X)
            st.save(target, overwrite=True)
            _checkfile(target, st)

        os.remove(target)

    def test_load(self, n=200, k=50):
        """Test ShiftScaleTransformer.load()."""
        # Clean up after old tests.
        target = "_loadtransformertest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Check that save() -> load() gives the same transformer.
        X = np.random.randint(0, 100, (n, k)).astype(float)
        for scaling, centering, byrow in itertools.product(
            *[
                {None, *self.Transformer._VALID_SCALINGS},
                (True, False),
                (True, False),
            ]
        ):
            st = self.Transformer(
                centering=centering,
                scaling=scaling,
                byrow=byrow if scaling else False,
                verbose=not centering,
            )
            st.state_dimension = n
            st.fit_transform(X, inplace=False)
            st.save(target, overwrite=True)
            st2 = self.Transformer.load(target)
            assert st == st2

        os.remove(target)

    # Main routines -----------------------------------------------------------
    def test_check_shape(self, n=12):
        """Test ShiftScaleTransformerMulti._check_shape()."""
        stm = self.Transformer()
        stm.state_dimension = n
        X = np.random.randint(0, 100, (n, 2 * n)).astype(float)
        stm._check_shape(X)

        with pytest.raises(ValueError) as ex:
            stm._check_shape(X[:-1])
        assert ex.value.args[0] == (
            f"states.shape[0] = {n - 1} != {n} = state_dimension"
        )

    def test_is_trained(self, n=20):
        """Test ShiftScaleTransformer._is_trained()."""
        Q = np.random.random((n, 2 * n))
        # Null transformer is always trained.
        st = self.Transformer()
        assert st._is_trained() is True
        st = self.Transformer().fit(Q)
        assert st._is_trained() is True
        st._check_is_trained()

        # Centering.
        st = self.Transformer(centering=True)
        assert st._is_trained() is False
        st.mean_ = np.random.random(n)
        assert st._is_trained() is True

        # Scaling.
        st = self.Transformer(centering=False, scaling="minmax")
        with pytest.raises(AttributeError) as ex:
            st._check_is_trained()
        assert ex.value.args[0] == (
            "transformer not trained (call fit() or fit_transform())"
        )

        st.scale_ = 10
        assert st._is_trained() is False
        st.shift_ = 20
        assert st._is_trained() is True

        st = self.Transformer(centering=True, scaling="standard")
        assert st._is_trained() is False
        st.mean_ = np.random.random(n)
        assert st._is_trained() is False
        st.scale_ = np.random.random(n)
        assert st._is_trained() is False
        st.shift_ = np.random.random(n)
        assert st._is_trained() is True

    def test_verify(self, n=150, k=400):
        """Use ShiftScaleTransformer.verify() to run tests."""
        Q = np.random.random((n, k))

        for scaling, centering in itertools.product(
            {None, *self.Transformer._VALID_SCALINGS}, (True, False)
        ):
            st = self.Transformer(centering=centering, scaling=scaling)
            st.fit(Q)
            st.verify()

    def test_fit_transform(self, n=200, k=50):
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

    def test_transform(self, n=200, k=50):
        """Test ShiftScaleTransformer.transform()."""
        X = np.random.randint(0, 100, (n, k)).astype(float)
        st = self.Transformer(verbose=False)

        # Test null transformation.
        X = np.random.randint(0, 100, (n, k)).astype(float)
        st.fit_transform(X)
        Y = np.random.randint(0, 100, (n, k)).astype(float)
        Z = st.transform(Y, inplace=True)
        assert Z is Y
        Z = st.transform(Y, inplace=False)
        assert Z is not Y
        assert Z.shape == Y.shape
        assert np.all(Z == Y)

        # Test mean shift.
        st = self.Transformer(centering=True, scaling=None)
        with pytest.raises(AttributeError) as ex:
            st.transform(Y, inplace=False)
        assert ex.value.args[0] == (
            "transformer not trained (call fit() or fit_transform())"
        )
        st.fit_transform(X)
        µ = st.mean_
        Z = st.transform(Y, inplace=False)
        assert np.allclose(Z, Y - µ.reshape(-1, 1))

        # Test each scaling.
        for scl in st._VALID_SCALINGS:
            X = np.random.randint(0, 100, (n, k)).astype(float)
            Y = np.random.randint(0, 100, (n, k)).astype(float)
            st = self.Transformer(centering=False, scaling=scl)
            st.fit(X)
            a, b = st.scale_, st.shift_
            Z = st.transform(Y, inplace=False)
            assert np.allclose(Z, a * Y + b)

            # Test transforming a one-dimensional array.
            Y = np.random.randint(0, 100, n).astype(float)
            Z = st.transform(Y, inplace=False)
            assert Z.shape == Y.shape
            assert np.allclose(Z, a * Y + b)

    def test_inverse_transform(self, n=200, k=50):
        """Test ShiftScaleTransformer.inverse_transform()."""
        X = np.random.randint(0, 100, (n, k)).astype(float)
        st = self.Transformer(centering=True, verbose=False)

        with pytest.raises(AttributeError) as ex:
            st.inverse_transform(X, inplace=False)
        assert ex.value.args[0] == (
            "transformer not trained (call fit() or fit_transform())"
        )

        def _test_single(st, Y):
            Z = st.transform(Y, inplace=False)
            assert Z.shape == Y.shape
            st.inverse_transform(Z, inplace=True)
            assert Z.shape == Y.shape
            assert np.allclose(Z, Y)

        def _test_locs(st, Y, locs):
            Ylocs = Y[locs]
            Z = st.transform(Y, inplace=False)
            Zlocs = Z[locs]
            Ynew = st.inverse_transform(Zlocs, inplace=False, locs=locs)
            assert Ynew.shape == Ylocs.shape
            assert np.allclose(Ynew, Ylocs)
            st.inverse_transform(Zlocs, inplace=True, locs=locs)
            assert Zlocs.shape == Ylocs.shape
            assert np.allclose(Zlocs, Ylocs)

        for scaling, centering in itertools.product(
            {None, *st._VALID_SCALINGS}, (True, False)
        ):
            st = self.Transformer(centering=centering, scaling=scaling)
            st.fit_transform(X, inplace=False)
            locs = np.unique(np.sort(np.random.randint(0, n, n // 4)))

            # Test matrix of snapshots.
            Y = np.random.randint(0, 100, (n, k)).astype(float)
            _test_single(st, Y)
            _test_locs(st, Y, locs)

            # Test a single snapshot.
            locs = slice(n // 6)
            Y = np.random.randint(0, 100, n).astype(float)
            _test_single(st, Y)
            _test_locs(st, Y, locs)

        with pytest.raises(ValueError) as ex:
            st.inverse_transform(Y, locs=locs)
        assert ex.value.args[0] == "states_transformed not aligned with locs"
