# pre/test_shift_scale.py
"""Tests for rom_operator_inference.pre._shift_scale.py."""

import os
import h5py
import pytest
import itertools
import numpy as np

import rom_operator_inference as opinf


# Data preprocessing: shifting and MinMax scaling / unscaling =================
def test_shift(set_up_basis_data):
    """Test pre._shift_scale.shift()."""
    X = set_up_basis_data

    # Try with bad data shape.
    with pytest.raises(ValueError) as ex:
        opinf.pre.shift(np.random.random((3,3,3)))
    assert ex.value.args[0] == "data X must be two-dimensional"

    # Try with bad shift vector.
    with pytest.raises(ValueError) as ex:
        opinf.pre.shift(X, X)
    assert ex.value.args[0] == "shift_by must be one-dimensional"

    # Correct usage.
    Xshifted, xbar = opinf.pre.shift(X)
    assert xbar.shape == (X.shape[0],)
    assert Xshifted.shape == X.shape
    assert np.allclose(np.mean(Xshifted, axis=1), np.zeros(X.shape[0]))
    for j in range(X.shape[1]):
        assert np.allclose(Xshifted[:,j], X[:,j] - xbar)

    Y = np.random.random(X.shape)
    Yshifted = opinf.pre.shift(Y, xbar)
    for j in range(Y.shape[1]):
        assert np.allclose(Yshifted[:,j], Y[:,j] - xbar)

    # Verify inverse shifting.
    assert np.allclose(X, opinf.pre.shift(Xshifted, -xbar))


def test_scale(set_up_basis_data):
    """Test pre._shift_scale.scale()."""
    X = set_up_basis_data

    # Try with bad scales.
    with pytest.raises(ValueError) as ex:
        opinf.pre.scale(X, (1,2,3), (4,5))
    assert ex.value.args[0] == "scale_to must have exactly 2 elements"

    with pytest.raises(ValueError) as ex:
        opinf.pre.scale(X, (1,2), (3,4,5))
    assert ex.value.args[0] == "scale_from must have exactly 2 elements"

    # Scale X to [-1,1] and then scale Y with the same transformation.
    Xscaled, scaled_to, scaled_from = opinf.pre.scale(X, (-1,1))
    assert Xscaled.shape == X.shape
    assert scaled_to == (-1,1)
    assert isinstance(scaled_from, tuple)
    assert len(scaled_from) == 2
    assert round(scaled_from[0],8) == round(X.min(),8)
    assert round(scaled_from[1],8) == round(X.max(),8)
    assert round(Xscaled.min(),8) == -1
    assert round(Xscaled.max(),8) == 1

    # Verify inverse scaling.
    assert np.allclose(opinf.pre.scale(Xscaled, scaled_from, scaled_to), X)


# Transformer classes for centering and scaling ===============================
class TestSnapshotTransformer:
    """Test pre.SnapshotTransformer."""
    def test_init(self):
        """Test pre.SnapshotTransformer.__init__()."""
        st = opinf.pre.SnapshotTransformer()
        for attr in ["scaling", "center", "verbose"]:
            assert hasattr(st, attr)

    # Properties --------------------------------------------------------------
    def test_properties(self):
        """Test pre.SnapshotTransformer properties (attribute protection)."""
        st = opinf.pre.SnapshotTransformer()

        # Test center.
        with pytest.raises(TypeError) as ex:
            st.center = "nope"
        assert ex.value.args[0] == "'center' must be True or False"
        st.center = True
        st.center = False

        # Test scale.
        with pytest.raises(ValueError) as ex:
            st.scaling = "minimaxii"
        assert ex.value.args[0].startswith("invalid scaling 'minimaxii'")

        with pytest.raises(TypeError) as ex:
            st.scaling = [2, 1]
        assert ex.value.args[0] == "'scaling' must be of type 'str'"

        for s in st._VALID_SCALINGS:
            st.scaling = s
        st.scaling = None

    def test_eq(self, n=200):
        """Test pre.SnapshotTransformer.__eq__()."""
        µ = np.random.randint(0, 100, (n,))
        a, b = 10, -3

        # Null transformers.
        st1 = opinf.pre.SnapshotTransformer()
        st2 = opinf.pre.SnapshotTransformer()
        assert st1 == st2
        assert st1 != 100

        # Mismatched attributes.
        st1.center = True
        st2.center = False
        assert not (st1 == st2)
        assert st1 != st2

        # Centering attributes.
        st1.mean_ = µ
        st2.center = True
        assert st1 != st2
        st2.mean_ = µ
        assert st1 == st2
        st2.mean_ = µ - 5
        assert st1 != st2

        # Scaling attributes.
        st1.scaling = "standard"
        st2.scaling = None
        assert st1 != st2
        st2.scaling = "minmax"
        assert st1 != st2
        st2.scaling = "standard"
        assert st1 == st2
        st1.scale_, st1.shift_ = a, b
        assert st1 != st2
        st2.scale_, st2.shift_ = a - 1, b + 1
        assert st1 != st2
        st2.scale_, st2.shift_ = a, b
        assert st1 == st2

    # Printing ----------------------------------------------------------------
    def test_str(self):
        """Test pre.SnapshotTransformer.__str__()."""
        st = opinf.pre.SnapshotTransformer()

        st.center = False
        st.scaling = None
        assert str(st) == "Snapshot transformer"

        st.center = True
        trn = "(call fit_transform() to train)"
        msc = "Snapshot transformer with mean-snapshot centering"
        assert str(st) == f"{msc} {trn}"
        for s in st._VALID_SCALINGS:
            st.scaling = s
            assert str(st) == f"{msc} and '{s}' scaling {trn}"

        st.center = False
        for s in st._VALID_SCALINGS:
            st.scaling = s
            assert str(st) == f"Snapshot transformer with '{s}' scaling {trn}"

    def test_statistics_report(self):
        """Test pre.SnapshotTransformer._statistics_report()."""
        X = np.arange(10) - 4
        report = opinf.pre.SnapshotTransformer._statistics_report(X)
        assert report == "-4.000e+00 |  5.000e-01 |  5.000e+00 |  2.872e+00"

    # Persistence -------------------------------------------------------------
    def test_save(self, n=200, k=50):
        """Test pre.SnapshotTransformer.save()."""
        # Clean up after old tests.
        target = "_savetransformertest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        def _checkfile(filename, st):
            assert os.path.isfile(filename)
            with h5py.File(filename, 'r') as hf:
                # Check transformation metadata.
                assert "meta" in hf
                assert len(hf["meta"]) == 0
                for attr in ("center", "scaling", "verbose"):
                    assert attr in hf["meta"].attrs
                    if attr == "scaling" and st.scaling is None:
                        assert not hf["meta"].attrs[attr]
                    else:
                        assert hf["meta"].attrs[attr] == getattr(st, attr)

                # Check transformation parameters.
                if st.center:
                    assert "transformation/mean_" in hf
                    assert np.all(hf["transformation/mean_"][:] == st.mean_)
                if st.scaling:
                    assert "transformation/scale_" in hf
                    assert hf["transformation/scale_"][0] == st.scale_
                    assert "transformation/shift_" in hf
                    assert hf["transformation/shift_"][0] == st.shift_

        # Check file creation and overwrite protocol on null transformation.
        st = opinf.pre.SnapshotTransformer()
        st.save(target[:-3])
        _checkfile(target, st)

        with pytest.raises(FileExistsError) as ex:
            st.save(target, overwrite=False)
        ex.value.args[0] == f"{target} (use overwrite=True to ignore)"

        st.save(target, overwrite=True)
        _checkfile(target, st)

        # Check non-null transformations.
        X = np.random.randint(0, 100, (n,k)).astype(float)
        for scaling, center in itertools.product({None, *st._VALID_SCALINGS},
                                                 (True, False)):
            st.center = center
            st.scaling = scaling
            st.fit_transform(X)
            st.save(target, overwrite=True)
            _checkfile(target, st)

        os.remove(target)

    def test_load(self, n=200, k=50):
        """Test pre.SnapshotTransformer.load()."""
        # Clean up after old tests.
        target = "_loadtransformertest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Try to load a bad file.
        with h5py.File(target, 'w'):
            pass

        with pytest.raises(ValueError) as ex:
            opinf.pre.SnapshotTransformer.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Check that save() -> load() gives the same transformer.
        st = opinf.pre.SnapshotTransformer()
        X = np.random.randint(0, 100, (n,k)).astype(float)
        for scaling, center in itertools.product({None, *st._VALID_SCALINGS},
                                                 (True, False)):
            st.scaling = scaling
            st.center = center
            st.fit_transform(X, inplace=False)
            st.save(target, overwrite=True)
            st2 = opinf.pre.SnapshotTransformer.load(target)
            assert st == st2

        os.remove(target)

    # Main routines -----------------------------------------------------------
    def test_is_trained(self):
        """Test pre.SnapshotTransformer._is_trained()."""
        st = opinf.pre.SnapshotTransformer()

        # Null transformer is always trained.
        st.center = False
        st.scaling = None
        assert st._is_trained() is True

        # Centering.
        st.center = True
        assert st._is_trained() is False
        st.mean_ = np.array([1,2,3])
        assert st._is_trained() is True

        # Scaling.
        st.center = False
        st.scaling = "minmax"
        assert st._is_trained() is False
        st.scale_ = 10
        assert st._is_trained() is False
        st.shift_ = 20
        assert st._is_trained() is True

    def test_fit_transform(self, n=200, k=50):
        """Test pre.SnapshotTransformer.fit_transform()."""

        def fit_transform_copy(st, A):
            """Assert A and B are not the same object but do have the same
            type and shape.
            """
            B = st.fit_transform(A, inplace=False)
            assert B is not A
            assert type(B) is type(A)
            assert B.shape == A.shape
            return B

        st = opinf.pre.SnapshotTransformer(verbose=True)

        # Test null transformation.
        st.center = False
        st.scaling = None
        X = np.random.randint(0, 100, (n,k)).astype(float)
        Y = st.fit_transform(X, inplace=True)
        assert Y is X
        Y = fit_transform_copy(st, X)
        assert np.all(Y == X)

        # Test centering.
        st.center = True
        st.scaling = None
        Y = fit_transform_copy(st, X)
        assert hasattr(st, "mean_")
        assert isinstance(st.mean_, np.ndarray)
        assert st.mean_.shape == (X.shape[0],)
        assert np.allclose(np.mean(Y, axis=1), 0)

        # Test scaling (without and with centering).
        for centering in (False, True):
            st.center = centering

            # Test standard scaling.
            st.scaling = "standard"
            Y = fit_transform_copy(st, X)
            for attr in "scale_", "shift_":
                assert hasattr(st, attr)
                assert isinstance(getattr(st, attr), float)
            assert np.isclose(np.mean(Y), 0)
            assert np.isclose(np.std(Y), 1)

            # Test min-max scaling.
            st.scaling = "minmax"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.min(Y), 0)
            assert np.isclose(np.max(Y), 1)

            # Test symmetric min-max scaling.
            st.scaling = "minmaxsym"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.min(Y), -1)
            assert np.isclose(np.max(Y), 1)

            # Test maximum absolute scaling.
            st.scaling = "maxabs"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.max(np.abs(Y)), 1)

            # Test minimum-maximum absolute scaling.
            st.scaling = "maxabssym"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.mean(Y), 0)
            assert np.isclose(np.max(np.abs(Y)), 1)

    def test_transform(self, n=200, k=50):
        """Test pre.SnapshotTransformer.transform()."""
        X = np.random.randint(0, 100, (n,k)).astype(float)
        st = opinf.pre.SnapshotTransformer(verbose=False)

        # Test null transformation.
        X = np.random.randint(0, 100, (n,k)).astype(float)
        st.fit_transform(X)
        Y = np.random.randint(0, 100, (n,k)).astype(float)
        Z = st.transform(Y, inplace=True)
        assert Z is Y
        Z = st.transform(Y, inplace=False)
        assert Z is not Y
        assert Z.shape == Y.shape
        assert np.all(Z == Y)

        # Test mean shift.
        st.center = True
        st.scaling = None
        with pytest.raises(AttributeError) as ex:
            st.transform(Y, inplace=False)
        assert ex.value.args[0] == \
            "transformer not trained (call fit_transform())"
        st.fit_transform(X)
        µ = st.mean_
        Z = st.transform(Y, inplace=False)
        assert np.allclose(Z, Y - µ.reshape(-1,1))

        # Test each scaling.
        st.center = False
        for scl in st._VALID_SCALINGS:
            X = np.random.randint(0, 100, (n,k)).astype(float)
            Y = np.random.randint(0, 100, (n,k)).astype(float)
            st.scaling = scl
            st.fit_transform(X)
            a, b = st.scale_, st.shift_
            Z = st.transform(Y)
            assert np.allclose(Z, a*Y + b)

    def test_inverse_transform(self, n=200, k=50):
        """Test pre.SnapshotTransformer.inverse_transform()."""
        X = np.random.randint(0, 100, (n,k)).astype(float)
        st = opinf.pre.SnapshotTransformer(verbose=False)

        st.center = True
        with pytest.raises(AttributeError) as ex:
            st.inverse_transform(X, inplace=False)
        assert ex.value.args[0] == \
            "transformer not trained (call fit_transform())"

        for scaling, center in itertools.product({None, *st._VALID_SCALINGS},
                                                 (True, False)):
            st.scaling = scaling
            st.center = center
            st.fit_transform(X, inplace=False)
            Y = np.random.randint(0, 100, (n,k)).astype(float)
            Z = st.transform(Y, inplace=False)
            st.inverse_transform(Z, inplace=True)
            assert np.allclose(Z, Y)


class TestSnapshotTransformerMulti:
    """Test pre.SnapshotTransformerMulti."""
    def test_init(self):
        """Test pre.SnapshotTransformer.__init__()."""
        stm = opinf.pre.SnapshotTransformerMulti(1)
        for attr in ["scaling", "center", "verbose",
                     "num_variables", "transformers"]:
            assert hasattr(stm, attr)

        # Center.
        stm = opinf.pre.SnapshotTransformerMulti(2, center=(True, True))
        assert stm.center == (True, True)
        stm.transformers[1].center = False
        assert stm.center == (True, False)
        stm = opinf.pre.SnapshotTransformerMulti(3, center=True)
        assert stm.center == (True, True, True)
        with pytest.raises(ValueError) as ex:
            opinf.pre.SnapshotTransformerMulti(3, center=[True, False])
        assert ex.value.args[0] == "len(center) = 2 != 3 = num_variables"
        with pytest.raises(ValueError) as ex:
            opinf.pre.SnapshotTransformerMulti(3, center="the center")
        assert ex.value.args[0] == "len(center) = 10 != 3 = num_variables"
        with pytest.raises(TypeError) as ex:
            opinf.pre.SnapshotTransformerMulti(3, center=100)
        assert ex.value.args[0] == "object of type 'int' has no len()"
        with pytest.raises(TypeError) as ex:
            opinf.pre.SnapshotTransformerMulti(2, center=(True, "yes"))
        assert ex.value.args[0] == "'center' must be True or False"

        # Scaling.
        stm = opinf.pre.SnapshotTransformerMulti(2, scaling=("minmax", None))
        assert stm.scaling == ("minmax", None)
        stm = opinf.pre.SnapshotTransformerMulti(2, scaling=[None, "maxabs"])
        assert isinstance(stm.scaling, tuple)
        assert stm.scaling == (None, "maxabs")
        stm = opinf.pre.SnapshotTransformerMulti(3, scaling="standard")
        assert stm.scaling == ("standard", "standard", "standard")
        with pytest.raises(TypeError) as ex:
            opinf.pre.SnapshotTransformerMulti(3, scaling=100)
        assert ex.value.args[0] == "object of type 'int' has no len()"
        with pytest.raises(ValueError) as ex:
            opinf.pre.SnapshotTransformerMulti(3, scaling=(True, False))
        assert ex.value.args[0] == "len(scaling) = 2 != 3 = num_variables"
        with pytest.raises(TypeError) as ex:
            opinf.pre.SnapshotTransformerMulti(2, scaling=(True, "minmax"))
        assert ex.value.args[0] == "'scaling' must be of type 'str'"

    # Properties --------------------------------------------------------------
    def test_properties(self):
        """Test pre.SnapshotTransformerMulti properties."""

        # Attribute setting blocked.
        stm = opinf.pre.SnapshotTransformerMulti(2)
        for attr in "num_variables", "center", "scaling":
            assert hasattr(stm, attr)
            with pytest.raises(AttributeError) as ex:
                setattr(stm, attr, 0)
            assert ex.value.args[0] == "can't set attribute"

        # Variable names.
        stm = opinf.pre.SnapshotTransformerMulti(3, variable_names=None)
        assert isinstance(stm.variable_names, list)
        assert len(stm.variable_names) == 3
        assert stm.variable_names == ["variable 1", "variable 2", "variable 3"]
        with pytest.raises(TypeError) as ex:
            stm.variable_names = (1, 2, 3)
        assert ex.value.args[0] == "variable_names must be list of length 3"
        with pytest.raises(TypeError) as ex:
            stm.variable_names = [1, 2]
        assert ex.value.args[0] == "variable_names must be list of length 3"
        stm.variable_names = ["Bill", "Charlie", "Percy"]

        # Verbose
        stm.verbose = 1
        assert stm.verbose is True
        stm.verbose = 0
        assert stm.verbose is False
        assert all(st.verbose is False for st in stm.transformers)
        stm.verbose = True
        assert stm.verbose is True

    def test_mean(self):
        """Test pre.SnapshotTransformerMulti.mean_."""
        centers = [False, True, False, True]
        scalings = [None, None, "standard", "minmax"]
        stm = opinf.pre.SnapshotTransformerMulti(4, centers, scalings)
        assert stm.mean_ is None

        # Set the centering vectors.
        stm.n_ = 7
        µs = [np.random.randint(0, 100, stm.n_) for _ in range(4)]
        for i,µ in enumerate(µs):
            if centers[i]:
                stm.transformers[i].mean_ = µ
            if scalings[i]:
                stm.transformers[i].scale_ = 0
                stm.transformers[i].shift_ = 0

        # Validate concatenated mean_.
        µµ = stm.mean_
        assert isinstance(µµ, np.ndarray)
        assert µµ.shape == (4*stm.n_,)
        for i,µ in enumerate(µs):
            s = slice(i*stm.n_, (i+1)*stm.n_)
            if centers[i]:
                assert np.allclose(µµ[s], µ)
            else:
                assert np.allclose(µµ[s], 0)

    def test_len(self):
        """Test pre.SnapshotTransformerMulti.__len__()."""
        for i in [2, 5, 10]:
            stm = opinf.pre.SnapshotTransformerMulti(i)
            assert len(stm) == i

    def test_getitem(self):
        """Test pre.SnapshotTransformerMulti.__getitem__()."""
        stm = opinf.pre.SnapshotTransformerMulti(10)
        for i in [3, 4, 7]:
            assert stm[i] is stm.transformers[i]

    def test_setitem(self):
        """Test pre.SnapshotTransformerMulti.__setitem__()."""
        stm = opinf.pre.SnapshotTransformerMulti(10)
        st = opinf.pre.SnapshotTransformer(center=True, scaling="minmax")
        for i in [3, 4, 7]:
            stm[i] = st
            assert stm.transformers[i] is st

        for i in range(stm.num_variables):
            stm[i].center = False
        assert stm.center == (False,)*stm.num_variables

        with pytest.raises(TypeError) as ex:
            stm[2] = 10
        assert ex.value.args[0] == \
            "assignment object must be SnapshotTransformer"

    def test_eq(self):
        """Test pre.SnapshotTransformerMulti.__eq__()."""
        # Null transformers.
        stm1 = opinf.pre.SnapshotTransformerMulti(3)
        assert stm1 != 100
        stm2 = opinf.pre.SnapshotTransformerMulti(2)
        assert stm1 != stm2
        stm2 = opinf.pre.SnapshotTransformerMulti(3)
        assert stm1 == stm2

        # Mismatched attributes.
        stm1 = opinf.pre.SnapshotTransformerMulti(3, center=True)
        stm2 = opinf.pre.SnapshotTransformerMulti(3, center=False)
        assert not (stm1 == stm2)
        assert stm1 != stm2

        stm1 = opinf.pre.SnapshotTransformerMulti(3, scaling="minmax")
        stm2 = opinf.pre.SnapshotTransformerMulti(3, scaling="minmax")
        assert stm1 == stm2
        st = opinf.pre.SnapshotTransformer(scaling="standard")
        stm1.transformers[1] = st
        assert stm1 != stm2
        stm2.transformers[1] = st
        assert stm1 == stm2

    def test_str(self):
        """Test pre.SnapshotTransformerMulti.__str__()."""
        names = ["var1", "var2", "var3"]
        stm = opinf.pre.SnapshotTransformerMulti(3, center=False,
                                                 scaling=None,
                                                 variable_names=names)
        stm.transformers[0].center = True
        stm.transformers[-1].scaling = "standard"

        assert str(stm) == \
            "Multi-variate snapshot transformer\n" \
            "* var1 | Snapshot transformer with mean-snapshot centering " \
            "(call fit_transform() to train)\n" \
            "* var2 | Snapshot transformer\n" \
            "* var3 | Snapshot transformer with 'standard' scaling " \
            "(call fit_transform() to train)"

    # Persistence -------------------------------------------------------------
    def test_save(self):
        """Test pre.SnapshotTransformerMulti.save()."""
        # Clean up after old tests.
        target = "_savetransformermultitest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        def _checkfile(filename, stm):
            assert os.path.isfile(filename)
            with h5py.File(filename, 'r') as hf:
                # Check transformation metadata.
                assert "meta" in hf
                assert len(hf["meta"]) == 0
                for attr in ("num_variables", "verbose"):
                    assert attr in hf["meta"].attrs
                    assert hf["meta"].attrs[attr] == getattr(stm, attr)

                # Check individual transformers.
                for i in range(stm.num_variables):
                    label = f"variable{i+1}"
                    assert label in hf
                    group = hf[label]
                    assert "meta" in group
                    assert "center" in group["meta"].attrs
                    assert "scaling" in group["meta"].attrs
                    st = stm.transformers[i]
                    assert group["meta"].attrs["center"] == st.center
                    if st.scaling is None:
                        assert not group["meta"].attrs["scaling"]
                        assert group["meta"].attrs["scaling"] is not None
                    else:
                        assert group["meta"].attrs["scaling"] == st.scaling

                    # Check transformation parameters.
                    if st.center:
                        assert "transformation/mean_" in group
                        assert np.all(
                            group["transformation/mean_"][:] == st.mean_)
                    if st.scaling:
                        assert "transformation/scale_" in group
                        assert group["transformation/scale_"][0] == st.scale_
                        assert "transformation/shift_" in group
                        assert group["transformation/shift_"][0] == st.shift_

        # Check file creation and overwrite protocol on null transformation.
        stm = opinf.pre.SnapshotTransformerMulti(15)
        stm.save(target[:-3])
        _checkfile(target, stm)

        with pytest.raises(FileExistsError) as ex:
            stm.save(target, overwrite=False)
        assert ex.value.args[0] == f"{target} (use overwrite=True to ignore)"

        stm.save(target, overwrite=True)
        _checkfile(target, stm)

        # Check non-null transformations.
        i = 0
        scalings = {None, *opinf.pre.SnapshotTransformer._VALID_SCALINGS}
        for center, scaling in itertools.product((True, False), scalings):
            stm.transformers[i].center = center
            stm.transformers[i].scaling = scaling
            i += 1
        X = np.random.randint(0, 100, (150,17)).astype(float)
        stm.fit_transform(X)
        stm.save(target, overwrite=True)
        _checkfile(target, stm)

        os.remove(target)

    def test_load(self, n=200, k=50):
        """Test pre.SnapshotTransformerMulti.load()."""
        # Clean up after old tests.
        target = "_loadtransformermultitest.h5"
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Try to load a bad file.
        with h5py.File(target, 'w'):
            pass

        with pytest.raises(ValueError) as ex:
            opinf.pre.SnapshotTransformerMulti.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Check that save() -> load() gives the same transformer.
        stm = opinf.pre.SnapshotTransformerMulti(15)
        i = 0
        scalings = {None, *opinf.pre.SnapshotTransformer._VALID_SCALINGS}
        for center, scaling in itertools.product((True, False), scalings):
            stm.transformers[i].center = center
            stm.transformers[i].scaling = scaling
            i += 1
        X = np.random.randint(0, 100, (150,19)).astype(float)
        stm.fit_transform(X)
        stm.save(target, overwrite=True)
        stm2 = opinf.pre.SnapshotTransformerMulti.load(target)
        assert stm2 == stm

        os.remove(target)

    # Main routines -----------------------------------------------------------
    def test_check_shape(self):
        """Test pre.SnapshotTransformerMulti._check_shape()."""
        stm = opinf.pre.SnapshotTransformerMulti(12)
        stm.n_ = 10
        X = np.random.randint(0, 100, (120,23)).astype(float)
        stm._check_shape(X)

        with pytest.raises(ValueError) as ex:
            stm._check_shape(X[:-1])
        assert ex.value.args[0] == \
            "snapshot set must have num_variables * n = 12 * 10 = 120 rows " \
            "(got 119)"

    def __testcase(self):
        centers = [
            False, True,
            False, False, False, False, False,
            True, True, True, True, True,
        ]
        scalings = [
            None, None,
            "standard", "minmax", "minmaxsym", "maxabs", "maxabssym",
            "standard", "minmax", "minmaxsym", "maxabs", "maxabssym"
        ]
        return opinf.pre.SnapshotTransformerMulti(len(centers),
                                                  center=centers,
                                                  scaling=scalings,
                                                  verbose=True)

    def test_fit_transform(self):
        """Test pre.SnapshotTransformerMulti.fit_transform()."""
        stm = self.__testcase()

        # Inplace transformation.
        X = np.random.randint(0, 100, (120,29)).astype(float)
        Y = stm.fit_transform(X, inplace=True)
        assert stm.n_ == 10
        assert stm._is_trained()
        assert Y is X

        # Non-inplace transformation.
        X = np.random.randint(0, 100, (120,29)).astype(float)
        Y = stm.fit_transform(X, inplace=False)
        assert stm.n_ == 10
        assert stm._is_trained()
        assert Y is not X
        assert type(Y) is type(X)
        assert Y.shape == X.shape

        # Null transformation.
        i = 0
        s = slice(i*stm.n_, (i+1)*stm.n_)
        assert np.allclose(Y[s], X[s])
        for attr in ("mean_", "scale_", "shift_"):
            assert not hasattr(stm.transformers[i], attr)

        # Centering only.
        i += 1
        s = slice(i*stm.n_, (i+1)*stm.n_)
        µ = np.mean(X[s], axis=1)
        assert np.allclose(Y[s], X[s] - µ.reshape(-1,1))
        assert hasattr(stm.transformers[i], "mean_")
        assert np.allclose(stm.transformers[i].mean_, µ)
        for attr in ("scale_", "shift_"):
            assert not hasattr(stm.transformers[i], attr)

        for ctr in [False, True]:
            # Standard scaling.
            i += 1
            s = slice(i*stm.n_, (i+1)*stm.n_)
            assert stm.transformers[i].scaling == "standard"
            assert np.isclose(np.mean(Y[s]), 0)
            assert np.isclose(np.std(Y[s]), 1)

            # Minmax scaling (to [0,1]).
            i += 1
            s = slice(i*stm.n_, (i+1)*stm.n_)
            assert stm.transformers[i].scaling == "minmax"
            assert np.isclose(np.min(Y[s]), 0)
            assert np.isclose(np.max(Y[s]), 1)

            # Symmetric Minmax scaling (to [-1,1]).
            i += 1
            s = slice(i*stm.n_, (i+1)*stm.n_)
            assert stm.transformers[i].scaling == "minmaxsym"
            assert np.isclose(np.min(Y[s]), -1)
            assert np.isclose(np.max(Y[s]), 1)

            # Symmetric Minmax scaling (to [-1,1]).
            i += 1
            s = slice(i*stm.n_, (i+1)*stm.n_)
            assert stm.transformers[i].scaling == "maxabs"
            assert np.isclose(np.max(np.abs(Y[s])), 1)

            # Symmetric Minmax scaling (to [-1,1]).
            i += 1
            s = slice(i*stm.n_, (i+1)*stm.n_)
            assert stm.transformers[i].scaling == "maxabssym"
            assert np.isclose(np.mean(Y[s]), 0)
            assert np.isclose(np.max(np.abs(Y[s])), 1)

    def test_transform(self):
        """Test pre.SnapshotTransformerMulti.transform()."""
        stm = self.__testcase()

        X = np.random.randint(0, 100, (120,29)).astype(float)
        with pytest.raises(AttributeError) as ex:
            stm.transform(X, inplace=False)
        assert ex.value.args[0] == \
            "transformer not trained (call fit_transform())"

        # Inplace transformation.
        stm.fit_transform(X)
        Y = np.random.randint(0, 100, (120,33)).astype(float)
        Z = stm.transform(Y, inplace=True)
        assert Z is Y

        # Non-inplace transformation.
        Y = np.random.randint(0, 100, (120,33)).astype(float)
        Z = stm.transform(Y, inplace=False)
        assert Z is not Y
        assert type(Z) is type(Y)
        assert Z.shape == Y.shape

    def test_inverse_transform(self):
        """Test pre.SnapshotTransformerMulti.transform()."""
        stm = self.__testcase()

        X = np.random.randint(0, 100, (120,29)).astype(float)
        with pytest.raises(AttributeError) as ex:
            stm.inverse_transform(X, inplace=False)
        assert ex.value.args[0] == \
            "transformer not trained (call fit_transform())"

        # Inplace transformation.
        stm.fit_transform(X)
        Y = np.random.randint(0, 100, (120,32)).astype(float)
        Z = stm.transform(Y, inplace=False)
        W = stm.inverse_transform(Z, inplace=True)
        assert W is Z

        Z = stm.transform(Y, inplace=False)
        W = stm.inverse_transform(Z, inplace=False)
        assert W is not Z
        assert np.allclose(W, Y)
