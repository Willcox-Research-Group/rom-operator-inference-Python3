# utils/test_hdf5.py
"""Tests for utils._hdf5."""

import os
import h5py
import pytest
import warnings
import numpy as np
import scipy.sparse as sparse

import opinf


def test_hdf5_filehandle():
    """Test utils._hdf5._hdf5_filehandle()."""
    subject = opinf.utils._hdf5._hdf5_filehandle

    # Clean up after old tests.
    target = "_hdf5handletest.h5"
    if os.path.isfile(target):  # pragma: no cover
        os.remove(target)

    # Input file is already an open h5py handle.
    h5file = h5py.File(target, "a")
    with subject(h5file, "load", False) as hf:
        assert hf is h5file
        assert bool(hf)
        assert hf.mode == "r+"
    assert bool(hf)  # check the file is still open.
    hf.close()
    os.remove(target)

    # Save mode without .h5 extension.
    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        with subject(target[:-3], "save", True):
            pass
    assert len(wn) == 1
    assert wn[0].message.args[0] == "expected file with extension '.h5'"
    os.remove(target[:-3])

    # Save mode with .h5 extension.
    with subject(target, "save", True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert os.path.isfile(target)
    assert not bool(hf)  # check the file is closed.

    # Try to overwrite but with overwrite=False.
    with pytest.raises(FileExistsError) as ex:
        with subject(target, "save", overwrite=False):
            pass
    assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

    # Overwrite.
    with subject(target, "save", overwrite=True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert not bool(hf)
    assert os.path.isfile(target)

    # Loading.
    with subject(target, "load", False) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r"
    assert not bool(hf)
    hf.close()
    os.remove(target)

    # Try loading a nonexistent file.
    with pytest.raises(FileNotFoundError) as ex:
        with subject(target, "load", overwrite=False):
            pass
    assert ex.value.args[0] == target

    # Invalid mode.
    with pytest.raises(ValueError) as ex:
        with subject(target, "moose", None):
            pass
    assert ex.value.args[0] == "invalid mode 'moose'"

    # Exception happens within block.
    with pytest.raises(RuntimeError) as ex:
        with subject(target, "save", overwrite=True) as hf:
            raise RuntimeError("error within block")
    assert ex.value.args[0] == "error within block"
    assert not bool(hf)
    os.remove(target)


def test_hdf5_savehandle():
    """Test utils._hdf5.hdf5_savehandle()."""
    subject = opinf.utils.hdf5_savehandle

    # Clean up after old tests.
    target = "_hdf5savehandletest.h5"
    if os.path.isfile(target):  # pragma: no cover
        os.remove(target)

    # Input file is already an open h5py handle.
    h5file = h5py.File(target, "a")
    with subject(h5file, False) as hf:
        assert hf is h5file
        assert bool(hf)
        assert hf.mode == "r+"
    assert bool(hf)  # check file is still open.
    hf.close()
    os.remove(target)

    # Save mode without .h5 extension.
    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        with subject(target[:-3], True):
            pass
    assert len(wn) == 1
    assert wn[0].message.args[0] == "expected file with extension '.h5'"
    os.remove(target[:-3])

    # Save mode with .h5 extension.
    with subject(target, True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert not bool(hf)
    assert os.path.isfile(target)

    # Try to overwrite but with overwrite=False.
    with pytest.raises(FileExistsError) as ex:
        with subject(target, overwrite=False):
            pass
    assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

    # Overwrite.
    with subject(target, overwrite=True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert not bool(hf)
    assert os.path.isfile(target)
    os.remove(target)


def test_hdf5_loadhandle():
    """Test utils._hdf5.hdf5_loadhandle()."""
    subject = opinf.utils.hdf5_loadhandle

    # Clean up after old tests.
    target = "_hdf5loadhandletest.h5"
    if os.path.isfile(target):  # pragma: no cover
        os.remove(target)

    with h5py.File(target, "w"):
        pass

    # Loading.
    with subject(target) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r"
    assert not bool(hf)

    # Exception within block is wrapped as LoadfileFormatError.
    with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
        with subject(target) as hf:
            raise RuntimeError("error within block")
    assert ex.value.args[0] == "error within block"

    with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
        with subject(target) as hf:
            raise opinf.errors.LoadfileFormatError("error2")
    assert ex.value.args[0] == "error2"

    class DummyWarning(Warning):
        pass

    # Warning within block is passed on.
    with pytest.warns(DummyWarning) as wn:
        with subject(target) as hf:
            warnings.warn("my dummy warning", DummyWarning)
    assert wn[0].message.args[0] == "my dummy warning"

    # Try loading a nonexistent file.
    os.remove(target)
    with pytest.raises(FileNotFoundError) as ex:
        with subject(target):
            pass
    assert ex.value.args[0] == target


def test_saveload_sparray(n=100, target="_saveloadsparraytest.h5"):
    """Test save_sparray() and load_sparray()."""

    with pytest.raises(TypeError) as ex:
        opinf.utils.save_sparray(None, None)
    assert ex.value.args[0] == "second arg must be a scipy.sparse array"

    A = sparse.dok_array((n, n), dtype=float)
    for _ in range(n // 10):
        i, j = np.random.randint(0, n, size=2)
        A[i, j] = np.random.random()

    if os.path.isfile(target):
        os.remove(target)

    with h5py.File(target, "w") as hf:
        opinf.utils.save_sparray(hf.create_group("sparsearray"), A)

    with h5py.File(target, "r") as hf:
        B = opinf.utils.load_sparray(hf["sparsearray"])

    diff = np.abs((A - B).data)
    assert np.allclose(diff, 0)
